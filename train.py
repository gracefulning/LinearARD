### 注意力对齐
import argparse
import json
import sys
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface, AutoConfig
from transformers import default_data_collator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kernels.attention_KL.integration import attn_kl_align
from data_loader import PG19BlockDatasetBuilder
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _rope_scaling_json(value):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON for --student-rope-scaling: {value}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--student-rope-scaling must be a JSON object")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Self-distillation training")

    required_group = parser.add_argument_group("Required hyperparameters (no defaults)")
    required_group.add_argument("--model-id", required=True, type=str)
    required_group.add_argument("--tokenizer-id", required=True, type=str)
    required_group.add_argument("--student-rope-scaling", required=True, type=_rope_scaling_json)
    required_group.add_argument("--num-gpus", required=True, type=int)
    required_group.add_argument("--ctx-len", required=True, type=int)
    required_group.add_argument("--batch-size", required=True, type=int)
    required_group.add_argument("--grad-accum", required=True, type=int)
    required_group.add_argument("--steps", required=True, type=int)
    required_group.add_argument("--base-lr", required=True, type=float)

    default_group = parser.add_argument_group("Default hyperparameters")
    default_group.add_argument("--writer-dir", type=str, default="checkpoints/tflogs")
    default_group.add_argument("--tune-student-qkv-only", type=_str2bool, default=True)
    default_group.add_argument("--tune-student-qkv-lora", type=_str2bool, default=True)
    default_group.add_argument("--lora-rank", type=int, default=1024)
    default_group.add_argument("--lora-alpha", type=int, default=2048)
    default_group.add_argument("--lora-dropout", type=float, default=0.05)
    default_group.add_argument("--use-gradient-checkpointing", type=_str2bool, default=True)
    default_group.add_argument("--warmup-steps", type=int, default=4)
    default_group.add_argument("--min-lr-ratio", type=float, default=0.9)
    default_group.add_argument("--save-every-steps", type=int, default=480)
    default_group.add_argument("--ddp-init-method", type=str, default="tcp://127.0.0.1:29501")
    default_group.add_argument("--save-step-ckpt-prefix", type=str, default="checkpoints/attn_align_step_")
    default_group.add_argument("--save-final-ckpt-dir", type=str, default="checkpoints/attn_align_final")
    default_group.add_argument("--temperature", type=float, default=1.0)
    default_group.add_argument("--lam-l", type=float, default=0.0)
    default_group.add_argument("--lam-a", type=float, default=1.0)
    default_group.add_argument("--logits-keep-k", type=int, default=512)
    default_group.add_argument("--max-grad-norm", type=float, default=5.0)

    return parser.parse_args()


##### Q/K/V 缓存：保存每层的 Q/K/V，用于外部 KL 计算
class QKVCache:
    def __init__(self, keep_s=None, keep_t=None):
        # 只缓存需要对齐的层，降低显存占用
        self.keep_s = set(keep_s or [])
        self.keep_t = set(keep_t or [])
        # 用字典按层索引保存，避免存下所有层
        self.qt = {}
        self.kt = {}
        self.vt = {}
        self.qs = {}
        self.ks = {}
        self.vs = {}

    def clear_teacher(self):
        self.qt.clear()
        self.kt.clear()
        self.vt.clear()

    def clear_student(self):
        self.qs.clear()
        self.ks.clear()
        self.vs.clear()


###### 自定义注意力实现：在不返回权重的情况下采集 Q/K/V
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

flash_attention_forward = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]


def record_qkv_attention(
    module,
    query, key, value,
    attention_mask=None,
    is_causal=True,
    qkv_cache: QKVCache = None,
    mode: str = None,          # "teacher" or "student"
    **kwargs
):
    attn_output, _ = flash_attention_forward(module, query, key, value, attention_mask, **kwargs)
    # 依赖模块的 layer_idx 来判断是否需要缓存
    layer_idx = getattr(module, "layer_idx", None)
    if qkv_cache is not None and mode is not None:
        if mode == "teacher":
            # teacher 只需要前向，无需梯度
            if layer_idx is not None and layer_idx in qkv_cache.keep_t:
                # 只缓存被选中的层
                qkv_cache.qt[layer_idx] = query.detach().contiguous()
                qkv_cache.kt[layer_idx] = key.detach().contiguous()
                qkv_cache.vt[layer_idx] = value.detach().contiguous()
        elif mode == "student":
            # student 需要保留梯度
            if layer_idx is not None and layer_idx in qkv_cache.keep_s:
                # 只缓存被选中的层
                qkv_cache.qs[layer_idx] = query.contiguous()
                qkv_cache.ks[layer_idx] = key.contiguous()
                qkv_cache.vs[layer_idx] = value.contiguous()
    return attn_output, None


AttentionInterface.register("record_qkv", record_qkv_attention)


def build_dataloader(tokenizer, block_size, batch_size, rank, world_size):
    builder = PG19BlockDatasetBuilder(
        model_name_or_path=tokenizer,
        block_size=block_size,
        split="train",
        pack_across_examples=False,
    )
    training_dataset = builder.build(rank=rank, world_size=world_size)
    return DataLoader(
        training_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        drop_last=True,
    )


def train_ddp(local_rank, world_size, args):
    writer = SummaryWriter(args.writer_dir)
    model_id = args.model_id
    tokenizer_id = args.tokenizer_id
    student_rope_scaling = args.student_rope_scaling
    tune_student_qkv_only = args.tune_student_qkv_only
    tune_student_qkv_lora = args.tune_student_qkv_lora
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    use_gradient_checkpointing = args.use_gradient_checkpointing
    warmup_steps = args.warmup_steps
    min_lr_ratio = args.min_lr_ratio
    save_every_steps = args.save_every_steps
    ddp_init_method = args.ddp_init_method
    save_step_ckpt_prefix = args.save_step_ckpt_prefix
    save_final_ckpt_dir = args.save_final_ckpt_dir
    T = args.temperature
    lam_L = args.lam_l
    lam_A = args.lam_a
    logits_keep_k = args.logits_keep_k
    max_grad_norm = args.max_grad_norm

    dist.init_process_group(
        backend="nccl",
        init_method=ddp_init_method,
        world_size=world_size,
        rank=local_rank,
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ------------载入教师模型-----------
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="record_qkv",
    ).to(device)
    teacher_model.config.head_dim = teacher_model.config.hidden_size // teacher_model.config.num_attention_heads

    # ------------载入学生模型-----------
    student_config = None
    if student_rope_scaling is not None:
        student_config = AutoConfig.from_pretrained(model_id)
        rope_cfg = dict(student_rope_scaling)
        if "rope_type" in rope_cfg and "type" not in rope_cfg:
            rope_cfg["type"] = rope_cfg.pop("rope_type")
        student_config.rope_scaling = rope_cfg
    student_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="record_qkv",
        config=student_config,
    ).to(device)
    student_model.config.head_dim = student_model.config.hidden_size // student_model.config.num_attention_heads
    student_model.to(device)

    ### 设置 head_dim
    def _get_head_dim(m):
        cfg = m.config
        if hasattr(cfg, "head_dim") and cfg.head_dim is not None:
            return int(cfg.head_dim)
        return int(cfg.hidden_size // cfg.num_attention_heads)

    sm_scale_t = (_get_head_dim(teacher_model) ** -0.5)
    sm_scale_s = (_get_head_dim(student_model) ** -0.5)

    #输出教师和学生模型的结构以及对齐配置
    def _model_stats(m):
        cfg = m.config
        num_layers = int(getattr(cfg, "num_hidden_layers", 0))
        num_heads = int(getattr(cfg, "num_attention_heads", 0))
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        params = sum(p.numel() for p in m.parameters())
        return num_layers, num_heads, head_dim, params
    t_layers, t_heads, t_head_dim, t_params = _model_stats(teacher_model)
    s_layers, s_heads, s_head_dim, s_params = _model_stats(student_model)
    if local_rank == 0:
        print(
            f"[teacher] layers={t_layers} heads={t_heads} head_dim={t_head_dim} params={t_params}"
        )
        print(
            f"[student] layers={s_layers} heads={s_heads} head_dim={s_head_dim} params={s_params}"
        )
    if t_layers != s_layers:
        raise ValueError(f"Self-distillation requires matching layer counts, got teacher={t_layers}, student={s_layers}")
    align_layer_indices = tuple(range(t_layers))
    if local_rank == 0:
        print(f"[align] sequential all-layer alignment enabled, num_layers={len(align_layer_indices)}")

    teacher_model.eval()  # teacher 固定
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    student_model.train()  # student 训练
    if tune_student_qkv_only == True and tune_student_qkv_lora == False:
        for p in student_model.parameters():
            p.requires_grad_(False)
        for n, p in student_model.named_parameters():
            # 兼容常见命名：q_proj/k_proj/v_proj；以及部分模型的 query_key_value / Wqkv 合并写法
            if ("q_proj" in n) or ("k_proj" in n) or ("v_proj" in n) or ("query_key_value" in n) or ("Wqkv" in n) or ("wqkv" in n):
                p.requires_grad_(True)
    elif tune_student_qkv_only == True and tune_student_qkv_lora == True:
        for p in student_model.parameters():  # 先全冻结
            p.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","query_key_value","Wqkv","wqkv"],
        )
        student_model = get_peft_model(student_model, lora_cfg)
        student_model.print_trainable_parameters()
    else:
        for p in student_model.parameters():
            p.requires_grad_(True)

    if use_gradient_checkpointing:  # 梯度检查点（放在 LoRA 之后，确保作用于实际前向路径）
        if hasattr(student_model, "base_model"):
            student_model.base_model.model.gradient_checkpointing_enable({"use_reentrant": False})
            student_model.base_model.model.config.use_cache = False
            student_model.base_model.model.enable_input_require_grads()
        else:
            student_model.gradient_checkpointing_enable({"use_reentrant": False})
            student_model.config.use_cache = False
            student_model.enable_input_require_grads()

    ctx_len = int(args.ctx_len)
    batch_size = int(args.batch_size)
    grad_accum = int(args.grad_accum)
    total_steps = int(args.steps)
    base_lr = float(args.base_lr)

    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW((p for p in student_model.parameters() if p.requires_grad), lr=base_lr)
    # Cache all layers for sequential alignment in self-distillation
    qkv_cache = QKVCache(keep_s=align_layer_indices, keep_t=align_layer_indices)
    global_step = 0

    # 按固定训练配置构建 dataloader
    dataloader = build_dataloader(tok, ctx_len, batch_size, local_rank, world_size)
    if getattr(dataloader, "sampler", None) is not None and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(0)
    data_iter = iter(dataloader)

    for step in range(total_steps):
        step_start = time.time()
        # 全局学习率调度（只 warmup 一次、退火一次）
        cur_step = float(global_step)
        if cur_step < warmup_steps:
            lr = base_lr * (cur_step / max(1.0, warmup_steps))
        else:
            progress = (cur_step - warmup_steps) / max(1.0, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 梯度累计：多个 micro-batch 合并一次更新
        for micro in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # 准备输入
            inputs2 = {k: v.to(device) for k, v in batch.items()}
            inputs2["use_cache"] = False

            # 采集 Q/K/V（teacher 不需要梯度）
            qkv_cache.clear_teacher()
            qkv_cache.clear_student()
            with torch.no_grad():
                t_inputs = {k: v for k, v in inputs2.items() if k != "labels"}
                t_out = teacher_model(
                    **t_inputs,
                    qkv_cache=qkv_cache,
                    mode="teacher",
                    logits_to_keep=logits_keep_k,
                    return_dict=True,
                )
                t_logits = t_out.logits  # [B, K, V]
            # 采集 Q/K/V（student 需要梯度）
            sync_ctx = student_model.no_sync() if micro < grad_accum - 1 else torch.enable_grad()
            with sync_ctx:
                s_inputs = {k: v for k, v in inputs2.items() if k != "labels"}
                s_out = student_model(
                    **s_inputs,
                    qkv_cache=qkv_cache,
                    mode="student",
                    logits_to_keep=logits_keep_k,
                    return_dict=True,
                )
                s_logits = s_out.logits  # [B, K, V]

                # 计算注意力 KL
                Q_relation_kl = 0.0
                K_relation_kl = 0.0
                V_relation_kl = 0.0
                L = len(align_layer_indices)
                attention_mask_full = inputs2.get("attention_mask", None)

                ## -------------- loss函数部分 --------------
                #### -----------注意力对齐--------
                for layer_idx in align_layer_indices:
                    qs = qkv_cache.qs[layer_idx]
                    ks = qkv_cache.ks[layer_idx]
                    vs = qkv_cache.vs[layer_idx]
                    qt = qkv_cache.qt[layer_idx]
                    kt = qkv_cache.kt[layer_idx]
                    vt = qkv_cache.vt[layer_idx]

                    attention_mask = attention_mask_full

                    # 计算Q-Q K-K
                    layer_q_kl = attn_kl_align(
                        qs, qs, qt, qt,
                        attn_mask=attention_mask,
                        causal=True,
                        sm_scale_s=sm_scale_s,
                        sm_scale_t=sm_scale_t,
                    )
                    layer_k_kl = attn_kl_align(
                        ks, ks, kt, kt,
                        attn_mask=attention_mask,
                        causal=True,
                        sm_scale_s=sm_scale_s,
                        sm_scale_t=sm_scale_t,
                    )

                    #计算V-V
                    layer_v_kl = attn_kl_align(
                        vs, vs, vt, vt,
                        attn_mask=attention_mask,
                        causal=True,
                        sm_scale_s=sm_scale_s,
                        sm_scale_t=sm_scale_t,
                    )
                    if attention_mask is None:
                        denom = qs.shape[0] * qs.shape[1] * qs.shape[2]
                    else:
                        denom = attention_mask.sum().clamp_min(1) * qs.shape[1]
                    Q_relation_kl = Q_relation_kl + layer_q_kl / denom
                    K_relation_kl = K_relation_kl + layer_k_kl / denom
                    V_relation_kl = V_relation_kl + layer_v_kl / denom
                attn_loss = (Q_relation_kl + K_relation_kl + V_relation_kl) / max(1, L)

                #### -----------输出对齐--------
                t_logp = F.log_softmax(t_logits / T, dim=-1)   # [B, K, V]
                s_logp = F.log_softmax(s_logits / T, dim=-1)   # [B, K, V]
                t_p    = t_logp.exp()                                  # [B, K, V]

                kl_per_tok = (t_p * (t_logp - s_logp)).sum(dim=-1)     # [B, K]

                logits_kl = kl_per_tok.mean()                          # 标量
                logits_loss = logits_kl


                loss = lam_A * attn_loss + lam_L * logits_loss
                (loss / grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=max_grad_norm)
        grad_norm_to_log = None
        if local_rank == 0:
            grad_norm_to_log = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=float("inf"))

        # 更新一次参数
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 定期保存
        if local_rank == 0 and global_step % save_every_steps == 0 and global_step != 0:
            student_model.module.save_pretrained(f"{save_step_ckpt_prefix}{global_step}")

        # 日志与可视化
        if local_rank == 0:
            print(
                "step:", step,
                "gstep:", global_step,
                "loss:", float(loss.detach().cpu()),
                "attn:", float(attn_loss.detach().cpu()),
                "logits:", float(logits_loss.detach().cpu()),
                "grad_norm:", float(grad_norm_to_log.detach().cpu()),
                "lr:", f"{lr:.5f}",
                "step_time_s:", f"{(time.time() - step_start):.2f}",
            )
            writer.add_scalar("train/total", float(loss.detach().cpu()), global_step)
            writer.add_scalar("train/attn", float((attn_loss).detach().cpu()), global_step)
            writer.add_scalar("train/logits", float(logits_loss.detach().cpu()), global_step)
            writer.add_scalar("train/grad_norm", float(grad_norm_to_log.detach().cpu()), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/Q_loss", float((Q_relation_kl).detach().cpu()), global_step)
            writer.add_scalar("train/K_loss", float((K_relation_kl).detach().cpu()), global_step)
            writer.add_scalar("train/V_loss", float((V_relation_kl).detach().cpu()), global_step)
        global_step += 1

    # 最终保存
    if local_rank == 0:
        student_model.module.save_pretrained(save_final_ckpt_dir)
    writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    if args.num_gpus <= 1:
        train_ddp(0, 1, args)
    else:
        mp.spawn(train_ddp, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)
