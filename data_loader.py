from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class _PG19IterableDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int,
        text_column: str,
        max_samples: Optional[int],
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        iterator = iter(self.dataset)
        if self.max_samples:
            iterator = itertools.islice(iterator, self.max_samples)

        for example in iterator:
            text = example[self.text_column]
            encoded = self.tokenizer(text, add_special_tokens=getattr(self.tokenizer, "add_bos_token", False))
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            usable_length = (len(input_ids) // self.block_size) * self.block_size
            for index in range(0, usable_length, self.block_size):
                block_ids = input_ids[index : index + self.block_size]
                block_mask = attention_mask[index : index + self.block_size]
                yield {
                    "input_ids": block_ids,
                    "attention_mask": block_mask,
                    "labels": block_ids[:],
                }


@dataclass
class PG19BlockDatasetBuilder:
    model_name_or_path: Union[str, PreTrainedTokenizerBase]
    split: str = "train"
    block_size: int = 512
    dataset_name: str = "emozilla/pg19"
    cache_dir: Optional[str] = None
    num_proc: Optional[int] = 80
    download_max_workers: Optional[int] = 128
    max_samples: Optional[int] = None

    # True: concatenate multiple samples before splitting into 512-token blocks
    # False: split each sample independently
    pack_across_examples: bool = False

    def build(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        if self.pack_across_examples:
            raise ValueError("pack_across_examples must be False for streaming dataset build.")

        tokenizer = self._create_tokenizer()

        if self.download_max_workers:
            snapshot_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                max_workers=self.download_max_workers,
            )

        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=True,
        )

        text_column = self._get_text_column(dataset)
        if world_size and rank is not None and hasattr(dataset, "shard"):
            dataset = dataset.shard(num_shards=world_size, index=rank)

        return _PG19IterableDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            block_size=self.block_size,
            text_column=text_column,
            max_samples=self.max_samples,
        )

    def _create_tokenizer(self) -> PreTrainedTokenizerBase:
        if isinstance(self.model_name_or_path, PreTrainedTokenizerBase):
            tokenizer = self.model_name_or_path
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if isinstance(self.model_name_or_path, str) and "llama-2" in self.model_name_or_path.lower():
            tokenizer.add_bos_token = True
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_text_column(self, dataset) -> str:
        if "text" in dataset.column_names:
            return "text"
        if "document" in dataset.column_names:
            return "document"
        raise ValueError(f"PG19 dataset missing expected text column: {dataset.column_names}")
