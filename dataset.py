
import torch
from processor import KoE5MRCProcessor
from arguments import KoE5DataTrainingArguments
from processor import InputFeatures
from typing import Union, Optional, List
from transformers import DataProcessor, InputExample, PreTrainedTokenizerBase
import os
import time
from filelock import FileLock
from datasets import Split, Dataset


# dataset.py
class KoE5Dataset:
    def __init__(
        self,
        dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_length = max_seq_length if max_seq_length is not None else tokenizer.model_max_length
        self.features = self._convert_to_features()

    def _convert_to_features(self):
        features = []
        for example in self.dataset:
            query = example["query"]
            document = example["document"]
            hard_negatives = example["hard_negative"]

            query_encoding = self.tokenizer(
                query, max_length=self.max_seq_length, padding="max_length", truncation=True
            )
            document_encoding = self.tokenizer(
                document, max_length=self.max_seq_length, padding="max_length", truncation=True
            )
            negative_encoding = self.tokenizer(
                hard_negatives, max_length=self.max_seq_length, padding="max_length", truncation=True
            )

            features.append({
                "query_input_ids": query_encoding["input_ids"],
                "query_attention_mask": query_encoding["attention_mask"],
                "document_input_ids": document_encoding["input_ids"],
                "document_attention_mask": document_encoding["attention_mask"],
                "hard_negative_input_ids": negative_encoding["input_ids"],
                "hard_negative_attention_mask": negative_encoding["attention_mask"],
            })
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]
