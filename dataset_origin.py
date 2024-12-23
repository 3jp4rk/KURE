
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
class KoE5Dataset(Dataset):

  args: KoE5DataTrainingArguments
  features: List[InputFeatures]
  
  def __init__(
      self,
      args: KoE5DataTrainingArguments,
      tokenizer: PreTrainedTokenizerBase,
      limit_length: Optional[int] = None,
    #   mode: Union[str, Split] = Split.train,
      mode: Union[str, Split] = "train",
      cache_dir: Optional[str] = None,
      test: Optional[bool] = False,
  ):
    self.args = args
    self.processor = KoE5MRCProcessor()
    # if isinstance(mode, str):
    #     try:
    #         mode = Split
    #     except KeyError:
    #         raise KeyError("mode is not a valid split name")

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        cache_dir if cache_dir is not None else args.data_dir,
        f"cached_{mode.value}_{tokenizer.__class__.__name__}_{args.max_seq_length}",
    )

    print(f"cache file exits: {os.path.exists(cached_features_file)}")

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Start loading features...")
        start = time.time()
        self.features = torch.load(cached_features_file)
        print(
            f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
        )

    else:
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            print(f"No cache files. Creating features from dataset file at {args.data_dir}")
            if mode == Split.dev:
                examples = self.processor.get_dev_examples(args.data_dir)
            elif mode == Split.test:
                examples = self.processor.get_test_examples(args.data_dir)
            else:
                examples = self.processor.get_train_examples(args.data_dir)

            if test:
                examples = examples[:100]
                print("Test mode activated: Got 100 examples!")

            self.features = self.processor.convert_examples_to_features(examples, tokenizer, args.max_seq_length)
            print("Converted examples to features!")

            start = time.time()
            torch.save(self.features, cached_features_file)
            print(
                f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
            )
  
  def __len__(self):
      return len(self.features)
  
  def __getitem__(self, i) -> InputFeatures:
      return self.features[i]
  
  def get_labels(self):
      return [1, 0]