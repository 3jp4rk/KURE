from dataclasses import dataclass, field
import dataclasses
from typing import Union, Optional, List
# import jsonschema
import json
from transformers import DataProcessor, InputExample, PreTrainedTokenizer
import os
# processor.py
class E5InputExample(InputExample):
    def __init__(self, query: str, positive_passage: str, negative_passage: str):
        super().__init__(
            f"query: {query}", 
            f"passage: {positive_passage}",
            f"passage: {negative_passage}"
        )

@dataclass(frozen=True)
class InputFeatures:
    # question
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class KoE5MRCProcessor(DataProcessor):
    """Processor for the KoE5 dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        if "train.json" in os.listdir(data_dir):
            return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")
        else:
            return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def _create_examples(self, datas, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, data in enumerate(datas):
            if isinstance(data["query"], list):
                query = data["query"][0]
            else:
                query = data["query"]
            if isinstance(data["document"], list):
                document = data["document"][0]
            else:
                document = data["document"]
            if "hard_negative" in data:
                if isinstance(data["hard_negative"], list):
                    hard_negative = data["hard_negative"][0]
                else:
                    hard_negative = data["hard_negative"]
            else:
                hard_negative = None
    
            examples.append(E5InputExample(
                query=query,
                positive_passage=document,
                negative_passage=hard_negative,
            ))
    
        return examples

    @classmethod
    def _read_json(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _read_jsonl(file_path, input_file):
        """Read a JSONL file and return a list of dictionaries."""
        data = []
        with open(input_file, 'r', encoding='utf-8-sig') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    
    def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    ):
        if max_length is None:
            max_length = tokenizer.model_max_length

        query_batch_encoding = tokenizer(
            [example.query for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        document_batch_encoding = tokenizer(
            [example.positive_passage for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        negative_batch_encoding = tokenizer(
            [example.negative_passage for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        from tqdm import tqdm
        for i in tqdm(range(len(examples)), desc="Converting examples to features..."):
            query_inputs = {k: query_batch_encoding[k][i] for k in query_batch_encoding}
            doc_inputs = {k: document_batch_encoding[k][i] for k in document_batch_encoding}
            neg_inputs = {k: negative_batch_encoding[k][i] for k in negative_batch_encoding}

            features.append({
                'query_input_ids': query_inputs['input_ids'],
                'query_attention_mask': query_inputs['attention_mask'],
                'document_input_ids': doc_inputs['input_ids'],
                'document_attention_mask': doc_inputs['attention_mask'],
                'hard_negative_input_ids': neg_inputs['input_ids'],
                'hard_negative_attention_mask': neg_inputs['attention_mask'],
            })

        for i, example in enumerate(examples[:3]):
            print("*** Example ***")
            print(f"features: {features[i]}")

        return features