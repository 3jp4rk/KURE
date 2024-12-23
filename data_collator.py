from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, PaddingStrategy, pad_without_fast_tokenizer_warning
from typing import Union, Optional

# data_collator.py
class DataCollatorForKoE5(DataCollatorMixin):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def torch_call(self, features):
        
        # print("torch_call invoked with features: \n", features)
        
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # 정규 형식 (input_ids, attention_mask 로 처리한 뒤 다시 prefix (query, document, hard_negative) 붙여줌
        cat_batch = {}
        for k in ['query', 'document', 'hard_negative']:
            batch_features = [
                {
                    'input_ids': feature[f'{k}_input_ids'],
                    'attention_mask': feature[f'{k}_attention_mask']
                } for feature in features if f'{k}_input_ids' in feature and f'{k}_attention_mask' in feature
            ]

            if batch_features:
                batch = pad_without_fast_tokenizer_warning(
                    self.tokenizer,
                    batch_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )

                cat_batch[f'{k}_input_ids'] = batch['input_ids']
                cat_batch[f'{k}_attention_mask'] = batch['attention_mask']
        
        if labels is None:
            return cat_batch
