keep



from transformers import AutoTokenizer, AutoModel, HfArgumentParser,TrainingArguments
from arguments import ModelArguments, DataArguments, KoE5DataTrainingArguments
from dataset import KoE5Dataset
from data_collator import DataCollatorForKoE5
from processor import KoE5MRCProcessor
# from trainer import CustomTrainer
import torch
import os
from datasets import load_dataset

import torch
from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn


def average_pool(hidden_states, attention_mask):
    # Use the attention mask to compute the sum of hidden states
    sum_hidden_states = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)    
    # Compute average, but exclude padding tokens
    avg_hidden_states = sum_hidden_states / attention_mask.sum(dim=1, keepdim=True)
    return avg_hidden_states


class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, data_collator):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         tokenizer=tokenizer, data_collator=data_collator)


    def compute_loss(self, model, inputs, return_outputs=False):
        
        embeddings = {}
  
        for k in ['query', 'document', 'hard_negative']:
            input_ids = inputs[f'{k}_input_ids']
            attention_mask = inputs[f'{k}_attention_mask']
    
            input_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': torch.zeros_like(input_ids)
            }
    
            output: BaseModelOutput = model(**input_dict)
            pooled_output = average_pool(output.last_hidden_state, input_dict['attention_mask'])
            embeddings[k] = pooled_output

        # LogSumExp loss applied 
        query_embeddings = embeddings['query']
        positive_embeddings = embeddings['document']
        negative_embeddings = embeddings['hard_negative']

        similarity_fct = nn.CosineSimilarity(dim=-1)
        tau = 0.02

        positive_scores = similarity_fct(query_embeddings, positive_embeddings) / tau
        negative_scores = similarity_fct(query_embeddings.unsqueeze(1), negative_embeddings) / tau

        max_positive_scores = torch.max(positive_scores, dim=0, keepdim=True)[0]
        max_negative_scores = torch.max(negative_scores, dim=1, keepdim=True)[0]
        max_scores = torch.max(max_positive_scores, max_negative_scores) # max_score을 구하여

        stable_positive_scores = positive_scores - max_scores # max_score을 빼줌
        stable_negative_scores = negative_scores - max_scores.unsqueeze(1) # 여기서도 max_score을 빼줌

        exp_positive_scores = torch.exp(stable_positive_scores)
        exp_negative_scores = torch.exp(stable_negative_scores)

        total_scores_sum = exp_positive_scores + exp_negative_scores.sum(dim=1)
        log_prob = torch.log(exp_positive_scores / total_scores_sum)

        loss = -log_prob.mean()
    



def main():
    
    parser = HfArgumentParser((ModelArguments, DataArguments, KoE5DataTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # train.py
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=data_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModel.from_pretrained(model_args.model_name_or_path)
    
    # print("Loading train_dataset...")
    # train_dataset = KoE5Dataset(
    #     args=data_args,
    #     tokenizer=tokenizer,
    #     mode='train',
    #     cache_dir=data_args.cache_dir,
    #     test=data_args.test,
    # )

    # print("Loading eval_dataset...")
    # eval_dataset = KoE5Dataset(
    #     args=data_args,
    #     tokenizer=tokenizer,
    #     mode='dev',
    #     cache_dir=data_args.cache_dir,
    #     test=data_args.test
    # )

    data_collator = DataCollatorForKoE5(
        tokenizer=tokenizer,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    
    ## dataset load
    data_path = "nlpai-lab/ko-triplet-v1.0"
    dataset = load_dataset(data_path)
    
    train_dataset = dataset['train']
    # eval_dataset = dataset['dev']
    
    # Example: Convert dataset['train'] into features
    
    
    hug_training_args = TrainingArguments(
        output_dir = training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        warmup_steps=training_args.warmup_steps,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        # cl_temperature=training_args.cl_temperature,
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    # train.py 이어서
    trainer = CustomTrainer(
        model=model,
        # args=training_args,
        args=hug_training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint

    if model_args.init_checkpoint is not None:
        print(f"Loading from {model_args.init_checkpoint} ...")
        state_dict = torch.load(os.path.join(model_args.init_checkpoint, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir=training_args.output_dir)
    
    
    

if __name__ == "__main__":
    
    main()


# 데이터가 이유 없이 빈값으로 들어올 때 -> 
# remove_unused_columns 이게 false로 되어 있는지 확인해야 함 !!!! 




