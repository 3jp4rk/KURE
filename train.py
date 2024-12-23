from transformers import AutoTokenizer, AutoModel, HfArgumentParser, TrainingArguments
from datasets import load_dataset
from data_collator import DataCollatorForKoE5
from torch.utils.data import DataLoader
from transformers import Trainer
import torch.nn as nn
import torch
import os
from arguments import ModelArguments, DataArguments, KoE5DataTrainingArguments
from dataset import KoE5Dataset

torch.autograd.set_detect_anomaly(True)

# os.environ["WANDB_DISABLED"] = "true" # not to write to wandb

def average_pool(hidden_states, attention_mask):
    sum_hidden_states = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
    avg_hidden_states = sum_hidden_states / attention_mask.sum(dim=1, keepdim=True)
    return avg_hidden_states

class CustomTrainer(Trainer):
    
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

            output = model(**input_dict)
            pooled_output = average_pool(output.last_hidden_state, input_dict['attention_mask'])
            embeddings[k] = pooled_output

        query_embeddings = embeddings['query']
        positive_embeddings = embeddings['document']
        negative_embeddings = embeddings['hard_negative']

        similarity_fct = nn.CosineSimilarity(dim=-1)
        tau = 0.02

        positive_scores = similarity_fct(query_embeddings, positive_embeddings) / tau
        negative_scores = similarity_fct(query_embeddings.unsqueeze(1), negative_embeddings) / tau

        max_positive_scores = torch.max(positive_scores, dim=0, keepdim=True)[0]
        max_negative_scores = torch.max(negative_scores, dim=1, keepdim=True)[0]
        max_scores = torch.max(max_positive_scores, max_negative_scores)

        stable_positive_scores = positive_scores - max_scores
        stable_negative_scores = negative_scores - max_scores.unsqueeze(1)

        exp_positive_scores = torch.exp(stable_positive_scores)
        exp_negative_scores = torch.exp(stable_negative_scores)

        total_scores_sum = exp_positive_scores + exp_negative_scores.sum(dim=1)
        log_prob = torch.log(exp_positive_scores / total_scores_sum)

        loss = -log_prob.mean()
        return (loss, output) if return_outputs else loss


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, KoE5DataTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=data_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModel.from_pretrained(model_args.model_name_or_path)

    # Load dataset using Hugging Face datasets library
    dataset = load_dataset("nlpai-lab/ko-triplet-v1.0", split="train")
    dataset = dataset.select(range(12000))
    # dataset = dataset.select()

    # Convert dataset into KoE5Dataset
    train_dataset = KoE5Dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=model_args.max_seq_length
    )

    data_collator = DataCollatorForKoE5(
        tokenizer=tokenizer,
        padding=True,
        max_length=model_args.max_seq_length,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    
    # from torch.utils.data import DataLoader

    # data_loader = DataLoader(
    #     train_dataset,
    #     batch_size=training_args.per_device_train_batch_size,
    #     collate_fn=data_collator,  # DataCollatorForKoE5 사용
    # )
    
    # 첫 배치 확인
    # for batch in data_loader:
    #     print("Batch from DataLoader:", batch)
    #     break
    
    # quit()
        
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Resume from checkpoint if specified
    checkpoint = training_args.resume_from_checkpoint or None
    if model_args.init_checkpoint is not None:
        print(f"Loading from {model_args.init_checkpoint} ...")
        state_dict = torch.load(os.path.join(model_args.init_checkpoint, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)

    # Start training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir=training_args.output_dir)
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
