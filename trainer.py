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
        tau = self.args.cl_temperature

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
    
        return loss
