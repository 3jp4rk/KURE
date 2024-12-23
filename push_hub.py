from transformers import AutoModel

path = "/data/ejpark/KoE5/MODELS/xlm-roberta-large-v1.2-InfoNCE-bs=8-ep=5-lr=1e-5--12000"
model = AutoModel.from_pretrained(path)

model.push_to_hub("fasoo/fasoo-embedding")