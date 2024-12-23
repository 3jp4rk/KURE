# from CustomModel import EmbeddingModel

from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer, models

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def make_sentTransformer(model_path, tokenizer_path=None):
    
    tokenizer_path = model_path if not tokenizer_path else tokenizer_path
    model = models.Transformer(model_path, tokenizer_name_or_path=tokenizer_path)
    pooling = models.Pooling(model.get_word_embedding_dimension(), pooling_mode="max")
    model = SentenceTransformer(modules=[model, pooling])
    
    return model


def run_bench(model_path, tokenizer_path):
    
    # model = EmbeddingModel(model_path) 
    

    tasks = [
        
        # text mining
        "IWSLT2017BitextMining", # "kor-eng", "eng-kor"
        "Tatoeba", # "kor-eng",
        
        
        # # sts
        "KLUE-STS",
        "KorSTS",
        "STS17",
        
        # retrieval
        "AutoRAGRetrieval",
        "BelebeleRetrieval",
        "Ko-StrategyQA",
        "MIRACLRetrieval",
        "MultiLongDocRetrieval",
        "PublicHealthQA",
        "SIB200ClusteringS2S",
        "FloresBitextMining",
        "MrTidyRetrieval",
        "XPQARetrieval",
        "KLUE-NLI",
        
    ]
    
    # tasks = [

    # ]
    
    # mteb_tasks = mteb.get_tasks(languages=["kor"])
    
    tasks = [
        mteb.get_task(task, languages=["kor"]) for task in tasks
    ]

    # languages = ["kor", "eng"]
    # languages = ["kor"]
    
    # evaluation = MTEB(tasks=tasks, languages=languages)
    # print(tasks)
    # quit()
    evaluation = MTEB(tasks=tasks)
    model = make_sentTransformer(model_path, tokenizer_path)
    results = evaluation.run(model, output_folder=f"results_aica/{model_path}")
    # results = evaluation.run(model, output_folder=f"results_aica/{model_path}")


    
#     model = make_sentTransformer(model_path, tokenizer_path)
#     tasks = mteb.get_tasks(tasks=[""])
    


if __name__ == "__main__":
    
    
    # tasks = mteb.get_tasks(languages=["kor"])
    # print(tasks)
    # quit()

    


    # model_name = "nlpai-lab/KoE5"
    # model = SentenceTransformer(model_name)

    new_model_name = "ellm_ret_new"
    old_model_name = "ellm_ret_old"

    # model_names = [new_model_name, old_model_name]
    
    model_names = [
        {
            "model_name": "/data/ejpark/KoE5/MODELS/xlm-roberta-large-v1.2-InfoNCE-bs=8-ep=5-lr=1e-5--12000",
            "tokenizer_path": None
        },
    ]
    
    for model_item in model_names:
        model_name = model_item['model_name']
        tokenizer_path = model_item['tokenizer_path']
        print(f"working on {model_name}...")
        run_bench(model_name, tokenizer_path)
        





