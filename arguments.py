from dataclasses import dataclass, field
from transformers import TrainingArguments



@dataclass
class ModelArguments:
    """
    Arguments for specifying the model and tokenizer.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name_or_path"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch name, tag name, or commit id)."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer implementations (backed by the 🤗 Tokenizers library)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Whether to use the token generated during `transformers-cli login` (needed for private models)."}
    )
    init_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to initialize the model from a checkpoint."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Path to initialize the model from a checkpoint."}
    )


@dataclass
class DataArguments:
    """
    Arguments for specifying the data directories and configurations.
    """
    # data_dir: str = field(
    #     metadata={"help": "Path to the data directory."}
    # )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Path to the cache directory."}
    )
    test: bool = field(
        default=False,
        metadata={"help": "Whether to run in test mode."}
    )


@dataclass
class KoE5DataTrainingArguments(TrainingArguments):
    """
    Arguments specific to training KoE5 models.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    cl_temperature: float = field(
        default=0.02,
        metadata={"help": "Contrastive learning temperature hyperparameter."}
    )
    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to a specific checkpoint to resume training from."}
    )
    # ddp_find_unused_parameters: bool = field(
    #     default=False,
    #     metadata={"help": "Path to a specific checkpoint to resume training from."}
    # )
