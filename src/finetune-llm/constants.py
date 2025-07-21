MAX_SEQ_LENGTH = 2048

# model config
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit": False,
    "dtype": "auto",
}

# parameter efficient finetuning config
LoRA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "task_type": "CAUSAL_LM",
}

# training config
TRAINING_ARGS = {
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 5,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "output_dir": "rick-llm-output",
    "seed": 42,
    "report_to": "none",
    "save_steps": 1000,
    "save_total_limit": 2,
    "remove_unused_columns": False,
}
