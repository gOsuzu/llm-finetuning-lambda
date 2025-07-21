import os
import dotenv
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from constants import MAX_SEQ_LENGTH, MODEL_CONFIG, TRAINING_ARGS, LoRA_CONFIG, COMET_CONFIG
import comet_ml

dotenv.load_dotenv()

def setup_comet_experiment():
    """Setup Comet ML experiment for tracking"""
    try:
        # Get Comet API key from environment
        comet_api_key = os.getenv("COMET_API_KEY")
        if not comet_api_key:
            print("Warning: COMET_API_KEY not found in environment variables. Comet tracking will be disabled.")
            return None
        
        # Initialize Comet experiment
        experiment = comet_ml.Experiment(
            api_key=comet_api_key,
            project_name=COMET_CONFIG["project_name"],
            workspace=COMET_CONFIG["workspace"],
            experiment_name=COMET_CONFIG["experiment_name"],
            log_code=COMET_CONFIG["log_code"],
            log_parameters=COMET_CONFIG["log_parameters"],
            log_metrics=COMET_CONFIG["log_metrics"],
            log_histograms=COMET_CONFIG["log_histograms"],
            log_gradients=COMET_CONFIG["log_gradients"],
        )
        
        # Log model and training configuration
        experiment.log_parameters({
            "model_name": MODEL_CONFIG["model_name"],
            "max_seq_length": MAX_SEQ_LENGTH,
            **LoRA_CONFIG,
            **TRAINING_ARGS
        })
        
        print(f"Comet experiment initialized: {experiment.get_key()}")
        return experiment
        
    except Exception as e:
        print(f"Error setting up Comet experiment: {e}")
        return None

def main():
    # Setup Comet experiment
    experiment = setup_comet_experiment()
    
    # モデルとトークナイザーの初期化
    model_name = MODEL_CONFIG["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=MODEL_CONFIG["dtype"],
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA設定
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LoRA_CONFIG["r"],
        lora_alpha=LoRA_CONFIG["lora_alpha"],
        lora_dropout=LoRA_CONFIG["lora_dropout"],
        target_modules=LoRA_CONFIG["target_modules"]
    )
    
    # PEFTモデルの作成
    model = get_peft_model(model, lora_config)
    
    # Log model info to Comet
    if experiment:
        experiment.log_parameter("total_parameters", sum(p.numel() for p in model.parameters()))
        experiment.log_parameter("trainable_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # データセットの読み込み
    dataset = load_dataset("gOsuzu/rick-and-morty-transcripts-sharegpt", split="train")
    
    # Log dataset info to Comet
    if experiment:
        experiment.log_parameter("dataset_size", len(dataset))
        experiment.log_parameter("dataset_name", "gOsuzu/rick-and-morty-transcripts-sharegpt")
    
    # チャットテンプレートの適用
    def apply_chat_template(example):
        chat_template = """<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{INPUT}<|im_end|>
<|im_start|>assistant
{OUTPUT}<|im_end|>"""
        
        conversations = example["conversations"]
        system_msg = conversations[0]["value"]
        user_msg = conversations[1]["value"]
        assistant_msg = conversations[2]["value"]
        
        formatted_text = chat_template.format(
            SYSTEM=system_msg,
            INPUT=user_msg,
            OUTPUT=assistant_msg
        )
        
        return {"text": formatted_text}
    
    processed_dataset = dataset.map(apply_chat_template)
    
    # トークナイゼーション
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None
        )
    
    tokenized_dataset = processed_dataset.map(tokenize_function, batched=True, remove_columns=processed_dataset.column_names)
    
    # トレーニング設定
    training_args = TrainingArguments(**TRAINING_ARGS)
    
    # データコレーター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Custom callback for additional Comet logging
    class CometCallback:
        def __init__(self, experiment):
            self.experiment = experiment
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and self.experiment:
                # Log additional metrics
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        self.experiment.log_metric(key, value, step=state.global_step)
    
    # トレーナーの初期化とトレーニング
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[CometCallback(experiment)] if experiment else None,
    )
    
    print("Starting training with Comet ML tracking...")
    trainer.train()
    
    # モデルの保存
    trainer.save_model()
    tokenizer.save_pretrained(TRAINING_ARGS["output_dir"])
    
    # Log final model info
    if experiment:
        experiment.log_parameter("training_completed", True)
        experiment.log_parameter("final_loss", trainer.state.log_history[-1].get("loss", "N/A") if trainer.state.log_history else "N/A")
        experiment.end()
        print(f"Training completed! View results at: {experiment.get_url()}")
    
    # Hugging Face Hubにプッシュ
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if not huggingface_token:
        print("⚠️  HUGGINGFACE_TOKENが設定されていません。")
        print("    .envファイルにHUGGINGFACE_TOKENを設定してください。")
        print("    例: HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return
    
    try:
        print("🚀 Hugging Face Hubにモデルをプッシュ中...")
        
        # モデルをプッシュ
        model.push_to_hub(
            "gOsuzu/RickQwen2.5-7B",
            token=huggingface_token,
            private=True,  # 非公開リポジトリ
        )
        
        # トークナイザーをプッシュ
        tokenizer.push_to_hub(
            "gOsuzu/RickQwen2.5-7B",
            token=huggingface_token,
            private=True,  # 非公開リポジトリ
        )
        
        print("✅ ファインチューニングが完了しました！")
        print("✅ モデルがHugging Face Hubにプッシュされました: gOsuzu/RickQwen2.5-7B")
        print("🌐 モデルURL: https://huggingface.co/gOsuzu/RickQwen2.5-7B")
        
    except Exception as e:
        print(f"❌ Hugging Face Hubへのプッシュに失敗しました: {e}")
        print("💡 ローカルに保存されたモデルは ./rick-llm-output にあります")
        print("💡 手動でプッシュする場合は以下を実行してください:")
        print("   cd ./rick-llm-output")
        print("   huggingface-cli login")
        print("   huggingface-cli repo create gOsuzu/RickQwen2.5-7B --type model")
        print("   git add . && git commit -m 'Add model' && git push")

if __name__ == "__main__":
    main() 