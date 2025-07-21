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
from constants import MAX_SEQ_LENGTH, MODEL_CONFIG, TRAINING_ARGS, LoRA_CONFIG

dotenv.load_dotenv()

def main():
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
    
    # データセットの読み込み
    dataset = load_dataset("gOsuzu/rick-and-morty-transcripts-sharegpt", split="train")
    
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
    
    # トレーナーの初期化とトレーニング
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()

    # モデルの保存
    trainer.save_model()
    tokenizer.save_pretrained(TRAINING_ARGS["output_dir"])
    
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