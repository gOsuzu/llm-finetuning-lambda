# LLM Fine-tuning Template with Lambda Cloud

This repository provides a project template for fine-tuning Large Language Models (LLMs) using GPUs offered by Lambda Cloud.

## 🎯 Overview

### Features
- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning
- **Lambda Cloud** GPU instance execution
- **Hugging Face Hub** automatic model pushing
- **Simple commands** from setup to execution

### Supported Models
- Qwen2.5-7B-Instruct is a default setting.
- Other Hugging Face Transformers models. You can specify in the consntants.py file.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd llm-finetuning-lambda
```

### 2. Set Environment Variables
```bash
# Create .env file
cp .env.example .env

# Set required environment variables
HUGGINGFACE_TOKEN=your_huggingface_token # for storing the data and finetuned model
LAMBDA_API_KEY=your_lambda_api_key # for GPU calculations
OPENAI_API_KEY=your_openai_api_key # for data preprocessing
```

### 3. Launch Lambda Cloud Instance
```bash
make launch-lambda-instance
```

### 4. Transfer Files and Setup
```bash
# Get IP address
make get-lambda-ip

# Transfer dependency files
rsync -av --relative -e "ssh -i src/lambda/ssh-key/llm-finetune-template-lambda.pem" Makefile src/lambda/requirements_common.txt src/lambda/requirements_torch.txt ubuntu@<IP_ADDRESS>:/home/ubuntu/


# Transfer fine-tuning code
rsync -av -e "ssh -i src/lambda/ssh-key/llm-finetune-template-lambda.pem" src/finetune-llm ubuntu@<IP_ADDRESS>:/home/ubuntu/src/


# Transfer environment variables
rsync -av -e "ssh -i src/lambda/ssh-key/llm-finetune-template-lambda.pem" .env ubuntu@<IP_ADDRESS>:/home/ubuntu/

```

### 5. Install Dependencies
```bash
ssh -i src/lambda/ssh-key/llm-finetune-template-lambda.pem ubuntu@<IP_ADDRESS>
make lambda-setup
```

### 6. Execute Fine-tuning
```bash
make finetune-lora
```

## 📋 Command Reference

### Lambda Cloud Management
```bash
# Generate SSH key for Lambda Cloud
make generate-ssh-key

# List instance types
make list-instance-types

# Launch instance
make launch-lambda-instance

# Get IP address
make get-lambda-ip

# Terminate instance
make terminate-instance
```

### Development & Dataset
```bash
# Create Hugging Face dataset
make create-hf-dataset

# Download model files
make download-model
```

### Remote Environment
```bash
# Install dependencies
make lambda-setup

# Execute LoRA fine-tuning
make finetune-lora
```

## 🔧 Configuration

### Model Configuration (`src/finetune-llm/constants.py`)

```python
# Model configuration
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit": False,
    "dtype": "auto",
}

# LoRA configuration
LoRA_CONFIG = {
    "r": 32,                    # LoRA rank
    "lora_alpha": 64,           # Scaling factor
    "lora_dropout": 0.1,        # Dropout rate
    "target_modules": [         # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

# Training configuration
TRAINING_ARGS = {
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 5,
    "output_dir": "rick-llm-output",
    # ... other settings
}
```

### Dataset Configuration

Currently uses the `gOsuzu/rick-and-morty-transcripts-sharegpt` dataset.
To use your own dataset, modify the following section in `src/finetune-llm/finetune_lora.py`:

```python
# Load dataset
dataset = load_dataset("your-dataset-name", split="train")
```

## 📁 Project Structure

```
llm-finetuning-lambda/
├── src/
│   ├── dataset.py              # Dataset creation script
│   ├── download_model.py       # Model download script
│   ├── finetune-llm/
│   │   ├── finetune_lora.py    # Main fine-tuning script
│   │   ├── constants.py        # Configuration file
│   │   └── __init__.py
│   └── lambda/
│       ├── commands.py         # Lambda Cloud API commands
│       ├── requirements_common.txt  # Common dependencies
│       ├── requirements_torch.txt   # PyTorch dependencies
│       └── ssh-key/            # SSH key storage directory
├── Makefile                    # Command definitions
├── .env                        # Environment variables
└── README.md                   # This file
```

## 🔍 Technical Details

### What is LoRA (Low-Rank Adaptation)?

LoRA is a technique for efficient fine-tuning of large language models.

#### Features
- **Memory Efficient**: Updates only low-rank matrices instead of all parameters
- **Fast Training**: High-speed learning with fewer parameters
- **Quality Preservation**: Maintains quality close to full fine-tuning

#### Parameters
- **r**: Rank (dimension of low-rank matrices)
- **lora_alpha**: Scaling factor
- **target_modules**: Layers to apply LoRA

### Dataset Format

The current dataset expects the following format:

```json
{
  "conversations": [
    {"role": "system", "value": "System message"},
    {"role": "user", "value": "User message"},
    {"role": "assistant", "value": "Assistant message"}
  ]
}
```

### Chat Template

The following template is used during fine-tuning:

```
<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{INPUT}<|im_end|>
<|im_start|>assistant
{OUTPUT}<|im_end|>
```

## 🚨 Troubleshooting

### Common Issues

#### 1. SSH Connection Error
```bash
# Check IP address
make get-lambda-ip

# Test SSH connection
ssh -i src/lambda/ssh-key/llm-finetune-template-lambda.pem ubuntu@<IP_ADDRESS>
```

#### 2. Dependency Error
```bash
# Reinstall dependencies
make lambda-setup
```

#### 3. Memory Insufficient
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use larger GPU instances

#### 4. Hugging Face Token Error
```bash
# Check if .env file is transferred
ssh -i src/lambda/ssh-key/llm-finetune-template-lambda.pem ubuntu@<IP_ADDRESS> "ls -la /home/ubuntu/.env"
```

## 💰 Cost Optimization

### Lambda Cloud Pricing
- **gpu.1x.a10**: ~$0.60/hour
- **gpu.1x.a100**: ~$1.20/hour

### Optimization Tips
1. **Choose Appropriate Instance**: Select based on model size
2. **Efficient Training**: Adjust LoRA parameters
3. **Early Termination**: Terminate instances after sufficient training

## 📊 Results Verification

### Training Logs
During training, logs like the following will be displayed:
```
{'loss': 4.4448, 'grad_norm': 6.03125, 'learning_rate': 4e-05, 'epoch': 0.01}
{'loss': 0.162, 'grad_norm': 0.8984375, 'learning_rate': 0.0, 'epoch': 4.95}
```

### Model Verification
After training completion, you can verify the model at:
```
https://huggingface.co/gOsuzu/RickQwen2.5-7B
```

## 🔄 Next Steps

### 1. Using the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gOsuzu/RickQwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("gOsuzu/RickQwen2.5-7B")
```

### 2. API Deployment
- AWS Lambda + API Gateway
- FastAPI + Docker
- Hugging Face Inference API

### 3. Customization
- Create custom datasets
- Use different models
- Adjust hyperparameters

**Happy Fine-tuning! 🚀**
