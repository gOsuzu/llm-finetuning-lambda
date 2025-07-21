.PHONY: lambda-commands

# Detect Python command
PYTHON := $(shell command -v python 2> /dev/null || command -v python3 2> /dev/null || echo python3)

create-hf-dataset:
	echo "Creating HF dataset"
	$(PYTHON) src/dataset.py

create-dataset: create-hf-dataset

generate-ssh-key:
	$(PYTHON) src/lambda/commands.py generate-ssh-key

list-ssh-keys:
	$(PYTHON) src/lambda/commands.py list-ssh-keys

list-instances:
	$(PYTHON) src/lambda/commands.py list-instances

list-instance-types:
	$(PYTHON) src/lambda/commands.py list-types

get-lambda-ip:
	$(PYTHON) src/lambda/commands.py get-ip

launch-lambda-instance:
	$(PYTHON) src/lambda/commands.py launch

lambda-help:
	$(PYTHON) src/lambda/commands.py

lambda-setup:
	echo "Installing dependencies"
	sudo apt update && sudo apt upgrade -y
	sudo apt install curl libcurl4-openssl-dev -y 
	sudo apt remove python3-jax python3-jaxlib -y
	sudo pip uninstall tensorflow tf-keras jax jaxlib -y
	pip install -r src/lambda/requirements_common.txt
	pip install -r src/lambda/requirements_torch.txt --index-url https://download.pytorch.org/whl/cu121

finetune-lora:
	echo "Finetuning LLM with LoRA approach"
	$(PYTHON) src/finetune-llm/finetune_lora.py

download-model:
	echo "Downloading model files"
	$(PYTHON) src/download_model.py

terminate-instance:
	python src/lambda/commands.py terminate