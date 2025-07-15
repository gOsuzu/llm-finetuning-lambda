.PHONY: lambda-commands

create-hf-dataset:
	echo "Creating HF dataset"
	python src/dataset.py