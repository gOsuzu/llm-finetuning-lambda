.PHONY: lambda-commands

create-hf-dataset:
	echo "Creating HF dataset"
	python src/dataset.py

generate-ssh-key:
	python src/lambda/commands.py generate-ssh-key

list-ssh-keys:
	python src/lambda/commands.py list-ssh-keys

list-instances:
	python src/lambda/commands.py list-instances

list-available-instance-types:
	python src/lambda/commands.py list-available-instance-types

get-lambda-ip:
	python src/lambda/commands.py get-lambda-ip

launch-lambda-instance:
	python src/lambda/commands.py launch

terminate-instances:
	python src/lambda/commands.py terminate
