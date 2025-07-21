import os
import json
import sys
# import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LAMBDA_API_KEY")
BASE_URL = "https://cloud.lambda.ai/api/v1"


def generate_ssh_key():
    """Generate SSH key and save to file."""
    url = f"{BASE_URL}/ssh-keys"
    open_file = "src/lambda/ssh-key/llm-finetune-template-lambda.pem"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {"name": f"llm-finetune-template-lambda"}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        private_key = response.json()["data"]["private_key"]

        with open(open_file, "w") as f:
            f.write(private_key)

        os.chmod(open_file, 0o600)
        print("SSH key generated and saved to file. Check your Lambda Labs account!")

    except requests.exceptions.RequestException as e:
        print(f"Error generating SSH key: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return None


def get_existing_ssh_key():
    """Get existing SSH key private key."""
    url = f"{BASE_URL}/ssh-keys"
    open_file = "src/lambda/ssh-key/llm-finetune-template-lambda.pem"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        ssh_keys = response.json()["data"]
        for key in ssh_keys:
            if key["name"] == "llm-finetune-template-lambda":
                # Note: Lambda Labs API doesn't return private keys for existing keys
                # We need to generate a new key or use a different approach
                print("SSH key 'llm-finetune-template-lambda' already exists, but private key is not available.")
                print("Please generate a new SSH key with a different name.")
                return None
        
        print("SSH key 'llm-finetune-template-lambda' not found")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching SSH keys: {e}")
        return None


def list_ssh_keys():
    """List all SSH keys associated with the account."""
    url = f"{BASE_URL}/ssh-keys"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching SSH keys: {e}")
        return None


def list_intances():
    """List all instances."""

    url = f"{BASE_URL}/instances"
    auth = (API_KEY, "") if API_KEY else None

    try:
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching instances: {e}")
        return None


def list_available_instance_types():
    """List all available instance types from Lambda Labs."""

    url = f"{BASE_URL}/instance-types"
    auth = (API_KEY, "") if API_KEY else None

    try:
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching instance types: {e}")
        return None


def get_llm_finetune_template_vm_ip():
    """Get the IP address of the LLM finetune template VM."""
    instances = list_intances()
    
    if instances is None:
        print("Failed to fetch instances")
        return None

    for instance in instances["data"]:
        if instance["name"] == "llm-finetune-template-instance":
            try:
                print(f"LLM finetune template VM IP: {instance['ip']}")
                return instance['ip']
            except KeyError:
                print("LLM finetune template VM IP not found")
                return None
    
    print("LLM finetune template instance not found")
    return None


def get_llm_finetune_template_vm_id():
    """Get the instance ID of the LLM finetune template VM."""
    instances = list_intances()
    
    if instances is None:
        print("Failed to fetch instances")
        return None

    for instance in instances["data"]:
        if instance["name"] == "llm-finetune-template-instance":
            try:
                print(f"LLM finetune template VM ID: {instance['id']}")
                return instance['id']
            except KeyError:
                print("LLM finetune template VM ID not found")
                return None
    
    print("LLM finetune template instance not found")
    return None


def launch_instance():
    """Launch a new Lambda Labs instance."""
    url = f"{BASE_URL}/instance-operations/launch"
    headers = {"Content-Type": "application/json"}
    auth = (API_KEY, "") if API_KEY else None

    try:
        with open("src/lambda/request.json", "r") as f:
            instance_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading config file: {e}")
        return None

    try:
        response = requests.post(url, auth=auth, headers=headers, json=instance_config)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error launching instance: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return None


def terminate_instances():
    """Terminate the LLM finetune template VM."""
    url = f"{BASE_URL}/instance-operations/terminate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    instance_id = get_llm_finetune_template_vm_id()
    if instance_id is None:
        print("Failed to get instance ID")
        return None

    data = {"instance_ids": [instance_id]}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully terminated instances: {instance_id}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error terminating instances: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return None





def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python commands.py <command>")
        print("Available commands:")
        print("generate-ssh-key - Generate a new SSH key")
        print("list-ssh-keys    - List all SSH keys")
        print("list-instances   - List all instances")
        print("list-types       - List available instance types")
        print("get-ip           - LLM finetune template VM IP")
        print("launch           - Launch instance")
        print("terminate        - Terminate instance")
        return

    command = sys.argv[1]

    commands = {
        "generate-ssh-key": generate_ssh_key,
        "list-ssh-keys": list_ssh_keys,
        "list-instances": list_intances,
        "list-types": list_available_instance_types,
        "get-ip": get_llm_finetune_template_vm_ip,
        "launch": launch_instance,
        "terminate": terminate_instances,
    }

    if command not in commands:
        print(f"Unknown command: {command}")
        return

    result = commands[command]()
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    elif result is not None:
        print(result)

if __name__ == "__main__":
    main()
