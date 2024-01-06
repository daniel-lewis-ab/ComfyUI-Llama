import importlib
import os
import subprocess
import json
from .Llama import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

base_dir = os.path.dirname(os.path.abspath(__file__))

def install_package():

    # Navigate to the repository directory
    os.chdir(base_dir)

    # Install requirements
    venv_activate = ".\\venv\\Scripts\\activate"
    requirements_install = "pip install -r requirements.txt"
    subprocess.run(f"{venv_activate} && {requirements_install}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def install_package2():

    # Navigate to the repository directory
    os.chdir(base_dir)

    # Install requirements
    venv_activate = ".\\venv\\Scripts\\activate"
    requirements_install = "pip install -r requirements.txt"
    subprocess.run(f"{venv_activate} && {requirements_install}", shell=True)

def read_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def update_repository(repo_path):
    try:
        # Navigate to the repository directory
        os.chdir(repo_path)

        # Execute 'git pull' command
        result = subprocess.run(['git', 'pull'], check=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())

    except Exception as e:
        print(f"An error occurred while updating the repository: {e}")

def update_config(file_path, auto_update_value):
    try:
        # Read the existing JSON file
        with open(file_path, 'r') as file:
            config = json.load(file)

        # Update the AutoUpdate value
        config['first_installation'] = auto_update_value

        # Write the updated config back to the file
        with open(file_path, 'w') as file:
            json.dump(config, file, indent=4)  # Indent for pretty printing

    except Exception as e:
        print(f"An error occurred: {e}")

auto_update_path = os.path.join(base_dir, 'AutoUpdate.json')
print(auto_update_path)
config = read_config(auto_update_path)
auto_update = config.get('AutoUpdate', False)
first_install = config.get('first_installation', False)

if first_install:
    try:
        install_package2()
        print("Llama installed")
    except Exception as e:
        print(f"An error occurred during installation: {e}")
        print("Something went wrong. Please try running the Update.bat inside ComfyUI/custom_nodes/Llama. If the error still remains, please open an issue on https://github.com/daniel-lewis-ab/Llama")
    update_config(auto_update_path, False)

if auto_update:
    try:
        update_repository(base_dir)
        print("Llama updated")
    except Exception as e:
        print(f"An error occurred during installation: {e}")
    try:
        install_package()
        print("Requirements updated")
    except Exception as e:
        print(f"An error occurred during installation: {e}")



