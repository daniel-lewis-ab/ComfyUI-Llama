# Llama for ComfyUI
Improved [llama-cpp](https://github.com/abetlen/llama-cpp-python) integration for ComfyUI.

# Installation

## If using Comfy Manager:
Not available at this time.

## If installing manually:
1. Clone this repo into `custom_nodes` folder.
2. Install llama-cpp-python.

# How to Use:
1. Download GGUF language learning models, which can be found on HuggingFace. You will need at least 1. Different models produce different results.
2. Place models in ```ComfyUI/custom_nodes/ComfyUI-Llama/models```. They can be renamed if you want.
3. Fire up/Restart ComfyUI and allow it to finish restarting.
4. Hit Ctrl+F5 to ensure the browser is refreshed.
5. Check your ComfyUI available nodes and find the menu for ours.

# Notable Updates
Current version is 0.1.0

# Features:
Currently uses dropdowns to load GGUF models in a consistent fashion with other ComfyUI nodes and can use them to generate strings of output text with seemingly correct seeding and temperature.

# Upcoming features:

# Core Nodes:

# Usage
See the example1.json or example1.png file.
See (https://abetlen.github.io/llama-cpp-python/) for documentation on the llama-cpp-python interface

# Known Issues
1. No known effective way to output long strings of text to nodes within ComfyUI for examination by users.
2. No known way to loopback output from an LLM model repeatedly, but I'm investigating Loopchain.
3. Haven't widely tested models, operating systems, file path trickery, emulators, or strange abuses of inputs.
4. Will simply crash if llama-cpp-python throws an Error.
5. Haven't thoroughly tested installer.
6. Haven't tried to list the custom_node with Comfy Manager yet.



