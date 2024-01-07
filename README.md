# Llama for ComfyUI

## What is this
This is an improved [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) integration for ComfyUI.

## Features:
- Currently let's you easily load GGUF models in a consistent fashion with other ComfyUI models and can use them to generate strings of output text with seemingly correct seeding and temperature.

## Upcoming Features:
- Intend to expand interaction and text display.

# Installation

## If using Comfy Manager:
1. Visit your Install Custom Nodes page, and search for ComfyUI-Llama.
2. Hit Install and restart when prompted.
3. Copy your GGUF files into ```./ComfyUI/custom_nodes/ComfyUI-Llama/models/*```
4. Hit Ctrl+F5 to hard reload the browser window.
5. The nodes should be in the LLM menu.

## If installing manually:
1. Clone this repo into `custom_nodes` folder.
2. Install llama-cpp-python.
3. Copy your GGUF files into ```./ComfyUI/custom_nodes/ComfyUI-Llama/models/*```
4. Hit Ctrl+F5 to hard reload the browser window.
5. The nodes should be in the LLM menu.


# Usage

## Instructions:
1. Download GGUF language learning models, which can be found on HuggingFace. You will need at least 1. Different models produce different results.

2. Place models in ```ComfyUI/custom_nodes/ComfyUI-Llama/models```. They can be renamed if you want.

3. Fire up/Restart ComfyUI and allow it to finish restarting.

4. Hit Ctrl+F5 to ensure the browser is refreshed.

5. Check your ComfyUI available nodes and find the LLM menu.

6. Load LLM Model

This is a simplified call of this:
[llama-cpp-python's init method](https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.__init__)

7. Call LLM Model

This is a simplified call of this:

[llama-cpp-python's call method](https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.__call__)

## Examples
See the example1.json or example1.png file.
See the documentation for [llama-cpp-python](https://abetlen.github.io/llama-cpp-python/) for documentation on the llama-cpp-python interface

# For Possible Contributors

## Known Issues
1. No known effective way to output long strings of text to nodes within ComfyUI for examination by users.
2. No known way to loopback output from an LLM model repeatedly, but I'm investigating Loopchain.
3. Haven't widely tested models, operating systems, file path trickery, emulators, or strange abuses of inputs.
4. Will simply crash if llama-cpp-python throws an Error.




