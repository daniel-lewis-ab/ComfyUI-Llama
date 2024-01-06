
import sys
import os
import folder_paths

# Folder path and venv
llama_dir = os.path.dirname(os.path.realpath(__file__))
venv_site_packages = os.path.join(folder_paths.base_path, 'venv', 'Lib', 'site-packages')
sys.path.append(venv_site_packages)

# Attempt to get llama_cpp if it doesn't exist
try:
    from llama_cpp import Llama
except ImportError:

    # Determine the correct path based on the operating system
    if os.name == 'posix':
        site_packages = os.path.join(sys.prefix, 'lib', 'python{}.{}/site-packages'.format(sys.version_info.major, sys.version_info.minor))
    else:  # For Windows
        site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

    sys.path.append(site_packages)
    from llama_cpp import Llama

# Inject 'llm' to folder_paths ourselves, so we can use it like we belong there and have behavioral consistency
supported_file_extensions = set(['.gguf'])
models_dir = os.path.join(llama_dir, "models")
folder_paths.folder_names_and_paths["llm"] = ([models_dir], supported_file_extensions)

class LLM_Load_Model:

    # A hopefully thin wrapper for the Llama class to bind it to ComfyUI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Model": (folder_paths.get_filename_list("llm"), ), 
            },
            "optional": {
                "n_ctx": ("INT", {"default": 0, "step":512, "min":0}),
            }
        }

    # ComfyUI will effectively return the Llama class instanciation provided by execute() and call it an LLM
    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, Model, n_ctx):

        # basically just calls __init__ on the Llama class
        model_path = folder_paths.get_full_path("llm", Model)
        llm = Llama(model_path=model_path, chat_format="llama-2", n_ctx=n_ctx, seed=-1)

        return (llm,)

class LLM_Call:

    # A hopefully thin wrapper for the Llama generate method to bind it to ComfyUI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "prompt":("STRING", {"default":"", "multiline":True}),
            },
            "optional": {
                "max_response_tokens": ("INT", {"default": 0}),
                "temperature": ("FLOAT", {"default": 0.8, "min":0.0, "max":1.0, "step":0.01, "round":0.01, "display":"number"}),
                "seed": ("INT", {"default": -1}),
            }
        }

    # ComfyUI will effectively return the text result of the Llama.__call__() as a STRING
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM, prompt, max_response_tokens, temperature, seed):

        # I'm using __call__ and not generate() because seed isn't available in generate!
        response = LLM.__call__(
            prompt=prompt, 
            max_tokens=max_response_tokens, 
            temperature=temperature, 
            seed=seed,
            stop=LLM.token_eos() )
        # print(response['choices'][0]['text'])
        return (response['choices'][0]['text'], )

NODE_CLASS_MAPPINGS = {
    "Load LLM Model":LLM_Load_Model,
    "Call LLM":LLM_Call,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load LLM Model": "Load LLM Model",
    "Call LLM":"Call LLM",
}


