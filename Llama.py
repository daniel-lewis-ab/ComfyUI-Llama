
import sys
import os
from typing import List
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
    """
    Load a llama.cpp model from model_path.
    (easy version)

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__
    """

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

    def execute(self, Model:str, n_ctx:int):

        # basically just calls __init__ on the Llama class
        model_path = folder_paths.get_full_path("llm", Model)

        try:
            llm = Llama(model_path=model_path, chat_format="llama-2", n_ctx=n_ctx, seed=-1)

        except ValueError:
            alert("The model path does not exist.  Perhaps hit Ctrl+F5 and try reloading it.")

        return (llm,)


class LLM_Load_Model_Advanced:
    """
    Load a llama.cpp model from model_path.
    (advanced version)

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__

    Missing:
        split_mode
        tensor-split
        kv_overrides
        chat_handler

    Different:
        seed - I'm using ComfyUI's seed functionality and not llama-cpp-python's

    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Model": (folder_paths.get_filename_list("llm"), ), 
            },
            "optional": {
                "n_gpu_layers": ("INT", {"default": 0, "min":0}),
                "main_gpu": ("INT", {"default": 0, "min":0}),
                "vocab_only": ("BOOLEAN", {"default": False}),
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),
                "n_ctx": ("INT", {"default": 512, "min":0, "step":512}),
                "n_batch": ("INT", {"default": 512, "min":0, "step":512}),
                "n_threads": ("INT", {"default": None}),
                "n_threads_batch": ("INT", {"default": None}),
                "rope_freq_base": ("FLOAT", {"default": 0.0, "min":0.0, "max":1.0, "step":0.01}),
                "rope_freq_scale": ("FLOAT", {"default": 0.0, "min":0.0, "max":1.0, "step":0.01}),
                "yarn_ext_factor": ("FLOAT", {"default": -1.0, "max":1.0, "step":0.01}),
                "yarn_attn_factor": ("FLOAT", {"default": 1.0, "min":0.0, "max":1.0, "step":0.01}),
                "yarn_beta_fast": ("FLOAT", {"default": 32.0, "min":0.0, "step":0.01}),
                "yarn_beta_slow": ("FLOAT", {"default": 1.0, "min":0.0, "max":1.0, "step":0.01}),
                "yarn_orig_ctx": ("INT", {"default": 0, "min":0}),
                "mul_mat_q": ("INT", {"default": 0, "min":0}),
                "logits_all": ("BOOLEAN", {"default": False}),
                "embedding": ("BOOLEAN", {"default": False}),
                "offload_kqv": ("BOOLEAN", {"default": False}),
                "last_n_tokens_size": ("INT", {"default": 64, "min":0}),
                "lora_base": ("STRING", {"default": None}),
                "lora_scale": ("FLOAT", {"default": 0.0, "min":0.0, "max":1.0, "step":0.01}),
                "lora_path": ("STRING", {"default": None}),
                "numa": ("BOOLEAN", {"default": False}),
                "chat_format": ("STRING", {"default": "llama-2"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    # ComfyUI will effectively return the Llama class instanciation provided by execute() and call it an LLM
    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(
        self, 
        Model:str, 
        n_gpu_layers:int, 
        main_gpu:int, 
        vocab_only:bool,
        use_mmap:bool,
        use_mlock:bool,
        seed:int,
        n_ctx:int,
        n_batch:int,
        n_threads:int,
        n_threads_batch:int,
        rope_freq_base:float,
        rope_freq_scale:float,
        yarn_ext_factor:float,
        yarn_attn_factor:float,
        yarn_beta_fast:float,
        yarn_beta_slow:float,
        yarn_orig_ctx:int,
        mul_mat_q:bool,
        logits_all:bool,
        embedding:bool,
        offload_kqv:bool,
        last_n_tokens_size:int,
        lora_base:str,
        lora_scale:float,
        lora_path:str,
        numa:bool,
        chat_format:str,
        verbose:bool):

        # basically just calls __init__ on the Llama class
        model_path = folder_paths.get_full_path("llm", Model)

        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers, 
                main_gpu=main_gpu, 
                vocab_only=vocab_only,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                seed=seed,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                rope_freq_base=rope_freq_base,
                rope_freq_scale=rope_freq_scale,
                yarn_ext_factor=yarn_ext_factor,
                yarn_attn_factor=yarn_attn_factor,
                yarn_beta_fast=yarn_beta_fast,
                yarn_beta_slow=yarn_beta_slow,
                yarn_orig_ctx=yarn_orig_ctx,
                mul_mat_q=mul_mat_q,
                logits_all=logits_all,
                embedding=embedding,
                offload_kqv=offload_kqv,
                last_n_tokens_size=last_n_tokens_size,
                lora_base=lora_base,
                lora_scale=lora_scale,
                lora_path=lora_path,
                numa=numa,
                chat_format=chat_format,
                verbose=verbose)

        except ValueError:
            alert("The model path does not exist.  Perhaps hit Ctrl+F5 and try reloading it.")

        return (llm,)


class LLM_Tokenize:
    """
    Tokenize a string.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.tokenize
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "text": ("TEXT", )
            },
            "optional": {
                "add_bos": ("BOOLEAN", {"default": True}),
                "special": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TOKENS",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, text:bytes, add_bos:bool, special:bool):

        try:
            tokens = LLM.tokenize(
                text=text,
                add_bos=add_bos,
                special=special)

        except RuntimeError:
            alert("RuntimeError: If the tokenization failed. ")

        return (tokens,)


class LLM_Detokenize:
    """
    Detokenize a list of tokens.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.detokenize
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "tokens": ("TOKENS", )
            },
        }

    RETURN_TYPES = ("TEXT",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, tokens:List[int]):

        thebytes = LLM.detokenize(tokens)

        return (thebytes,)


class LLM_Reset:
    """
    Reset the model state.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.reset
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
            },
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama):
        llm = LLM.reset()
        return (llm,)


class LLM_Eval:
    """
    Evaluate a list of tokens.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.eval
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "tokens": ("TOKENS", )
            },
        }
    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, tokens:List[int]):

        thebytes = LLM.eval(tokens)
        return None


class LLM_Sample:
    """
    Sample a token from the model.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.sample
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
            },
            "optional": {
                "top_k":("INT",{"default":40}),
                "top_p":("FLOAT",{"default":0.95}),
                "temp":("FLOAT",{"default":0.8}),
                "repeat_penalty":("FLOAT",{"default":1.1}), 
            }
        }
    RETURN_TYPES = ("TOKEN", )
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama):

        thebytes = LLM.sample(
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty)
        return (thebytes, )


class LLM_Call:
    """
    Generate text from a prompt.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__
    """
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
        try:
            response = LLM.__call__(
                prompt=prompt, 
                max_tokens=max_response_tokens, 
                temperature=temperature, 
                seed=seed,
                stop=LLM.token_eos() )
        except ValueError:
            alert('ValueError: If the requested tokens exceed the context window.');
        except RuntimeError:
            alert('RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.')
        return (response['choices'][0]['text'], )

NODE_CLASS_MAPPINGS = {
    "Load LLM Model":LLM_Load_Model,
    "Load LLM Model Advanced":LLM_Load_Model_Advanced,
    "LLM_Detokenize":LLM_Detokenize,
    "LLM_Tokenize":LLM_Tokenize,
    "LLM_Reset":LLM_Reset,
    "LLM_Eval":LLM_Eval,
    "LLM_Sample":LLM_Sample,
    "Call LLM":LLM_Call,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load LLM Model": "Load LLM Model",
    "Load LLM Model Advanced": "Load LLM Model Advanced",
    "LLM_Tokenize": "LLM_Tokenize",
    "LLM_Detokenize": "LLM_Detokenize",
    "LLM_Reset": "LLM_Reset",
    "LLM_Eval": "LLM_Eval",
    "LLM_Sample": "LLM_Sample",
    "Call LLM":"Call LLM",
}


