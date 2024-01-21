
import sys
import os
import folder_paths
from typing import List
from collections.abc import Sequence
from .logger import logger


# Folder path and venv
llama_dir = os.path.dirname(os.path.realpath(__file__))
venv_site_packages = os.path.join(folder_paths.base_path, 'venv', 'Lib', 'site-packages')
sys.path.append(venv_site_packages)


# Attempt to get llama_cpp if it doesn't exist
try:
    from llama_cpp import Llama
except ImportError:

    logger.warn("Unable to find llama-cpp-python, attempting to fix.")
    # Determine the correct path based on the operating system
    if os.name == 'posix':
        site_packages = os.path.join(sys.prefix, 'lib', 'python{}.{}/site-packages'.format(sys.version_info.major, sys.version_info.minor))
    else:  # For Windows
        site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

    sys.path.append(site_packages)
    try:
        from llama_cpp import Llama
        logger.info("Successfully acquired llama-cpp-python.")
    except ImportError:
        logger.exception("Nope.  Actually unable to find llama-cpp-python.")


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
            logger.exception("The model path does not exist.  Perhaps hit Ctrl+F5 and try reloading it.")

        return (llm,)



class LLM_Load_Model_Advanced:
    """
    Load a llama.cpp model from model_path.
    (advanced version)

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__

    Missing:
        tensor-split
        kv_overrides
        chat_handler
        **kwargs

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
                "split_mode": (["LLAMA_SPLIT_NONE", "LLAMA_SPLIT_LAYER", "LLAMA_SPLIT_ROW"], {"default": "LLAMA_SPLIT_LAYER"}),
                "main_gpu": ("INT", {"default": 0, "min":0}),
                "vocab_only": ("BOOLEAN", {"default": False}),
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),
                "n_ctx": ("INT", {"default": 512, "min":0, "step":512}),
                "n_batch": ("INT", {"default": 512, "min":0, "step":512}),
                "n_threads": ("INT", {"default": None}),
                "n_threads_batch": ("INT", {"default": None}),
                "rope_scaling_type": (["LLAMA_ROPE_SCALING_UNSPECIFIED","LLAMA_ROPE_SCALING_NONE","LLAMA_ROPE_SCALING_LINEAR","LLAMA_ROPE_SCALING_YARN"],
{"default": "LLAMA_ROPE_SCALING_UNSPECIFIED"}),
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
        split_mode:str,
        main_gpu:int, 
        tensor_split:List[float],
        vocab_only:bool,
        use_mmap:bool,
        use_mlock:bool,
        seed:int,
        n_ctx:int,
        n_batch:int,
        n_threads:int,
        n_threads_batch:int,
        rope_scaling_type:str,
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

        LLAMA_SPLIT = {
            'LLAMA_SPLIT_NONE': 0,
            'LLAMA_SPLIT_LAYERS': 1,
            'LLAMA_SPLIT_ROWS': 2,
        }

        LLAMA_ROPE_SCALING = {
            'LLAMA_ROPE_SCALING_UNSPECIFIED': -1,
            'LLAMA_ROPE_SCALING_NONE': 0,
            'LLAMA_ROPE_SCALING_LINEAR': 1,
            'LLAMA_ROPE_SCALING_YARN': 2,
        }

        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers, 
                split_mode=LLAMA_SPLIT[split_mode],
                main_gpu=main_gpu, 
                vocab_only=vocab_only,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                seed=seed,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                rope_scaling_type=LLAMA_ROPE_SCALING[rope_scaling_type],
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
            logger.exception("The model path does not exist.  Perhaps hit Ctrl+F5 and try reloading it.")
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
                "text": ("STRING", {"default": "", "multiline":True, })
            },
            "optional": {
                "add_bos": ("BOOLEAN", {"default": True}),
                "special": ("BOOLEAN", {"default": False}),
            }
        }


    OUTPUT_IS_LIST = (True, )
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, text:str, add_bos:bool, special:bool):

        try:
            tokens = LLM.tokenize(
                text=text.encode('utf-8'),
                add_bos=add_bos,
                special=special)

        except RuntimeError:
            logger.exception("RuntimeError: If the tokenization failed. ")

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
                "tokens": ("INT", {"default":[0], "forceInput":True,}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM, tokens):
        try:
            tokens = [tokens] if isinstance(tokens, int) else list(tokens)
            thebytes = LLM.detokenize(tokens)
            thestring = thebytes.decode('utf-8')
            return (thestring,)
        except Exception as e:
            print(f"Error in detokenize method: {e}")
            # Add more detailed error handling as needed
            return None


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
                "tokens": ("INT", {"default":[0], "forceInput":True,})
            },
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, tokens):

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
    RETURN_TYPES = ("INT", )
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama):

        thebytes = LLM.sample(
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty)
        return (thebytes, )


class LLM_Generate:
    """
    Create a generator of tokens from a prompt.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.generate

    Problems:
        tokens needs to be a Sequence[int]

    Yields:
        int
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "tokens":("TOKENS",),
            },
            "optional": {
                "top_k":("INT",{"default":40}),
                "top_p":("FLOAT",{"default":0.95}),
                "min_p":("FLOAT",{"default":0.05}),
                "typical_p":("FLOAT",{"default":1.0}),
                "temp":("FLOAT",{"default":0.8}),
                "repeat_penalty":("FLOAT",{"default":1.1}), 
                "reset":("BOOLEAN",{"default":True}),
                "frequency_penalty":("FLOAT",{"default":0.0}), 
                "presence_penalty":("FLOAT",{"default":0.0}), 
                "tfs_z":("FLOAT",{"default":1.0}), 
                "microstat_mode":("INT",{"default":0}),
                "microstat_tau":("FLOAT",{"default":5.0}), 
                "microstat_eta":("FLOAT",{"default":0.1}), 
                "penalize_nl":("BOOLEAN",{"default":True}),
                "logits_processor":("STRING",{"default":None}),
                "stopping_criteria":("STRING",{"default":None}),
                "grammar":("STRING",{"default":None}),
            }
        }

    RETURN_TYPES = ("GENERATOR",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, 
        LLM, 
        tokens:Sequence[int],
        top_k:int,
        top_p:float,
        min_p:float,
        typical_p:float,
        temp:float,
        repeat_penalty:float,
        reset:bool,
        frequency_penalty:float,
        presence_penalty:float,
        tfs_z:float,
        microstat_mode:int,
        microstat_tau:float,
        microstat_eta:float,
        penalize_nl:bool,
        logits_processor,
        stopping_criteria,
        grammar):

        generator = LLM.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
            reset=reset,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tfs_z=tfs_z,
            microstat_mode=microstat_mode,
            microstat_tau=microstat_tau,
            microstat_eta=microstat_eta,
            penalize_nl=penalize_nl,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            grammar=grammar)

        return (generator, )


class LLM_Create_Embedding:
    """
    Embed a string.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_embedding

    Bug: return value is an Embedding Object, but LLM_Embed takes a string.  What gives?
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
            },
            "optional": {
                "input_str":("STRING", {"default":"", "multiline":True, }), # Union[str, List[str]]
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("EMBEDDING",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, input_str:str):

        embeddingResponse = LLM.create_embedding(input_str=input_str)
        return (embeddingResponse, )


class LLM_Embed:
    """
    Embed a string.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.embed
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
            },
            "optional": {
                "input_str":("STRING", {"default":"", "multiline":True, }),
            }
        }
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, input_str:str):

        list_of_floats = LLM.embed(input_str=input_str)
        return (list_of_floats, ) # List[float] - A list of embeddings


class LLM_Create_Completion:
    """
    Generate text from a prompt.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion
    Missing:
        stream
        logprobs
        stop (currently stop on TOKEN_EOS)
        stopping_criteria
        logits_processor
        grammar
        logits_bias
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "prompt":("STRING", {"default":"", "multiline":True}),
            },
            "optional": {
                "suffix":("STRING", {"default":"", "multiline":True}),
                "max_response_tokens": ("INT", {"default": 16}),
                "temperature": ("FLOAT", {"default": 0.8, "min":0.0, "max":1.0, "step":0.01, "round":0.01, "display":"number"}),
                "top_p":("FLOAT", {"default":0.95}),
                "min_p":("FLOAT", {"default":0.05}),
                "typical_p":("FLOAT", {"default":1.0}),
                "echo":("BOOLEAN", {"default":False}),
                "frequency_penalty":("FLOAT", {"default":0.0}),
                "presence_penalty":("FLOAT", {"default":0.0}),
                "repeat_penalty":("FLOAT", {"default":1.1}),
                "top_k": ("INT", {"default": 40}),
                "seed": ("INT", {"default": -1}),
                "tfs_z":("FLOAT", {"default":1.0}),
                "mirostat_mode": ("INT", {"default": 0}),
                "mirostat_tau":("FLOAT", {"default":5.0}),
                "mirostat_eta":("FLOAT", {"default":0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(
        self, 
        LLM:Llama, 
        prompt:str, 
        suffix:str,
        max_response_tokens:int,
        temperature:float,
        top_p:float,
        min_p:float,
        typical_p:float,
        echo:bool,
        frequency_penalty:float,
        presence_penalty:float,
        repeat_penalty:float,
        top_k:int,
        seed:int,
        tfs_z:float,
        mirostat_mode:int,
        mirostat_tau:float,
        mirostat_eta:float):

        try:
            response = LLM.create_completion(
                prompt=prompt,
                suffix=suffix,
                max_tokens=max_response_tokens, 
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                echo=echo,
                stop=LLM.token_eos(),
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                seed=seed, 
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta)
        except ValueError:
            logger.exception('ValueError: If the requested tokens exceed the context window.');
        except RuntimeError:
            logger.exception('RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.')
        return (response['choices'][0]['text'], )


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
            logger.exception('ValueError: If the requested tokens exceed the context window.');
        except RuntimeError:
            logger.exception('RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.')
        return (response['choices'][0]['text'], )


class LLM_Call_Advanced:
    """
    Generate text from a prompt.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__call__
    Missing:
        stream
        logprobs
        stop (currently stop on TOKEN_EOS)
        stopping_criteria
        logits_processor
        grammar
        logits_bias
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "prompt":("STRING", {"default":"", "multiline":True}),
            },
            "optional": {
                "suffix":("STRING", {"default":"", "multiline":True}),
                "max_response_tokens": ("INT", {"default": 16}),
                "temperature": ("FLOAT", {"default": 0.8, "min":0.0, "max":1.0, "step":0.01, "round":0.01, "display":"number"}),
                "top_p":("FLOAT", {"default":0.95}),
                "min_p":("FLOAT", {"default":0.05}),
                "typical_p":("FLOAT", {"default":1.0}),
                "echo":("BOOLEAN", {"default":False}),
                "frequency_penalty":("FLOAT", {"default":0.0}),
                "presence_penalty":("FLOAT", {"default":0.0}),
                "repeat_penalty":("FLOAT", {"default":1.1}),
                "top_k": ("INT", {"default": 40}),
                "seed": ("INT", {"default": -1}),
                "tfs_z":("FLOAT", {"default":1.0}),
                "mirostat_mode": ("INT", {"default": 0}),
                "mirostat_tau":("FLOAT", {"default":5.0}),
                "mirostat_eta":("FLOAT", {"default":0.1}),
            }
        }

    # ComfyUI will effectively return the text result of the Llama.__call__() as a STRING
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(
        self, 
        LLM:Llama, 
        prompt:str, 
        suffix:str,
        max_response_tokens:int,
        temperature:float,
        top_p:float,
        min_p:float,
        typical_p:float,
        echo:bool,
        frequency_penalty:float,
        presence_penalty:float,
        repeat_penalty:float,
        top_k:int,
        seed:int,
        tfs_z:float,
        mirostat_mode:int,
        mirostat_tau:float,
        mirostat_eta:float):

        # I'm using __call__ and not generate() because seed isn't available in generate!
        try:
            response = LLM.__call__(
                prompt=prompt,
                suffix=suffix,
                max_tokens=max_response_tokens, 
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                echo=echo,
                stop=LLM.token_eos(),
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                top_k=top_k,
                seed=seed, 
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta)
        except ValueError:
            logger.exception('ValueError: If the requested tokens exceed the context window.');
        except RuntimeError:
            logger.exception('RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.')
        return (response['choices'][0]['text'], )


class LLM_Save_State:
    """
    No idea

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.save_state
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
            },
            "optional": {
            }
        }
    RETURN_TYPES = ("STATE",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, input_str:str):

        state = LLM.save_state()
        return (state, )

class LLM_Load_State:
    """
    No idea

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.load_state
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
                "STATE":("STATE",),
            },
            "optional": {
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama, state):

        LLM.load_state(state=state)
        return None


class LLM_Token_BOS:
    """
    Return the beginning-of-sequence token.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.token_bos
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
            },
            "optional": {
            }
        }
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama):

        token = LLM.token_bos()
        return (token, )


class LLM_Token_EOS:
    """
    Return the end-of-sequence token.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.token_eos
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM",),
            },
            "optional": {
            }
        }
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self, LLM:Llama):

        token = LLM.token_eos()
        return (token, )

# Bugged ones:
# "LLM_Create_Embedding":LLM_Create_Embedding,
# "LLM_Generate":LLM_Generate,

NODE_CLASS_MAPPINGS = {
    "Load LLM Model Basic":LLM_Load_Model,
    "Call LLM Basic":LLM_Call,
    "LLM_Tokenize":LLM_Tokenize,
    "LLM_Detokenize":LLM_Detokenize,
    "LLM_Reset":LLM_Reset,
    "LLM_Eval":LLM_Eval,
    "LLM_Sample":LLM_Sample,
    "LLM_Embed":LLM_Embed,
    "LLM_Save_State":LLM_Save_State,
    "LLM_Load_State":LLM_Load_State,
    "LLM_Token_BOS":LLM_Token_BOS,
    "LLM_Token_EOS":LLM_Token_EOS,
    "Load LLM Model Advanced":LLM_Load_Model_Advanced,
    "Call LLM Advanced":LLM_Call_Advanced,
    "LLM_Create_Completion Advanced":LLM_Create_Completion,
}

# Bugged ones:
# "LLM Create Embedding":"LLM Create Embedding",
# "LLM Generate":"LLM Generate",

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load LLM Model Basic": "Load LLM Model Basic",
    "Call LLM Basic":"Call LLM Basic",
    "LLM Tokenize": "LLM Tokenize",
    "LLM Detokenize": "LLM Detokenize",
    "LLM Reset": "LLM Reset",
    "LLM Eval": "LLM Eval",
    "LLM Sample": "LLM Sample",
    "LLM Embed":"LLM Embed",
    "LLM_Save_State":"LLM_Save_State",
    "LLM_Load_State":"LLM_Load_State",
    "LLM_Token_BOS":"LLM_Token_BOS",
    "LLM_Token_EOS":"LLM_Token_EOS",
    "Load LLM Model Advanced": "Load LLM Model Advanced",
    "Call LLM Advanced":"Call LLM Advanced",
    "LLM Create Completion Advanced":"LLM Create Completion Advanced",
}


