
import sys
import os
import folder_paths
from typing import List
from .logger import logger


# Folder path and venv
llama_dir = os.path.dirname(os.path.realpath(__file__))
venv_site_packages = os.path.join(folder_paths.base_path, 'venv', 'Lib', 'site-packages')
sys.path.append(venv_site_packages)


# Attempt to get semantic_router if it doesn't exist
try:
    from llama_cpp import Llama
    from semantic_router import RouteLayer
    from semantic_router.encoders import HuggingFaceEncoder
    from semantic_router.utils.function_call import get_schema

except ImportError:

    logger.warn("Unable to find semantic_router, attempting to fix.")
    # Determine the correct path based on the operating system
    if os.name == 'posix':
        site_packages = os.path.join(sys.prefix, 'lib', 'python{}.{}/site-packages'.format(sys.version_info.major, sys.version_info.minor))
    else:  # For Windows
        site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

    sys.path.append(site_packages)
    try:
        from llama_cpp import Llama
        from semantic_router import RouteLayer
        from semantic_router.encoders import HuggingFaceEncoder
        from semantic_router.utils.function_call import get_schema
        logger.info("Successfully acquired semantic_router.")
    except ImportError:
        logger.exception("Nope.  Actually unable to find semantic_router.")



class Define_Route:
    """
    Create a Route

    https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default":""}),
                "prompt": ("STRING", {"default":""}),
            },
        }

    RETURN_TYPES = ("Route",)
    FUNCTION = "execute"
    CATEGORY = "LLM"
    INPUT_IS_LIST = True

    def execute(self, name:str, utterances:str):

        route = Route(
            name=name,
            utterances=utterances,
        )

        return (route,)


class HuggingFaceEncoder:
    """
    Create a Route

    https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("HuggingFaceEncoder",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(self):
        hfe = HuggingFaceEncoder()
        return (hfe,)


class Bind_Router:
    """
    Bind a routelayer to the LLM

    https://github.com/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM":("LLM", ),
                "name":("STRING",{"default":""}),
                "encoder":("HuggingFaceEncoder", ),
                "routes":("Route", ),
            },
        }

    RETURN_TYPES = ("LLM","Router",)
    FUNCTION = "execute"
    CATEGORY = "LLM"

    def execute(
        self,
        LLM:Llama,
        name:str,
        encoder:HuggingFaceEncoder,
        routes):

        new_llm = LlamaCppLLM(
            name=name,
            llm=LLM)

        rl = RouteLayer(encoder=encoder, routes=routes, llm=new_llm)
        
        return new_llm, rl



NODE_CLASS_MAPPINGS = {
    "Define_Route":Define_Route,
    "HuggingFaceEncoder":HuggingFaceEncoder,
    "Bind_Router":Bind_Router,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Define_Route": "Define Route",
    "HuggingFaceEncoder": "HuggingFace Encoder",
    "Bind_Router": "Bind_Router",
}


