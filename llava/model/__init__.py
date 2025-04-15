# import os

# AVAILABLE_MODELS = {
#     "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
#     "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
#     "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
#     "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
#     # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
#     # Add other models as needed
# }

# for model_name, model_classes in AVAILABLE_MODELS.items():
#     try:
#         exec(f"from .language_model.{model_name} import {model_classes}")
#     except Exception as e:
#         print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")


import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        # Updated import path - note how we reference the module
        # Using the full path from the project root
        exec(f"from llava.model.language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.model.language_model.{model_name}. Error: {e}")