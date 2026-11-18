import logging

import comfy.model_management
import comfy_kitchen

from .int8_attention import int8_attention_override
from .lora_bake import bake_patches


class MossBooster:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "bake_lora": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Merge the LoRAs into the weights once instead of every step."},
                ),
                "int8_attention": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "INT8 tensor-core attention, about 2.5x faster than BF16."},
                ),
                "int8_start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "This share of the first steps keeps exact attention.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model"

    def apply(self, model, bake_lora, int8_attention, int8_start_percent):
        if bake_lora and model.patches and model.weight_wrapper_patches:
            logging.warning("MOSS Booster: the model has weight wrappers, LoRAs stay unbaked")
            model = model.clone()
        elif bake_lora and model.patches:
            model = bake_patches(model)
        else:
            model = model.clone()

        if int8_attention:
            if comfy_kitchen.int8_attention_is_available(comfy.model_management.get_torch_device()):
                override = int8_attention_override(int8_start_percent)
                model.model_options["transformer_options"]["optimized_attention_override"] = override
            else:
                logging.warning(
                    "MOSS Booster: INT8 attention is not available on this GPU, using the default attention"
                )
        return (model,)


NODE_CLASS_MAPPINGS = {"MossBooster": MossBooster}
NODE_DISPLAY_NAME_MAPPINGS = {"MossBooster": "MOSS Booster"}
