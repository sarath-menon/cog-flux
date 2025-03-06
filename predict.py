import os
from typing import Tuple, List
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass

from bfl_predictor import BflBf16Predictor, BflControlNetFlux, BflFillFlux, BflFp8Flux
from diffusers_predictor import DiffusersFlux
from flux.modules.conditioner import PreLoadedHFEmbedder
from fp8.util import LoadedModels
from weights import WeightsDownloadCache

# Configure torch settings
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20

# Constants
FLUX_DEV = "flux-dev"
FLUX_DEV_FP8 = "flux-dev-fp8"
MAX_IMAGE_SIZE = 1440

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    # "16:9": (1344, 768),
    # "21:9": (1536, 640),
    # "3:2": (1216, 832),
    # "2:3": (832, 1216),
    # "4:5": (896, 1088),
    # "5:4": (1088, 896),
    # "3:4": (896, 1152),
    # "4:3": (1152, 896),
    # "9:16": (768, 1344),
    # "9:21": (640, 1536),
}

@dataclass(frozen=True)
class Inputs:
    prompt: str = "Prompt for generated image"
    aspect_ratio: str = "1:1"
    num_outputs: int = 1
    seed: int = None
    output_format: str = "webp"
    output_quality: int = 80
    disable_safety_checker: bool = False
    lora_weights: str = None
    lora_scale: float = 1.0
    megapixels: str = "1"

class Predictor:
    def __init__(self):
        self.base_setup()
        self.bf16_model = BflBf16Predictor(FLUX_DEV, offload=self.should_offload(),restore_lora_from_cloned_weights=True)
        self.weights_cache = WeightsDownloadCache()

        self.fp8_model = BflFp8Flux(
            FLUX_DEV_FP8,
            loaded_models=self.bf16_model.get_shared_models(),
            torch_compile=False,
            compilation_aspect_ratios=ASPECT_RATIOS,
            offload=self.should_offload(),
            weights_download_cache=self.weights_cache,
            restore_lora_from_cloned_weights=True,
        )

    def base_setup(self):
        # Setup code for safety checkers and other initialization
        pass

    def should_offload(self):
        total_mem = torch.cuda.get_device_properties(0).total_memory
        self.offload = total_mem < 48 * 1024**3
        if self.offload:
            print("GPU memory is:", total_mem / 1024**3, ", offloading models")
        return self.offload

    def predict(
        self,
        prompt: str = Inputs.prompt,
        aspect_ratio: str = Inputs.aspect_ratio,
        image: str = None,
        prompt_strength: float = 0.80,
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = 28,
        guidance: float = 3,
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        go_fast: bool = True,
        lora_weights: str = Inputs.lora_weights,
        lora_scale: float = Inputs.lora_scale,
        megapixels: str = Inputs.megapixels,
        
    ) -> List[str]:
        if image and go_fast:
            print("img2img not supported with fp8 quantization; running with bf16")
            go_fast = False

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        model = self.fp8_model if go_fast else self.bf16_model
        
        model.handle_loras(lora_weights, lora_scale)

        imgs, np_imgs = model.predict(
            prompt,
            num_outputs,
            num_inference_steps,
            guidance=guidance,
            legacy_image_path=image,
            prompt_strength=prompt_strength,
            seed=seed,
            width=width,
            height=height,
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )

    def size_from_aspect_megapixels(self, aspect_ratio: str, megapixels: str = "1") -> Tuple[int, int]:
        width, height = ASPECT_RATIOS[aspect_ratio]
        if megapixels == "0.25":
            width, height = width // 2, height // 2
        return (width, height)

    def postprocess(self, images, disable_safety_checker, output_format, output_quality, np_images) -> List[str]:
        # Postprocessing code for saving images and safety checking
        output_paths = []
        for i, img in enumerate(images):
            output_path = f"out-{i}.{output_format}"
            save_params = {"quality": output_quality, "optimize": True} if output_format != "png" else {}
            img.save(output_path, **save_params)
            output_paths.append(output_path)
        return output_paths

def make_multiple_of_16(n):
    return ((n + 15) // 16) * 16

if __name__ == "__main__":
    predictor = Predictor()


    # results = predictor.predict(prompt="A man eating ice cream")


    results = predictor.predict(prompt="A handsome man in a suit, 25 years old, cool, futuristic, eating a pizza", lora_weights="https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/mjv6_lora.safetensors"
    )

    # results = predictor.predict(prompt="A man eating ice cream", image = "/home/cog-flux/resources/images/cr7.png")

    print(f"Generated images: {results}")