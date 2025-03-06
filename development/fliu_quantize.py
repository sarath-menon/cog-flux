#%%
import dotenv
dotenv.load_dotenv()

import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel, CLIPTextModel
from optimum.quanto import freeze, qfloat8, quantize

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

LORA_WEIGHTS_PATH = "/workspace/models/lora_training/FLUX.1-dev"
T5_CACHE = f"{LORA_WEIGHTS_PATH}/text_encoder_2/model-00001-of-00002.safetensors"
#%%

transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", torch_dtype=dtype)
quantize(transformer, weights=qfloat8)
freeze(transformer)
# %%
text_encoder_2 = T5EncoderModel.from_pretrained(LORA_WEIGHTS_PATH, subfolder="text_encoder_2", torch_dtype=dtype)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)
# %%
pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2
# pipe.enable_model_cpu_offload()
# %%

from huggingface_hub import hf_hub_download

repo_name = "ByteDance/Hyper-SD"
# Take 8-steps lora as an example
ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora(lora_scale=0.125)
pipe.to("cuda", dtype=torch.float16)

# %%
