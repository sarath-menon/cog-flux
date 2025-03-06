#%%

import dotenv
dotenv.load_dotenv()

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download
base_model_id = "/workspace/models/lora_training/FLUX.1-dev"
repo_name = "ByteDance/Hyper-SD"
# Take 8-steps lora as an example
ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
import os

#%%

pipe = FluxPipeline.from_pretrained(base_model_id, token=os.environ["HF_TOKEN"])
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora(lora_scale=0.125)
pipe.to("cuda", dtype=torch.float16)

#%%

pipe.unload_lora_weights()
# save locally
pipe.save_pretrained("path/to/fused-pipeline")

#%%
print("Generating image...")
image=pipe(prompt="a very british rodent sipping tea", num_inference_steps=8, guidance_scale=3.5).images[0]
image.save("output.png")
print("Image saved as output.png")



# %%


