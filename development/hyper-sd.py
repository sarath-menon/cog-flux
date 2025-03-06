#%%
import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download
base_model_id = "black-forest-labs/FLUX.1-dev"
repo_name = "ByteDance/Hyper-SD"
# Take 8-steps lora as an example
ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"

#%%

pipe = FluxPipeline.from_pretrained(base_model_id, token="xxx")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora(lora_scale=0.125)
pipe.to("cuda", dtype=torch.float16)

print("Generating image...")
image=pipe(prompt="a very british rodent sipping tea", num_inference_steps=8, guidance_scale=3.5).images[0]
image.save("output.png")
print("Image saved as output.png")


