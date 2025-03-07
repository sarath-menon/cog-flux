#%%
import dotenv
dotenv.load_dotenv()

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

ckpt_path = (
    "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
)   
#%%
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

#%%
prompt = "An elephant holding a sign that says hello world"
image = pipe(prompt, generator=torch.manual_seed(0)).images[0]
image.save("flux-gguf.png")
# %%
