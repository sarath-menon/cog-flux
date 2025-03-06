#%%
import torch
from diffusers import FluxPipeline,  FluxTransformer2DModel
from diffusers.utils import load_image
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from diffusers import AutoencoderKL

#%%
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat163333
# ).to("cuda")

LORA_WEIGHTS_PATH = "/workspace/models/lora_training/FLUX.1-dev"
text_encoder = CLIPTextModel.from_pretrained(LORA_WEIGHTS_PATH + "/text_encoder")
text_encoder_2 = T5EncoderModel.from_pretrained(LORA_WEIGHTS_PATH + "/text_encoder_2")

tokenizer = CLIPTokenizer.from_pretrained(LORA_WEIGHTS_PATH + "/tokenizer")
tokenizer_2 = T5TokenizerFast.from_pretrained(LORA_WEIGHTS_PATH + "/tokenizer_2")
vae = AutoencoderKL.from_pretrained(LORA_WEIGHTS_PATH + "/vae")
transformer = FluxTransformer2DModel.from_pretrained(LORA_WEIGHTS_PATH + "/transformer")

#%%


pipe = FluxPipeline.from_pretrained(
                transformer = transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                torch_dtype=torch.bfloat16,
            ).to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flux_ip_adapter_input.jpg").resize((1024, 1024))

pipe.load_ip_adapter(
    "XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
)
pipe.set_ip_adapter_scale(1.0)

#%%
image = pipe(
    width=1024,
    height=1024,
    prompt="wearing sunglasses",
    negative_prompt="",
    true_cfg=4.0,
    generator=torch.Generator().manual_seed(4444),
    ip_adapter_image=image,
).images[0]

image.save('flux_ip_adapter_output.jpg')