from torch import Tensor, nn
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer

CLIP_PATH = '/workspace/models/lora_training/FLUX.1-dev/text_encoder'
TOKENIZER_1_PATH = '/workspace/models/lora_training/FLUX.1-dev/tokenizer'

TOKENIZER_2_PATH = '/workspace/models/lora_training/FLUX.1-dev/tokenizer_2'
T5_PATH = '/workspace/models/lora_training/FLUX.1-dev/text_encoder_2'

class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, is_clip=False, **hf_kwargs):
        super().__init__()
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_1_PATH, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(CLIP_PATH, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_2_PATH, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(T5_PATH, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
    

class PreLoadedHFEmbedder(nn.Module):
    """
    Does the same thing as the HFEmbedder, but lets you share the tokenizer & hf module. Could also just share the HFEmbedder but here we are.
    """
    def __init__(self, is_clip: bool, max_length: int, tokenizer, hf_module):
        super().__init__()
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        self.tokenizer = tokenizer
        self.hf_module = hf_module

        self.hf_module = self.hf_module.eval().requires_grad_(False)
    
    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
