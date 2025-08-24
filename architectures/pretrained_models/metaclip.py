import torch
import torch.nn as nn
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel


################################################################################################
################################################################################################
class MetaCLIP(nn.Module):
    def __init__(
        self, model_name: str, compile: bool = False, device: str = None
    ) -> None:
        super().__init__()
        self.tokenizer = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir="../pretrained_models/models",
            use_fast=True,
            padding_size="right",
        ).tokenizer
        self.text_model = CLIPModel.from_pretrained(
            model_name,
            cache_dir="../pretrained_models/models",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).text_model.eval()

        # not sure if set False makes a big difference in run time.
        for param in self.text_model.parameters():
            param.requires_grad = False
        if compile:
            self.text_model = torch.compile(self.text_model)

        if device:
            self.device = device
            self.text_model = self.text_model.to(self.device)

    def _tokenize_text(self, text: List[str]) -> Dict:
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )  # .to(self.device)
        return tokenized_text

    @torch.no_grad()
    # @torch.inference_mode()
    def _get_embeddings(self, model_inputs: Dict) -> Dict:
        out = self.text_model(**model_inputs)
        attention_masks = model_inputs["attention_mask"]
        # we need to make the values that belong to the padding masks to be 0
        # attention mask shape [Batch, Seq Len]
        # hidden_state shape [Batch, Seq Len, Features]
        # we want to make the Features of padding mask locations along the seq len = 0
        last_hidden_state = out.last_hidden_state * attention_masks[:, :, None].to(
            out.last_hidden_state.dtype
        )
        return {"hidden_states": last_hidden_state, "pooled": out.pooler_output}

    @torch.no_grad()
    def forward(
        self, text: List[str], return_pooled: bool = False, device=None
    ) -> Dict:
        inputs = self._tokenize_text(text).to(device)
        embeddings = self._get_embeddings(inputs)
        return {
            "hidden_states": embeddings["hidden_states"],  # [B, S, F]
            "pooled": embeddings["pooled"] if return_pooled else None,  # [B, F]
        }
