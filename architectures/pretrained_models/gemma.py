import torch
import torch.nn as nn
from typing import List, Dict
from transformers.models.gemma2.modeling_gemma2 import Gemma2Model
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast


################################################################################################
################################################################################################
class Gemma(nn.Module):
    def __init__(self, compile: bool = False, device: str = None):
        super().__init__()
        self.compile = compile
        self.model_name = "google/gemma-2-2b"
        self.tokenizer = GemmaTokenizerFast.from_pretrained(
            self.model_name,
            padding_size="right",
            cache_dir="../pretrained_models/models",
        )
        self.text_model = Gemma2Model.from_pretrained(
            self.model_name,
            cache_dir="../pretrained_models/models",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        # not sure if set False makes a big difference in run time.
        for param in self.text_model.parameters():
            param.requires_grad = False
        if device:
            self.device = device
            self.text_model = self.text_model.to(self.device)
        if compile:
            self.text_model = torch.compile(self.text_model)

    def _tokenize_text(self, text: List[str]) -> Dict:
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,  # TODO: set in config
        )  # .to(self.device)
        return tokenized_text

    @torch.no_grad()
    def _get_embeddings(self, model_inputs: Dict) -> Dict:
        out = self.text_model(**model_inputs).last_hidden_state
        attention_masks = model_inputs["attention_mask"]
        # we need to make the values that belong to the padding masks to be 0
        # attention mask shape [Batch, Seq Len]
        # hidden_state shape [Batch, Seq Len, Features]
        # we want to make the Features of padding mask locations along the seq len = 0
        out = out * attention_masks[:, :, None].to(out.dtype)
        return {"hidden_states": out}

    @torch.no_grad()
    def forward(self, text: List[str], device=None) -> Dict:
        inputs = self._tokenize_text(text).to(device)
        embeddings = self._get_embeddings(inputs)["hidden_states"]
        # return shape [B, S, F]
        return {"hidden_states": embeddings}
