import torch
import torch.nn as nn
from typing import Dict
from architectures.pretrained_models.metaclip import MetaCLIP
from architectures.pretrained_models.gemma import Gemma


################################################################################################
################################################################################################
class CaptionEncoder(nn.Module):
    # TODO: Make this class extensible/reconfigurageble to other models easily. Also refactor.
    def __init__(self, config):
        super().__init__()
        # clip small and large
        self.clip_s = MetaCLIP(config.clip_s_name, device=None).eval()
        self.clip_l = MetaCLIP(config.clip_l_name, device=None).eval()

        # llm - gemma
        self.llm = Gemma(device=None).eval()

    def forward(self, text, device) -> Dict:
        clip_s_out = self.clip_s.forward(text, return_pooled=True, device=device)
        clip_l_out = self.clip_l.forward(text, return_pooled=True, device=device)
        llm_out = self.llm.forward(text, device=device)
        out = {
            "clip_s_hidden": clip_s_out["hidden_states"].to(torch.float32),
            "clip_s_pooled": clip_s_out["pooled"].to(torch.float32),
            "clip_l_hidden": clip_l_out["hidden_states"].to(torch.float32),
            "clip_l_pooled": clip_l_out["pooled"].to(torch.float32),
            "llm_hidden": llm_out["hidden_states"].to(torch.float32),
        }
        return out


################################################################################################
################################################################################################
