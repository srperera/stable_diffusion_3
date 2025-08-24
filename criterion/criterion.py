import torch.nn as nn
from typing import Dict


#############################################################################################
#############################################################################################
class MSELoss:
    def __init__(self, loss_args: Dict = None):
        self._loss = nn.MSELoss()

    def __call__(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


#############################################################################################
#############################################################################################
def build_loss_fn(loss_type: str, loss_args: Dict = None):
    if loss_type == "mse":
        return MSELoss(loss_args)
    else:
        raise ValueError("Specified loss function is not supported")


#############################################################################################
#############################################################################################
