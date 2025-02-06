import torch

def isCUDA():
    if torch.cuda.is_available():
        return True
    return False