import torch
import numpy as np
from loguru import logger


def get_tensor(data, device=torch.device("cuda:0"), dtype=torch.float32):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype, device=device)


def set_device(self):
    self.is_cuda = torch.cuda.is_available()
    self.dtype = torch.float32
    self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
    logger.info(f"using GPU: {self.is_cuda} - using processor: *({self.device})")


def set_device_cpu(self):
    self.is_cuda = False
    self.dtype = torch.float32
    self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
    logger.info(f"Forcing CPU... using processor: *({self.device})")
