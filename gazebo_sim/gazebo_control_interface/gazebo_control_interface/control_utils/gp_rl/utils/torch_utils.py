import torch
from loguru import logger

DEFAULT_DEVICE = torch.device("cuda:0")
DEFAULT_DTYPE = torch.float32


def get_tensor(data, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    else:
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


def to_gpu(obj):
    obj.to(torch.device("cuda:0"), dtype=torch.float32)
    