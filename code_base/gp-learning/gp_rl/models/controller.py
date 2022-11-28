from loguru import logger
import torch

float_type = torch.float32
torch.set_default_dtype(torch.float32)


class Controller:
    def __init__(self, **kwargs):
        super(Controller, self).__init__()
        logger.debug("===== Configuring controller with parameters =====")
        for key, value in kwargs.items():
            setattr(self, key, value)
            logger.debug(f"attribute {key}: {value}")
        # set device
        if not self.force_cpu:
            self.set_device()
        else:
            logger.info("Forcing CPU as processor...")
            self.set_device_cpu()

    def init_controller(self):
        if self.controller_type == "Linear":
            logger.info("initializing Linear controller")
            self.linear_model = torch.nn.Linear(
                self.state_dim, self.control_dim, device=self.device, dtype=self.dtype
            )
            self.saturation = torch.nn.Hardtanh()
            self.controller = torch.nn.Sequential(self.linear_model, self.saturation)
        elif self.controller_type == "NN":
            logger.info(f"initializing NN controller with layers {self.NNlayers}")
            self.controller = torch.nn.Sequential(
                torch.nn.Linear(
                    self.state_dim,
                    self.NNlayers[0],
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.nn.Linear(
                    self.NNlayers[0],
                    self.NNlayers[1],
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.nn.Linear(
                    self.NNlayers[1],
                    self.control_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.nn.Hardtanh(),
            )
        else:
            raise Exception("Unknown controller type {}".format(self.controller_type))
        logger.debug(f"Controller Model: {self.controller}")
        self.controller.predict = self.controller.forward
        return self.controller

    def set_device(self):
        self.is_cuda = torch.cuda.is_available()
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        logger.info(f"using GPU: {self.is_cuda} - using processor: *({self.device})")

    def set_device_cpu(self):
        self.is_cuda = False
        # self.Tensortype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        self.dtype = torch.float32
        self.device = torch.device("cuda:0") if self.is_cuda else torch.device("cpu")
        logger.info(f"Forcing CPU... using processor: *({self.device})")
