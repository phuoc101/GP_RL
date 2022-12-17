from loguru import logger
import torch

from gp_rl.utils.torch_utils import set_device, set_device_cpu

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
            set_device(self)
        else:
            logger.info("Forcing CPU as processor...")
            set_device_cpu(self)

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
                    bias=False,
                ),
                torch.nn.Linear(
                    self.NNlayers[0],
                    self.NNlayers[1],
                    device=self.device,
                    dtype=self.dtype,
                    bias=False,
                ),
                torch.nn.Linear(
                    self.NNlayers[1],
                    self.control_dim,
                    device=self.device,
                    dtype=self.dtype,
                    bias=False,
                ),
                torch.nn.Hardtanh(),
            )
        else:
            raise Exception("Unknown controller type {}".format(self.controller_type))
        logger.debug(f"Controller Model: {self.controller}")
        self.controller.predict = self.controller.forward
        return self.controller
