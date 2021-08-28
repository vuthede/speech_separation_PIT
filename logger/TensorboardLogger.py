
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import torch


class TensorBoardLogger():
    def __init__(self, root="./", experiment_name="experiment1"):
        self.root = root
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(f'{root}/{experiment_name}')
    
    def log(self, metric="loss", value=0.0, step=0):
        self.writer.add_scalar(metric, value, step)

    def log_image(self, name, img, kind="single"):
        assert kind in ["single", "NCHW", "NHWC"]

        if kind == "single":
            self.writer.add_image(name, img)
        elif kind == "NCHW":
            images = torch.FloatTensor(img)
            img_grid = torchvision.utils.make_grid(images)
            self.writer.add_image(name, img_grid)
        elif kind == "NHWC":
            img = np.array(img)
            images = np.transpose(img, (0, 3, 1, 2))
            images = torch.FloatTensor(images)
            img_grid = torchvision.utils.make_grid(images)
            self.writer.add_image(name, img_grid)
        else:
            raise NotImplementedError
