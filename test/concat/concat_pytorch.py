import torch
from torch.autograd import Variable
import torch.onnx
import torch.nn as nn
import numpy
import torch.nn.functional as F

class model_constant(nn.Module):
    def __init__(self):
        super(model_constant, self).__init__()

    def forward(self, x1, x2):
        return torch.cat((x1,x2), 1)

model = model_constant()

x1 = torch.ones(3,5)
x2 = torch.ones(3,5)

y1 = model(x1 ,x2)
print("y1: ", y1)
print("y1 shape: ", y1.shape)

torch.onnx.export(model, (x1,x2), "model_concat.onnx")
