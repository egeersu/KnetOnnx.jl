import torch
from torch.autograd import Variable
import torch.onnx
import torch.nn as nn
import numpy
import torch.nn.functional as F

class model_constant(nn.Module):
    def __init__(self):
        super(model_constant, self).__init__()

    def forward(self, x1):
        return x1[1,2]

model = model_constant()

x1 = torch.tensor([[1,2,3], [4,5,6]])
y1 = model(x1)

print("y1: ", y1)
print("y1 shape: ", y1.shape)

torch.onnx.export(model, x1, "model_gather.onnx")
