import torch

from resnet import resnet50






resnet = resnet50(pretrained=True)

ipt = torch.rand(2, 3, 256, 256)
out = resnet(ipt)
print("out.shape", out.shape)