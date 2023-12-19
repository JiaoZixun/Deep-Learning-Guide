from segformer import segformer_b0, segformer_b1, segformer_b2, segformer_b3, segformer_b4, segformer_b5
import torch




segformer = segformer_b0(pretrained=True)

ipt = torch.rand(2, 3, 256, 256)
out = segformer(ipt)
for i, x in enumerate(out) :
    print("第 {} 层 特征shape: {}".format(i, x.shape))
