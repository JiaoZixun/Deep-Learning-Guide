# Deep Learning Guide

帮助初学者快速上路（入门），了解如何搭建深度网络，可以根据需要进行模块化快速搭建
主要分为三个部分
1. backbone
2. dataloader
3. trick

backbone部分汇总了主流的特征提取结构  
dataloader部分给出了不同数据集如何读取的模板，以手部姿态估计方向数据集为例  
trick部分汇总近期前沿论文提出的新方法（即插即用模块）
  
对于结构创新可以从第一部分和第三部分入手

## backbone

### 1. resnet  
1）resnet  
论文：  
结构图：  
要点：  
``` python
                                        # torch.Size([2, 3, 256, 256])
x = self.conv1(x)                       # torch.Size([2, 64, 128, 128])
x = self.bn1(x)                         # torch.Size([2, 64, 128, 128])
x = self.leakyrelu(x)                   # torch.Size([2, 64, 128, 128])
x = self.maxpool(x)                     # torch.Size([2, 64, 64, 64])
x = self.layer1(x)                      # torch.Size([2, 256, 64, 64])
x = self.layer2(x)                      # torch.Size([2, 512, 32, 32])
x = self.layer3(x)                      # torch.Size([2, 1024, 16, 16])
x = self.layer4(x)                      # torch.Size([2, 2048, 8, 8])
x = x.mean(3).mean(2)                   # torch.Size([2, 2048])
x = x.view(x.size(0), -1)               # torch.Size([2, 2048])
x = self.fc(x)                          # torch.Size([2, 1000])
return x
```
使用指南：
1. 先运行check_model.py 可以看到每一层输出的大小，输入都是三通道256*256大小的tensor  
resnet50经过四层从64通道增加到2048通道
```
self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 通过这个调节
```
最后的预测可以调整fc，num_classes就是你需要预测的类别
```
self.fc = nn.Linear(512 * block.expansion, num_classes)
```
### 2. transformer  
1） segformer   
论文：  
结构图：  
要点：  
使用指南：

## dataloader


## trick