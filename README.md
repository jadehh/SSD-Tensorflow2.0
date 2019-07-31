# SSD-Tensorflow2.0
## tensorflow 2.0 ssd
vgg 分类网络
Conv2D 层
MaxPool 层
在 conv5 之后接上两个全连接层，最后softmax进行分类

```
python nets/vgg.py
```
ssd 对 vgg 做了一点变化

在fc6层和fc7层与以往的VGG不同的是，使用卷积层来代替以前的全连接层
卷积层使用的是局部信息，全连接层使用的是全局信息
但是在最后fc6 和 fc7 特征尺寸变大，

可以让卷积网络在一张更大的输入图片上滑动，得到每个区域的输出（这样就突破了输入尺寸的限制）。