---
title: CVPR2019 Decoders Matter for Semantic Segmentation
categories:
- 计算机视觉
---

今天为大家推荐一篇CVPR2019 关于语义分割的文章 [Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation](https://arxiv.org/abs/1903.02120), 该文章提出了一种不同于双线性插值的上采样方法，能够更好的建立每个像素之间预测的相关性。得益于这个强大的上采样方法，模型能够减少对特征图分辨率的依赖，能极大的减少运算量。该工作在PASCAL VOC数据集上达到了88.1%的mIOU，超过了DeeplabV3+的同时只有其30%的计算量。

### 1. Introduction

​	在之前的语义分割方法中，双线性插值通常作为其最后一步来还原特征图的分辨率，由于非线性差值不能建立起每个像素的预测之间的关系，因此为了得到精细的结果，对特征图的分辨率要求较高，同时带来了巨额的计算量。

​	为了解决这个问题，本工作提出了Data-dependent Up-sampling (DUpsample)，能够减少上采样操作对特征图分辨率的依赖，大量的减少计算量。同时得益于DUpsample， Encoder中的low-level feature能够以更小的运算量与Decoder中的high-level feature进行融合，模型结构如下所示：

![](/images/DUNet.png)

我们可以看到，该网络将传统的非线性插值替换成DUpsample，同时在feature fuse方面，不同于之前方法将Decoder中的特征上采样与Encoder特征融合，本工作将Encoder中的特征下采样与Decoder融合，大大减少了计算量 ，这都得益于DUpsample。

### 2 Our Approach

之前的语义分割方法使用下列公式来得到最终的损失：
$$
L(F, Y)=Loss(softmax(bilinear(F)), Y).
$$

其中Loss通常为交叉熵损失，F为特征图，Y为ground truth，由于双线性插值过于简单，对特征图F的分辨率较高，因此引入了大量的计算。一个重要的发现是语义分割输入图像的label Y并不是i.i.d的，所以$Y$可以被压缩成$Y'$，我们令$Y\in R^{H\times W \times C}$, 并将$Y$划分成$\frac{H}{r} \times \frac{W}{r}$的子窗口，每个子窗口的大小为$r\times r$，接着我们将每个子窗口$S \in \{0,1\}^{r\times r \times C} $ 拉伸成向量$v \in \{0, 1\}^N$，其中$N = r \times r \times C$，随即我们将向量$v$压缩成低维向量$x$，我们使用线性投影来完成，最后，我们有：
$$
x = Pv; v'=Wx
$$
其中$P \in R^{C'\times N}$，用来将$v$压缩成$x$，$W \in R^{N\times C'}$为reconstruction matrix, $v'$为重建后的$v$，我们可以用压缩后的向量$x$组合成$Y'$.

矩阵$P$和矩阵$W$可以通过最小化下列式子得到：
$$
P^*,W^*=argmin_{P,W}\sum_{v}||v-WPv||^2
$$
我们可以使用梯度下降，或者在正交约束的条件下使用PCA求解。

使用压缩后的$Y'$为目标，我们可以使用下列损失函数来预训练网络：
$$
L(F, Y) = ||F-Y'||^2
$$
另一种直接的方法是在$Y$空间计算loss，也就是并非将$Y$压缩到$Y'$, 我们可以将$F$使用$W$（上面预训练得到的）上采样然后计算损失，公式如下：
$$
L(F, Y) = Loss(softmax(DUpsample(F)), Y)
$$


其中以两倍为例，DUpsample的操作如下图所示：

![](/images/DUpsample.png)

我们可以用1X1卷积来完成上述的权重与特征相乘的过程。但是当我们将这个模块嵌入到网络时会遇到优化问题。因此我们使用softmax with temperature 函数来解决这个问题：
$$
softmax(z_i)=\frac{exp(z_i/T)}{\sum_{j}exp(z_j/T)}.
$$


我们发现T可以使用梯度下降学习得到，这样减少了调试的麻烦。



有大量的工作说明，与low-level features结合可以显著的提升分割的精度，其做法如下：
$$
F = f(concat(upsample(F_i), F_{last})),
$$


f是在上采样之后的卷积操作，其计算量依赖于特征图的空间大小，这样做会显著增加计算量。得益于DUpsample，我们可以使用下列操作来减少计算量：
$$
F = f(concat(downsample(F_i),F_{last})),
$$


这样做不仅保证了在低分辨率下的有效性，而且减少了计算量，同时允许任意level feature的融合。

只有使用了DUpsample，上述操作才变得可行，否则语义分割的精度会被双线性插值限制。

### 3 Experiments

本次实验使用以下两种数据集：PASCAL VOC 2012 和 PASCAL Context benchmark. 我们使用ResNet-50或Xception-50作为我们的backbone，具体训练细节详见论文。

首先我们设计实验说明双线性插值的上限远远低于DUpsample。首先我们搭建一个简易网络实现auto-encoder，其中上采样方式分别使用双线性插值与DUpsample, 输入分别为ground_truth，得到下表中的mIOU*，这个指标代表上采样方法的上限。同时我们使用ResNet50作为主干网络，输入为raw image去实现语义分割，得到下表中的mIOU：

![](/images/upsample.png)

通过上表我们可以发现:1) 在相同条件下，DUpsampling效果优于bilinear。2）DUpsampling在output_stride=32的情况下效果与bilinear在output_stride=16的情况下结果相当。

接下来我们设计实验说明融合不同的low-level特征对结果的影响，如下表所示：

![](/images/low-level.png)

值得说明的是，并不是所有与low-level feature的融合都会提升结果，例如conv1_3，因为其结果不够鲁棒。因此和什么low-level feature相结合对语义分割的结果有很大的影响。



接下来我们设计实验与双线性插值进行比较：

![](/images/compare.png)

可以看到我们的方法优于传统的双线性插值上采样方法。同时我们验证了不同的softmax对结果的影响，在没有使用softmax with tenperature的情况下只有69.81的mIOU（这里没设置消融实验有些疑惑，感觉不同的softmax对实验结果影响挺大的）。

![](/images/softmax.png)

最后将我们的方法与最新的模型进行比较，结果如下（分别为PASCAL VOC与PASCAL context）：

![](/images/Screenshot from 2019-03-11 20-12-31.png)

![](/images/Screenshot from 2019-03-11 20-13-56.png)

我们的方法在只用deeplabv3+ 30%的计算量的情况下，超越了当前所有的方法。

总的来说，我觉得这个论文提出的想法很有趣，是一篇很有insight的论文。

由于论文现在还没有开源，笔者尝试实现了一下DUpsample的操作和网络：https://github.com/LinZhuoChen/DUpsampling，目前还在建设中，欢迎大家关注和star :)

