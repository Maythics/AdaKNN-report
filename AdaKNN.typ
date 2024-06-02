#import "template.typ": *

#show: apply-template.with(
  title:[*AdaBoost-KNN* *Project Report*],
  right_header:"数据建模与分析课程作业",
  left_header:"Statistical Learning HomeWork",
  author: "章翰宇",
  ID:"3220104133",
  )

\

=== Repository

#t 本次作业的notebook代码以及报告等相关内容均已经上传至我的仓库，如有疑问请访问#text(purple)[https://github.com/Maythics/AdaKNN-report]

= Introduction

== Method

#t 概述方法：Adaboost本来用于二分裂问题，但是KNN则是针对多分类（KNN二分类也没什么意思），因此需要知道如何让Adaboost支持多分类，下面是（详见reference）简单陈述：

对于多维，将样本的标签也设置为多维的，比如：$y = cases(1 quad y in k "family",-1/(k-1)  quad "else" )$

修改传统的Adaboost中的指数损失函数为：

$ L(y,f(x)) =sum_i exp(-1/k vectorarrow(y_i) dot vectorarrow(f_m(x_i))) $

把当前的分类器拆分成之前的加上最近一步的：
$ f_m(x)=f_(m-1)(x)+alpha_m g_m(x) $

#t 然后代入，看看损失函数在正确与否的情况下分别是多少，得到一个能用事情函数表达出来的式子，再对$alpha$求偏导，思路和李航书中Adaboost相同……

#t 最终，得到的算法如下：

+ 初始化权重$w_i$

+ 在权重$w_i$下训练分类器

+ 计算错误率$"error"= sum_i w_i I(g_m eq.not vectorarrow(y_i))/(sum_i w_i)$

+ 计算子分类器权重$alpha_m = ln((1-"error")/"error")+ln(N-1)$，$N$是类别数

+ 更新样本点权重$w_i=w_i exp(alpha_m I(g_m eq.not vectorarrow(y_i)))$

+ 归一化权重，然后循环往复

#t 下面还要解决一个问题，就是如何将KNN用于优化，因为KNN基于的是固定的k个最近邻，好像没什么参数可以调的，于是这里使用的是weigthed KNN，这样就有权重了

#t 具体含义

#t 例子：比如 $k=3$ ，则先找三个最邻近的点，它们权重分别为A(0.3)，A(0.6)，B(0.7)，0.3+0.6=0.9>0.7 因此，以最后选择归到A类中

#t 可以看出，其实这就是一个投票池系统，每来一个新的点，就先找k个最近，然后再发动这k个代表带权投票，投票值最大的类胜出；再来一个点，再发动一次找最近，最近的几个再带权投票……

#t 在本Project中，训练时采用的是随机滚动洗牌的方法，输入某些带有标签的数据后，这样训练：

+ 第一轮，先人为地从数据中分出一个测试集，剩余的当模型点集

+ 根据模型点集中的点，尝试用weighted KNN分类测试集

+ 根据在测试集上的表现，更新测试集的权重（按照上面的Adaboost）

+ 归一化权重，我这里是将训练集+模型点集一起归一化

+ 将模型集与训练集合并，然后洗牌，重新分出测试集和模型集，重复

#t 在某一轮分错的点，其权重会增大，之后的几轮中，它被洗牌到模型集之后，就会在投票时获得更大的票数。也就是，“之前的分错会导致之后投票时话语权更大”，这样就实现了weighted KNN的变权重，之后会更加照顾分错点的意见

== Reference

#t 我参考了两篇文献以及CSDN论坛

#t 一篇是#text(blue)[改进的AdaBoost集成学习方法研究[D].暨南大学,2021.DOI:10.27167/d.cnki.gjinu.2020.001350.]

#t 一篇是#text(blue)[Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.]

#t CSDN内容：#text(blue)[https://blog.csdn.net/weixin_43298886/article/details/110927084]

= Codes and Results

== experiment environment

#t 我使用Anaconda中的环境，使用jupyternotebook进行编辑，使用了*torch*库，其版本为2.2.1

#figure(
  image("./images/torch ver.png", width:50%),
  caption: "检查版本"
)

== Codes

代码如下：

#codex(read("./code/AdaKNN.py"),lang :"python")

实验效果：我采用了如下图所示的训练样本点(x_data与y_data)，其中不同颜色代表不同的类别

#figure(
  image("./images/dots.png", width:50%),
  caption: "代码中的data点"
)

输出
