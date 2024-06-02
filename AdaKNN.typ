#import "template.typ": *

#show: apply-template.with(
  title:[*AdaBoost-KNN* *Project Report*],
  right_header:"数据建模与分析课程作业",
  left_header:"Statistical Learning HomeWork",
  author: "章翰宇",
  ID:"3220104133",
  )

\

#t 本次作业的notebook代码以及报告等相关内容均已经上传至我的仓库

= Introduction

== Method



== Reference

#t 我参考了两篇文献以及CSDN论坛

#t 一篇是#text(blue)[改进的AdaBoost集成学习方法研究[D].暨南大学,2021.DOI:10.27167/d.cnki.gjinu.2020.001350.]

#t 一篇是#text(blue)[Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.]

#t CSDN内容：#text(blue)[https://blog.csdn.net/weixin_43298886/article/details/110927084]

== experiment environment

#t 我使用Anaconda中的环境，使用jupyternotebook进行编辑，使用了*torch*库，其版本为2.2.1

#figure(
  image("./images/torch ver.png", width:50%),
  caption: "检查版本"
)

= Codes and Results

代码如下：

#codex(read("./code/AdaKNN.py"),lang :"python")

实验效果：我采用了如下图所示的训练样本点(x_data与y_data)，其中不同颜色代表不同的类别

#figure(
  image("./images/dots.png", width:50%),
  caption: "代码中的data点"
)

输出
