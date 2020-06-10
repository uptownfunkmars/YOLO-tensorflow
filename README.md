# YOLO-tensorflow

tensorflow implementation of 'YOLO : Real-Time Object Detection' 


yolo v1 的tensorflow实现

这是一个tensorflow实现的YOLO Version 1.0。

### 数据集

使用Pascal Voc 2012公开数据集。

### 各代码功能简介

+ pascal_preprocess.py
> 负责将Pascal Voc的xml标记文件转化为字典（dict）.
+ img_transform.py
> 负责对图片进行仿射变换，和颜色抖动。
+ data.py
> 定义了YOLO的网络结构。
+ loss_fn.py
> 定义了训练过程的loss function.
+ yolo_train.py
> 训练过程。
> python train.py开始训练。


代码参考了以下三个仓库：

* https://github.com/thtrieu/darkflow
* https://github.com/gliese581gg/YOLO_tensorflow
* https://github.com/xiaohu2015/DeepLearning_tutorials/tree/master/ObjectDetections/yolo


