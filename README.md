# yolo 王者目标识别

## 准备素材

### 素材来源

素材的来源主要是：

- 训练营：以摆拍为主（train 开头的素材集）
  - hero：各种英雄转方向，转角度，释放技能。
  - soldier：各种兵，如小兵，魔法师，弩车，超级兵，主宰兵（紫色），风暴龙王兵（橙色）。注意红蓝双方兵不一样。
  - crystol：水晶
  - tower：防御塔，注意四分钟前的一塔前面有个盾牌，以及红蓝双方区别。
  - grass：草丛（草会动，可以多截图几张）
  - dragon：两只手的主宰和暴君，四只手的主宰和暴君，风暴龙王。注意开龙期间是有动画的可以录一下。
  - buff：红蓝buff，注意各种视角和打人的动画。
  - monster：野怪，注意各种视角和打人的动画。红蓝双方视角问题可能野怪的方向不同
- 回放：以不同视角为主，部分情况可能会有角色原地做移动动画的情况，或者特效没出来。
- 实时录制（round 开头的素材集）：打就行了
- 屏幕截图（screenshot 开头的素材集）：作为补充

### 制备训练集

使用 FFmpge 切割素材为图片，大约三秒一张图，视情况而定。

#### FFMpeg 

参阅 [ffmpeg隔几秒取一帧](https://blog.csdn.net/racesu/article/details/109491612)。

- `q:v`: 0到51的整数值作为输入。这个范围中的值越小，视频的质量就越好，但文件大小也越大。
- `fps=fps=1/3`：每 3 秒一张图
- `round1-1/d.jpg`: 输出位置和格式

```bash
ffmpeg.exe -i round1-1.mp4 -f image2 -q:v 2 -vf fps=fps=1/3 round1-1/image-round1-1-%3d.jpg
```

#### 标记

##### 使用 X-AnyLabeling

参阅 [文档](https://github.com/CVHub520/X-AnyLabeling/blob/main/README_zh-CN.md#%E6%96%87%E6%A1%A3)。

###### 安装 X-AnyLabeling

参阅 [1.1 从源码运行](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/get_started.md#11-%E4%BB%8E%E6%BA%90%E7%A0%81%E8%BF%90%E8%A1%8C)。

###### 使用 X-AnyLabeling

参阅 [用户手册](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/user_guide.md)。

###### 使用预训练模型标记图片

参阅 [加载已适配的用户自定义模型](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/custom_model.md#%E5%8A%A0%E8%BD%BD%E5%B7%B2%E9%80%82%E9%85%8D%E7%9A%84%E7%94%A8%E6%88%B7%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B)。

使用 [yolo11-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml) 作为模型参数训练用于标记的预训练模型。

使用 [x-label.yaml](./x-label/x-label.yaml)

```yaml
type: yolo11_seg
name: yolo11s-seg-wangzhe
display_name: YOLO11s-Seg 王者荣耀实例分割模型
model_path: best.onnx
nms_threshold: 0.45
confidence_threshold: 0.25
classes:
  - hero
  - soldier
  - crystol
  - tower
  - grass
  - dragon
  - buff
  - monster
```

```
yolo segment export format=onnx model=best.pt
```

###### 导出数据集

使用导出-导出 YOLO 分割标签导出**多边形**标记数据。

> 使用导出-导出 YOLO 水平框标签导出**矩形**标记数据。本实验不使用此选项。

使用的标签文件见 [classes.txt](data/classes.txt)。

## 训练模型

### 准备 YOLO 环境

参阅 [Install Ultralytics](https://docs.ultralytics.com/quickstart/#install-ultralytics)

### 模型参数

缩放常数选择 `n`（默认值）。

已验证的模型参数如下。

记得修改类型数目 `nc: 8 # number of classes`。

- [yolov8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml)
- [yolo11.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml)
- [yolov5.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)：训练成功，转换失败。

以下使用 [yolov11.yaml](train/yolov11.yaml)。

### 数据集

数据集目录如下所示。参见 [dataset](data/dataset)。

```
.
├── images
│   └── screenshot-1
│       ├── 1.png # 图片应该放在此处，主文件名应该和标签对应。jpg和png图片测试可混用。
├── labels
│   ├── screenshot-1 # 两个子目录的文件夹应该同名
│   │   ├── 1.txt # 导出的标签文件应该放在此处
│   └── screenshot-1.cache # 由 yolo 自动生成
└── dataset.yaml
```

数据集描述文件见 [dataset.yaml](data/dataset/dateset.yaml)

### 训练代码

```python
from ultralytics import YOLO

def main():
    model = YOLO('yolov11.yaml')  # 从YAML建立模型层，如需从原有权重中加载，在最后添加 .load('best.pt')
    results = model.train(data="../data/dateset.yaml", epochs=200, imgsz=640, save=True, save_period=1) # data 数据集描述文件路径，epochs 迭代次数（建议200次以上），save=True, save_period 保存中间的权重以及保存周期

if __name__ == '__main__':
    main()
```

```
python train.py
```

训练完成后输出如下内容（节选）。最优权重保存在 `runs/detect/train/weights/best.pt`。

```
Validating runs\detect\train\weights\best.pt...
WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.
Ultralytics 8.3.31 🚀 Python-3.11.10 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192MiB)
YOLOv11 summary (fused): 238 layers, 2,583,712 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 5/5 [00:01<0
                   all        145        881      0.932      0.917      0.957      0.705
                  hero         51         74      0.872      0.825      0.917      0.551
               soldier         70        244      0.891      0.861      0.915      0.575
               crystol         21         21      0.987          1      0.995      0.865
                 tower         82        101      0.959      0.931      0.961      0.744
                 grass        114        379      0.953       0.96      0.989      0.768
                dragon         16         16      0.967          1      0.995      0.856
                  buff         11         11      0.898          1      0.988      0.761
               monster         22         35       0.93       0.76      0.895      0.517
Speed: 0.1ms preprocess, 0.9ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to runs\detect\train
```

训练成果目录如下所示。

```
.
├── F1_curve.png
├── PR_curve.png
├── P_curve.png
├── R_curve.png
├── args.yaml
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── labels.jpg
├── labels_correlogram.jpg
├── results.csv
├── results.png
├── train_batch0.jpg
├── train_batch1.jpg
├── train_batch1900.jpg
├── train_batch1901.jpg
├── train_batch1902.jpg
├── train_batch2.jpg
├── val_batch0_labels.jpg
├── val_batch0_pred.jpg
├── val_batch1_labels.jpg
├── val_batch1_pred.jpg
├── val_batch2_labels.jpg
├── val_batch2_pred.jpg
└── weights
    ├── best.pt
    └── last.pt
```


### 导出 onxx 以供转换

参见 [模型编译](https://milkv.io/zh/docs/duo/application-development/tdl-sdk/tdl-sdk-yolov8#%E6%A8%A1%E5%9E%8B%E7%BC%96%E8%AF%91)。

[yolov8_export.py](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x/blob/main/sample/yolo_export/yolov8_export.py) 可从官方仓库下载。

```
python yolov8_export.py --weights best.pt --img-size 640 640
```

## 模型转换

参阅 [TPU-MLIR 转换模型](https://milkv.io/zh/docs/duo/application-development/tdl-sdk/tdl-sdk-yolov8#tpu-mlir-%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9E%8B)

请参考 TPU-MLIR 文档 配置好 TPU-MLIR 工作环境，参数解析请参考 TPU-MLIR 文档。

配置好工作环境后,在与本项目同级目录下创建一个model_yolov8n目录,将模型和图片文件放入其中。

模型转换命令如下：

```
model_transform.py \
--model_name wangzhe \
--model_def ../best.onnx \
--input_shapes [[1,3,640,640]] \
--mean 0.0,0.0,0.0 \
--scale 0.0039216,0.0039216,0.0039216 \
--keep_aspect_ratio \
--pixel_format rgb \
--mlir wangzhe.mlir
```

```
run_calibration.py wangzhe.mlir \
--dataset ../wangzhe/ \
--input_num 100 \
-o wangzhe_cali_table
```

用校准表生成 int8 对称 cvimodel:

```
model_deploy.py \
--mlir wangzhe.mlir \
--quant_input --quant_output \
--quantize INT8 \
--calibration_table wangzhe_cali_table \
--processor cv181x \
--model wangzhe-int8-sym.cvimodel
```

## 运行模型

运行模型的代码见 [v8.cpp](duocode/v11.cpp)。替换 [sample_yolov8.cpp](https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x/blob/main/sample/cvi_yolo/sample_yolov8.cpp) 按照 [简介](https://milkv.io/zh/docs/duo/application-development/tdl-sdk/tdl-sdk-introduction) 编译即可。

将编译好的 `sample_yolov8` 上传到开发板，运行 `./sample_yolov8 ./wangzhe-int8-sym.cvimodel ./test1.png ./test1-res.jpg` 即可。结果见 [result](result)。


```
[root@milkv-duo]~/test# ./sample_yolov8 ./wangzhe-int8-sym.cvimodel ./test1.png ./test1-res.jpg
enter CVI_TDL_Get_YOLO_Preparam...
asign val 0 
asign val 1 
asign val 2 
setup yolov8 param 
enter CVI_TDL_Get_YOLO_Preparam...
setup yolov8 algorithm param 
yolov8 algorithm parameters setup success!
---------------------openmodel-----------------------
version: 1.4.0
wangzhe Build at 2024-11-17 19:24:30 For platform cv181x
Max SharedMem size:2508800
---------------------to do detection-----------------------
image read,width:3168
image read,hidth:1440
objnum:7
Detect crystol(2): 1781.433594 26.692432 2346.078857 540.206177 0.873866
Detect tower(3): 973.372681 505.802856 1279.624634 1045.728394 0.804268
Detect soldier(1): 1452.727295 623.431396 1640.592407 816.327576 0.760650
Detect soldier(1): 2545.095947 1229.370605 2760.516846 1439.000000 0.736484
Detect soldier(1): 1304.465210 804.322693 1462.080322 980.739136 0.710806
Detect tower(3): 665.742432 0.000000 886.644775 203.547089 0.655286
Detect soldier(1): 2446.042236 860.136475 2645.260010 1087.134399 0.655286
```

![test1-res.jpg](result/test1-res.jpg)
