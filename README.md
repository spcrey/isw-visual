# 南海内波可视化

### 用法示例：温度（Temp）

```cmd
python visual_image_sequence.py --data_folder=data\ISWFM-NSCS-6day --fea_name=temp --nop=5 --image_folder=.\output\image_sequence_temp --t0=0 --t1=23 --x0=300 --y0=60 --x1=1500 --y1=750 --zx_aspect=20
```

- 数据文件夹位置：.\data\ISWFM-NSCS-1day
- 图片保存文件夹位置：.\sequence_images_temp
- 线程数：5，取决于设备
- T 范围：（0，23）
- Z 范围：（0，90）
- XY 平面切线位移向量：(300, 1500) -> (60, 750)
- zx_aspect 为生成图片高宽比，值越大，图片的高与宽的比值越大，适用于 temp

### 用法示例：海平面高度（Eta）

```cmd
python visual_image_sequence.py --data_folder=data\ISWFM-NSCS-6day --fea_name=eta --nop=5 --image_folder=output\image_sequence_eta --t0=0 --t1=23 --x0=300 --y0=60 --x1=1500 --y1=750 --yx_aspect=1.76
```

- T 范围：（0，23）
- Z 范围：无需指定该参数，值就是是高度信息
- XY 不再作为向量，而是同 T 一样作为范围，因此需要保证 x1>x0，y1>y0
- yx_aspect 作用同 zx_aspect，但只能适用于 eta

### 用法示例：查找数据（Amp）

```cmd
python get_coord_data.py --data_folder=data\ISWFM-NSCS-6day --x0=10 --y0=10 --x1=1000 --y1=800 --output_folder=output
```
