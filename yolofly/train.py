from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json

# 初始化基线模型
model = YOLO('/home/zd/WorkSpace/visTrain/ultralytics/ultralytics/cfg/models/fly/d6_yolov8.yaml')#'/home/zd/WorkSpace/runs/AITOD/FULL/weights/last.pt')#
model.model = model.model.to('cuda:0')  # 强制移动到 GPU（防止构造不完全）

# 手动覆盖预训练参数为 None（关键）
model.ckpt = None
model.overrides['pretrained'] = False
# 训练（增强饱和度）
model.train(
    #data='/home/zd/Workspace/AI-TOD/aitod.yaml',
    data='/home/zd/WorkSpace/database/visDrone/visdrone.yaml', #model.train(data='/home/zd/Workspace/AI-TOD/aitod.yaml', patience=300, batch=4,epochs=300,imgsz=640)
    pretrained=False,
    epochs=300,
    imgsz=640,
    batch=8,
    amp=False,
    project='/home/zd/WorkSpace/runs/visdrone2019',
    name='d6yolo(DDC2FBIFPNHead)',
    warmup_epochs=15,
    patience=20,
    )


metrics = model.val()
print("Baseline (Enhanced with Saturation) Validation metrics:")
print(f"mAP={metrics.box.map:.4f}, mAP50={metrics.box.map50:.4f}, mAP75={metrics.box.map75:.4f}")


