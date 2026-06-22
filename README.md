# FLY-YOLO / Direction-Aware Cross-Scale Fusion for Parameter-Efficient UAV Small-Object Detection

[![Paper](https://img.shields.io/badge/Paper-The%20Visual%20Computer-blue)](https://link.springer.com/journal/371)
![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)

Official model configuration for **FLY-YOLO**, a parameter-efficient small-object detector for UAV imagery, currently under review at **The Visual Computer**.

## 📄 Paper

> **Direction-Aware Cross-Scale Fusion for Parameter-Efficient UAV Small-Object Detection**  
> Dong Zhou, Jun Liu  
> *Submitted to The Visual Computer, 2026*

## 📁 Repository Structure

```
├── /yolofly/fly-yolo.yaml          # Model configuration (YOLO format)
└── README.md              # This file
```

## ⚙️ Model Configuration

[`/yolofly/fly-yolo.yaml`](/yolofly/fly-yolo.yaml) contains the complete architecture specification for FLY-YOLO (nano scale), including:

- **Backbone**: C3k2-TCF blocks with DARF (Direction-Aware Receptive Field)
- **Neck**: CIF (Cross-scale Interactive Fusion) modules with bidirectional top-down / bottom-up pathway
- **Head**: DD-Head (parameter-efficient dual-branch detection head) on P2/P3/P4 outputs

### Key Design Principles

| Component | Description |
|-----------|-------------|
| **P2/P3/P4 Pyramid** | High-resolution P2 detection layer added; P5 head removed (P5 features retained in neck) |
| **DARF** | Direction encoding → GAP+MLP+Softmax → adaptive 3×/5×/7-weighting of multi-scale DWConv |
| **CIF** | Weighted fusion + cross-gated local/strip branches + spatial attention gate |
| **DD-Head** | AFBlock (texture+direction) + LFBlock (3×3+5×5 DWConv) with decoupled cls/reg |

## 📊 Results

| Dataset | mAP50 (%) | mAP (%) | Params (M) | GFLOPs |
|---------|-----------|---------|------------|--------|
| VisDrone2019 | 45.07 | 27.41 | 2.46 | 21.4 |
| AI-TOD | 48.78 | 21.76 | 2.46 | 21.4 |

## 🚧 Full Code Release

The complete source code (model implementation, training scripts, evaluation pipeline, and pretrained weights) will be made publicly available **upon paper acceptance**.

## 🧪 How to Use (Preview)

Once the full code is released, you will be able to train FLY-YOLO using the Ultralytics YOLO framework:

```bash
# Training
yolo train model=fly-yolo.yaml data=visdrone.yaml epochs=300 imgsz=640 batch=8

# Evaluation
yolo val model=runs/train/exp/weights/best.pt data=visdrone.yaml imgsz=640
```

## 📧 Contact

Dong Zhou — 22307010001@stu.wit.edu.cn

## 📚 Citation

```bibtex
@article{zhou2026fly,
  title={Direction-Aware Cross-Scale Fusion for Parameter-Efficient UAV Small-Object Detection},
  author={Zhou, Dong and Liu, Jun},
  journal={The Visual Computer},
  year={2026},
  note={Under review}
}
```

## 📝 License

This project is released under the MIT License.
