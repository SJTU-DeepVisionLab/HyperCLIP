# Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space

<p align="center">
  <img src="framework.jpg" alt="Framework" width="600"/>
</p>

This repository provides the official implementation of our **CVPR 2025** paper:

**Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space**  
Zelin Peng, Zhengqin Xu, Zhilin Zeng, Changsong Wen, Yu Huang, Menglin Yang, Feilong Tang, Wei Shen  
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025*  
Pages 4562–4572

---

We explore a principled fine-tuning strategy for CLIP in **hyperbolic space** to support open-vocabulary semantic segmentation. Our approach leverages:

- **Radius-aware transformation** of CLIP embeddings to encode hierarchical structures.
- A **dual cross-modality communication module** to align vision and language features at fine granularity levels.
- An efficient fine-tuning pipeline without the need for box-level annotations.

---

## Installation and Data Preparation

Please refer to the [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) repository for guidance on:

- Environment setup (Python version, dependencies, etc.)
- Dataset preparation (e.g., COCO, ADE20K, Pascal VOC)

---

## Training and Evaluation

You can launch the entire training and evaluation pipeline using:

```bash
bash run_train_test.sh
