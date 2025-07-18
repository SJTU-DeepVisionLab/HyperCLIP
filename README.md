# Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space

<p align="center">
  <img src="framework.jpg" alt="Framework" width="600"/>
</p>

## 🔍 Overview

**HyperCLIP** is a lightweight and effective fine-tuning framework built upon CLIP for **open-vocabulary semantic segmentation**. Motivated by the observation that segmentation requires alignment at **pixel-level hierarchical granularity**, this work explores fine-tuning CLIP in **hyperbolic space**, which shifts the hierarchical granularity of CLIP's embedding from image-level to pixel-level, thereby equipping it with segmentation capability.

### Key Findings
- **hyperbolic radius changing via fine-tuning:** The hyperbolic radius of CLIP's text embeddings **decreases**, enabling tighter alignment with pixel-level visual semantics.
- **hyperbolic radius adjustment** HyperCLIP explicitly introduces **hyperbolic radius adjustment** for CLIP's embeddings to better align vision and language representations in hyperbolic space.
- **Parameter efficiency:** Only **~4%** of CLIP’s parameters are fine-tuned, yet HyperCLIP attains **state-of-the-art performance** across **three open-vocabulary segmentation benchmarks**.
- **Characteristic hyperbolic level:** After fine-tuning, text embeddings converge to a **stable hyperbolic radius** across different datasets, suggesting that segmentation tasks correspond to a **characteristic hierarchy level** in hyperbolic geometry.


## 📊 Visualizing Hyperbolic Radius Alignment

The figure below illustrates how CLIP embeddings evolve during HyperCLIP fine-tuning:

- Image-level semantics (large radius) → pixel-level semantics (smaller radius).

<p align="center">
  <img src="figs/hyper_radius_alignment.png" alt="HyperCLIP: hyperbolic radius contraction and hierarchical alignment" width="600"/>
</p>


# Installation and Data Preparation

Please refer to the [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) repository for guidance on:

- Environment setup (Python version, dependencies, etc.)
- Dataset preparation (e.g., COCO, ADE20K, Pascal VOC)

# Training and Evaluation

You can launch the entire training and evaluation pipeline using:

```bash
bash run_train_test.sh

```

# Acknowledgement
Thanks to the excellent works and their codebases of [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg). 

# Citation

Please consider citing our paper if the code is helpful in your research and development.

```bibtex
@inproceedings{peng2025understanding,
  title={Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation in Hyperbolic Space},
  author={Peng, Zelin and Xu, Zhengqin and Zeng, Zhilin and Wen, Changsong and Huang, Yu and Yang, Menglin and Tang, Feilong and Shen, Wei},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={4562--4572},
  year={2025}
}
```
