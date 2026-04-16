# Introducing MOSAIC-SEN2-CC: A Multispectral Dataset and Adaptation Framework for Remote Sensing Change Captioning


📢 **This paper is published in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), 2025.**  
🔗 [IEEE Xplore Link]([https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11130644](https://ieeexplore.ieee.org/document/11181102/))  
📄 DOI: [10.1109/JSTARS.2025.3615113](https://doi.org/10.1109/JSTARS.2025.3600613) \
\
🌐 [**MOSAIC Research Group Website**](https://avesis.yildiz.edu.tr/arastirma-grubu/mosaic)

## 🔎 Summary
The paper introduces multispectral change captioning for remote sensing, a task not previously explored beyond RGB images. It presents the **MOSAIC-SEN2-CC dataset** with Sentinel-2 image pairs and captions across multiple change categories. A new **MSICC** framework leveraging BigEarthNet features and a transformer decoder is proposed, and existing methods are adapted to multispectral data. Results show that using spectral information improves change captioning performance.

---

You can view a sample visualization and MOSAIC dataset overview in the document below:
📄 [View Overview (MOSAIC-CC-H.pdf)](MOSAIC-CC-H.pdf)
DOI: 
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=ChangeCapsInRS/MOSAIC-SEN2-CC)]()
[![GitHub stars](https://img.shields.io/github/stars/ChangeCapsInRS/MOSAIC-SEN2-CC?style=social)]()
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FJSTARS.2025.3615113-blue)](https://doi.org/10.1109/JSTARS.2025.3615113)
---

⭐ **Share us a star if this repo helps your research!**  

🔥 Our new work on **change captioning and multimodal reasoning** is continuously updated here. Stay tuned! 🔥

---

## 📘 MOSAIC-SEN2-CC Dataset
We introduce **MOSAIC-SEN2-CC**, a multispectral remote sensing change captioning dataset.

- [Download Link (Google Drive)](https://drive.google.com/file/d/1eqqVaYQfyjPKjziABe-Xmr4v3HnMlDF3/view?usp=sharing)
- [Download Link (Zenodo, DOI: coming soon)
- Train / Val / Test splits provided with 10 Sentinel 2 multispectral band images
- Includes paired images + change captions

## ⚙️ Installation and Dependencies
```bash (will be updated)
git clone https://github.com/MOSAIC-Lab/MModalCC.git
cd MModalCC
conda create -n mmodalcc_env python=3.6
conda activate mmodalcc_env
pip install -r requirements.txt
```


## 📊 Evaluation Metrics

For evaluation, we use standard captioning metrics: **BLEU, CIDEr, ROUGE-L, METEOR, and SPICE**.

Due to GitHub storage limitations, the `eval_func/meteor` and `eval_func/spice` subfolders are **not included** in this repository.  
You can download them from Google Drive:

- [METEOR & SPICE Evaluation Scripts (Google Drive)](https://drive.google.com/file/d/1GseNGhs2qhIW6G72fktrWckbTaZ98vws/view?usp=sharing)

After downloading, place them under:
```bash
./eval_func/
├── bleu/
├── cider/
├── rouge/
├── meteor/
└── spice/
```

---
## 📂 Data Preparation

### Download the Dataset
Download **SECOND-CC** dataset from Google Drive:

- [SECOND-CC Dataset (Google Drive)](https://drive.google.com/file/d/1GseNGhs2qhIW6G72fktrWckbTaZ98vws/view?usp=sharing)
- [SECOND-CC Dataset (Zenodo, will be updated)

---

## 🔎 Inference Demo (will be updated)

You can download our pretrained model checkpoint: [Google Drive](https://drive.google.com/file/d/1VYGYn9UbdCRnVJrpOWZAx-WRXHZwzW4C/view?usp=sharing)

After downloading, put it into:

./checkpoint/

Run demo:
```bash
python eval_MModalCC.py --beam_size 4 --data_folder .\createdFileBlackAUG --path .\checkpoint
```
Generated captions will be saved in the workspace as well as ground truth captions.

---

## 🏋️ Training (will be updated)

Make sure dataset preprocessing is done. Then run:

Run training:
```bash
python train_MModalCC.py
  --data_folder ./createdFileBlackAUG/ \
  --data_name SECOND_CC_5_cap_per_img_10_min_word_freq \
  --encoder_image resnet101 \
  --epochs 30 \
  --batch_size 28 \
  --encoder_lr 5e-5 \
  --decoder_lr 5e-5 \
  --fine_tune_encoder True

```
  
---

## 📑 Citation

If you find our work useful, please cite:
```bash

@ARTICLE{karaca2025robust,
  author={Busra Tuzlupinar, Enes Ozelbas, Mehmet Fatih Amasyali, Ali Can Karaca},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Introducing MOSAIC-SEN2-CC: A Multispectral
Dataset and Adaptation Framework for Remote
Sensing Change Captioning}, 
  year={2025},
  volume={18},
  number={},
  pages={25410-25426},
  doi={10.1109/JSTARS.2025.3615113}}

 
```

---

## 🙏 Reference

We thank the following repositories:

- [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

- [RSICCformer (Liu et al., TGRS 2022)](https://github.com/Chen-Yang-Liu/RSICC)


