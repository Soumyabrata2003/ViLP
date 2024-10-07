<div align="center">

<h2>【ICVGIP'2023】ViLP: Knowledge Exploration using Vision, Language and Pose Embeddings for Video Action Recognition </h2>

[![Conference](https://img.shields.io/badge/ICVGIP-2023-brightgreen.svg
)](https://dl.acm.org/doi/abs/10.1145/3627631.3627637) 
[![arXiv](https://img.shields.io/badge/Arxiv-2311.15732-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2308.03908)


This is the official implementation of our **ViLP**, which leverages cross-modal bridge to enhance video recognition by exploring tri-directional knowledge.
</div>

## Overview
🚴**ViLP** explores cross-modal knowledge from the pre-trained vision-language model (e.g., CLIP) to introduce the combination of pose, visual information, and text attributes which has not been explored yet.

![ViLP](Model_new.png)


## Content
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)  
- [Testing](#testing)  
- [BibTeX & Citation](#bibtex)
- [Acknowledgment](#acknowledgment)

<a name="prerequisites"></a>
## Prerequisites

<details><summary>The code is built with following libraries. </summary><p>

- [PyTorch](https://pytorch.org/) >= 1.8
- RandAugment
- pprint
- tqdm
- dotmap
- yaml
- csv
- Optional: decord (for on-the-fly video training)
- Optional: torchnet (for mAP evaluation on ActivityNet)
</p></details>


<a name="data-preparation"></a>
## Data Preparation



### Video Loader

**(Recommend)** To train all of our models, we extract videos into frames for fast reading. Please refer to [MVFNet](https://github.com/whwu95/MVFNet/blob/main/data_process/DATASETS.md) repo for the detailed guide of dataset processing.  
The annotation file is a text file with multiple lines, and each line indicates the directory to frames of a video, total frames of the video and the label of a video, which are split with a whitespace. 
<details open><summary>Example of annotation</summary>

```sh
abseiling/-7kbO0v4hag_000107_000117 300 0
abseiling/-bwYZwnwb8E_000013_000023 300 0
```
</details>

(Optional) We can also decode the videos in an online fashion using [decord](https://github.com/dmlc/decord). This manner should work but are not tested. All of the models offered have been trained using offline frames. 
<details><summary>Example of annotation</summary>

```sh
  abseiling/-7kbO0v4hag_000107_000117.mp4 0
  abseiling/-bwYZwnwb8E_000013_000023.mp4 0
```
</details>


### Annotation
Annotation information consists of two parts: video label, and category description.

- Video Label: As mentioned above, this part is same as the traditional video recognition. Please refer to [lists/k400/kinetics_rgb_train_se320.txt](lists/k400/kinetics_rgb_train_se320.txt) for the format.
- Category Description: We also need a textual description for each video category.  Please refer to [lists/k400/kinetics_400_labels.csv](lists/k400/kinetics_400_labels.csv) for the format.

<a name="training"></a>
## 🚀 Training

1. **Single Machine**: To train our model on Kinetics-400 with 8 GPUs in *Single Machine*, you can run:
```sh
# We train the 8 Frames ViT-B/32 video model (i.e., video branch).
sh scripts/run_train.sh  configs/k400/k400_train_rgb_vitb-32-f8.yaml
```

<a name="testing"></a>
## ⚡ Testing
We support single-view validation (default) and multi-view (4x3 views) validation.

```sh
# The testing command for obtaining top-1/top-5 accuracy.
sh scripts/run_test.sh Your-Config.yaml Your-Trained-Model.pt

# The command for zero-shot evaluation is similar.
sh scripts/run_test_zeroshot.sh Your-Config.yaml Your-Trained-Model.pt
```

We provide more examples of testing commands below.

<details><summary>Zero-shot Evaluation<p></summary>


We use the Kinetics-400 pre-trained model (e.g., [ViT-L/14 with 8 frames](configs/k400/k400_train_rgb_vitl-14-f8.yaml)) to perform cross-dataset zero-shot evaluation, i.e., UCF101, HMDB51.


- Full-classes Evaluation: Perform evaluation on the entire dataset.

```sh

# On UCF101: reporting the half-classes and full-classes results
# Half-classes: 86.63 ± 3.4, Full-classes: 80.83
sh scripts/run_test_zeroshot.sh  configs/ucf101/ucf_zero_shot.yaml exps/k400/ViT-L/14/8f/k400-vit-l-14-f8.pt

# On HMDB51: reporting the half-classes and full-classes results
# Half-classes: 61.37 ± 3.68, Full-classes: 52.75
sh scripts/run_test_zeroshot.sh  configs/hmdb51/hmdb_zero_shot.yaml exps/k400/ViT-L/14/8f/k400-vit-l-14-f8.pt

```
</details>

<a name="bibtex"></a>
## 📌 BibTeX & Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry😁.


```bibtex
@inproceedings{chaudhuri2023vilp,
  title={Vilp: Knowledge exploration using vision, language, and pose embeddings for video action recognition},
  author={Chaudhuri, Soumyabrata and Bhattacharya, Saumik},
  booktitle={Proceedings of the Fourteenth Indian Conference on Computer Vision, Graphics and Image Processing},
  pages={1--7},
  year={2023}
}
```



<a name="acknowledgment"></a>
## 🎗️ Acknowledgement

This repository is built based on [BIKE](https://github.com/whwu95/BIKE) and [Text4Vis](https://github.com/whwu95/Text4Vis). Sincere thanks to their wonderful works.


## 👫 Contact
For any question, please file an issue.

