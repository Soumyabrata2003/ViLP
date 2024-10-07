<div align="center">

<h2>【ICVGIP'2023】ViLP: Knowledge Exploration using Vision, Language and Pose Embeddings for Video Action Recognition </h2>

[![Conference](https://img.shields.io/badge/ICVGIP-2023-brightgreen.svg
)](https://dl.acm.org/doi/abs/10.1145/3627631.3627637) 
[![arXiv](https://img.shields.io/badge/Arxiv-2311.15732-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2308.03908)


This is the official implementation of our **ViLP**, which leverages cross-modal bridge to enhance video recognition by exploring tri-directional knowledge.
</div>

## Overview
🚴**BIKE** explores bidirectional cross-modal knowledge from the pre-trained vision-language model (e.g., CLIP) to introduce auxiliary attributes and category-dependent temporal saliency for improved video recognition.

![BIKE](docs/bike.png)


## Content
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
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
