# Road Cracks Segmentation using Swin Transformer
The goal of this project is to perform semantic segmentation on images of roads to accurately detect and segment cracks. This is crucial for automating road inspection and maintenance workflows.

## Model
This model utilizes a Swin Transformer (Tiny variant) as the backbone, integrated into a UPerNet (Unified Perceptual Parsing) architecture to leverage both global and local context for precise segmentation. The model weights were initialized from a pre-trained Swin Transformer Tiny checkpoint provided by OpenMMLab that was trained on the ADE20K dataset.
This model is trained using a combination of 2 losses: Cross-Entropy Loss was used in the early stages of training, followed by the [Lovasz-Softmax Loss](https://arxiv.org/abs/1705.08790) which is better at directly optimizing the IoU on finer cracks.

While a standard [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) can also be used for this task, [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) was chosen due to its efficiency in dense prediction tasks.


## Dataset
The model was trained on [Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset) with around 11.200 images which merges data from 12 available crack segmentation datasets.

The file structure of the dataset is as follows:
```none
├── data
│   ├── cracks
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx.jpg
│   │   │   │   ├── yyy.jpg
│   │   │   │   ├── zzz.jpg
│   │   │   ├── val
│   │   ├── images
│   │   │   ├── train
│   │   │   │   ├── xxx.png
│   │   │   │   ├── yyy.png
│   │   │   │   ├── zzz.png
│   │   │   ├── val

```

## Results
Here are some comparisons between the original segmentation and model's output:  


| Image         | Original segmentation             |  Segmentation after training |
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/user-attachments/assets/8e9d2f9d-37b0-4827-9333-10ff001222c7) | ![](https://github.com/user-attachments/assets/3b4b525d-65cb-4cc5-a65f-3b4efeac231f) | ![](https://github.com/user-attachments/assets/3a324db5-9ae8-4a7a-a933-4d7fb5e8a0e6) |
![](https://github.com/user-attachments/assets/22643d51-8440-461e-8bae-888955c9eaba) |![](https://github.com/user-attachments/assets/c9e0e735-32a2-4be7-aeb3-eb1e59126d3e) | ![](https://github.com/user-attachments/assets/7adb765c-ab1a-45d3-85dc-55dad7015c01) | 
![](https://github.com/user-attachments/assets/5aa882b2-517a-433f-850a-14b5ee7b8b65) | ![](https://github.com/user-attachments/assets/9539edf1-d389-4186-b9ea-99221e2c591c) | ![](https://github.com/user-attachments/assets/4e790ba2-d351-4bfe-86f8-62dcec4b6363)

Below are some results on random road crack images from Google Images:  
| Image             |   Result |
:-------------------------:|:-------------------------:
![](https://github.com/user-attachments/assets/7151f799-4e2d-44db-804f-1647c86c0ac3) | ![](https://github.com/user-attachments/assets/0a729a71-e76f-4722-8a9f-e7d3bc73f0bb)
![](https://github.com/user-attachments/assets/3ddc3fce-790b-4570-b44a-0e6c37f16e9e) | ![](https://github.com/user-attachments/assets/0ec9327a-22f1-45cf-8d0a-81fdf52ed7ff)

Models to be uploaded soon...

# Swin Transformer for Semantic Segmentaion

This repo contains the supported code and configuration files to reproduce semantic segmentaion results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). It is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0).

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train an UPerNet model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py 8 --options model.pretrained=<PRETRAIN_MODEL> 
```

**Notes:** 
- `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
- The default learning rate and training schedule is for 8 GPUs and 2 imgs/gpu.


## Citing Swin Transformer
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Other Links

> **Image Classification**: See [Swin Transformer for Image Classification](https://github.com/microsoft/Swin-Transformer).

> **Object Detection**: See [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

> **Self-Supervised Learning**: See [MoBY with Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL).

> **Video Recognition**, See [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).
