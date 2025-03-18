# Road Cracks Segmentation using Swin Transformer
The goal of this project is to perform semantic segmentation on images of roads to accurately detect and segment cracks. This is crucial for automating road inspection and maintenance workflows.

This model utilizes a Swin Transformer (Tiny variant) as the backbone, integrated into a UPerNet (Unified Perceptual Parsing) architecture to leverage both global and local context for precise segmentation. The model weights were initialized from pre-trained Swin Transformer Tiny checkpoint provided by OpenMMLab that was trained on the ADE20K dataset.
This model is trained using a combination of 2 losses: Cross-Entropy Loss was used in early stages of training, followed by the [Lovasz-Softmax Loss](https://arxiv.org/abs/1705.08790) which is better at directly optimizing the IoU on finer cracks.

While a standart [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) can also be used for this task, [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) was chosen due to its efficiency in dense prediction tasks.


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

## Main results
| Backbone | Method | Crop Size | Lr Schd | mIoU | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | UPerNet | 384x384 | 15K | 51.55 | [config](configs/swin/config_upernet_swin_large_patch4_window12_384x384_15k_cracks.py) | model(to be uploaded soon) |
| Swin-T | UPerNet | 384x384 | 40K | 54.89 | [config](configs/swin/config_upernet_swin_large_patch4_window12_384x384_40k_cracks_lovasz.py) | model (to be uploaded soon) |

Here are some comparisons between the original segmentation and model's output:  


| Original segmentation             |  Segmentation after training |
:-------------------------:|:-------------------------:
![](link_to_ground_truth)  |  ![](link_to_result)
![](link_to_ground_truth)  |  ![](link_to_result)
![](link_to_ground_truth)  |  ![](link_to_result)

Below are some results on random road crack images from Google Images:  
| Image             |   Result |
:-------------------------:|:-------------------------:
 ![](https://github.com/user-attachments/assets/c7199e55-0dd8-4e13-b73f-db29f76418fa) | ![](https://github.com/user-attachments/assets/24a582d1-b7d1-4a85-9bc6-7326b77a4c03)
![](link_to_result_3)  |  ![](link_to_result_4)


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
