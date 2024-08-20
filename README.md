### SpeHeaTal

## A Cluster-Enhanced Segmentation Method for Sperm Morphology Analysis

## Quick Tour:

This is a implementation for our AAAI-2025 paper "SpeHeaTal: A Cluster-Enhanced Segmentation Method for Sperm Morphology Analysis". To run the code, please make sure you have prepared your data following the same structure as follows (you can also refer to the examplar data in this repository).

## Project Structure

Here is an overview of the project's directory structure:

```
SpeHeaTal/
    - preprocessing_step.py            # Preprocessing and initial segmentation file
    - sperm_segmentation_main.py         # main file of sperm segmentation
    - sam_vit_h_4b8939.pth       # SAM model-vit-h
    - sam_vit_l_0b3195.pth       # SAM model-vit-l
    - sam_vit_b_01ec64.pth       # SAM model-vit-b
    - README.md             # This file
    - requirements.txt      # The required Python packages

    - segment-anything-main/       # SAM model folder
    - original_image/      # Sperm original image folder
    - 2 synchronized brightness sperm/      # Super-resolution images, pre-processed images and mask files
    - Run/       # Program running temporary folder
```

## Pipeline

<img width="1208" alt="Pipeline" src="https://github.com/user-attachments/assets/3a04c04b-92f4-46b7-ad8d-d0254febb95d">

The pipeline of our **SpeHeatal**. SpeHeatal works in a “decomposition-combination” manner, utilizing SAM and **Con2Dis** for segmenting heads and tails, respectively, and subsequently splicing them into complete masks.



## Requirements
```
git clone https://github.com/SpeHeaTal/SpeHeaTal.git
pip install -r requirements.txt
```

## Pretrained Models
Please click the links below to download the checkpoint for the desired model type, and place all pretrained models in the root directory.

- `default` or `vit_h`:[ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`:[ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`:[ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Getting Started
This code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. It is recommended to download the Segment Anything Model using the following command:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
In order to run SAM, please ensure you install the necessary dependencies:
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### Preprocessing and segmentation

Initially, place the prepared original sperm images in the `original_image` folder, and name them in the format of 001.jpg, 002.jpg, etc. Subsequently, enhance the resolution using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master) for a fourfold increase in image quality, and save these enhanced images in the `2 synchronized brightness sperm` folder (retain the same naming convention as in the `original_image` folder).

For image preprocessing, preliminary segmentation, and mask screening, please run the `preprocessing_step.py` script:

```
python preprocessing_step.py
```
After that, select the image you need to further segment and set the value of `j` to the name of the image.

Finally, run the segmentation code using the following command:

```
python sperm_segmentation_main.py
```


## References
- [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://github.com/xinntao/Real-ESRGAN/tree/master)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
