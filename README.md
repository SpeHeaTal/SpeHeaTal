### SpeHeaTal

## A Cluster-Enhanced Segmentation Method for Sperm Morphology Analysis

<img width="752" alt="show" src="https://github.com/user-attachments/assets/218a42cc-852d-4f44-9e7d-f78ab532c8f5">


The accurate assessment of sperm morphology is crucial in andrological diagnostics, where the segmentation of sperm images presents significant challenges. Existing approaches frequently rely on large annotated datasets and often struggle with the segmentation of overlapping sperm and the presence of dye impurities. To address these challenges, this paper first analyzes the issue of overlapping sperm tails from a geometric perspective and introduces a novel clustering algorithm, **Con2Dis**, which effectively segments overlapping tails by considering three essential factors: **Con**nectivity, **Con**formity, and **Dis**tance. Building on this foundation, we propose an unsupervised method, **SpeHeatal**, designed for the comprehensive segmentation of the **SPE**rm **HEA**d and **TA**i**L**. **SpeHeatal** employs the Segment Anything Model (SAM) to generate masks for sperm heads while filtering out dye impurities, utilizes **Con2Dis** to segment tails, and then applies a tailored mask splicing technique to produce complete sperm masks. Experimental results underscore the superior performance of **SpeHeatal**, particularly in handling images with overlapping sperm.

## Pipeline

<img width="1208" alt="Pipeline" src="https://github.com/user-attachments/assets/3a04c04b-92f4-46b7-ad8d-d0254febb95d">


The pipeline of our **SpeHeatal**. SpeHeatal works in a “decomposition-combination” manner, utilizing SAM and **Con2Dis** for segmenting heads and tails, respectively, and subsequently splicing them into complete masks.

## Install
```
git clone https://github.com/SpeHeaTal/SpeHeaTal.git
pip install -r requirements.txt
```


## Super resolution
When you need to use your own pictures for testing, please use Real-ESRGAN super-resolution technology to perform four times super-resolution on your own pictures.
[Real-ESRGAN.](https://github.com/xinntao/Real-ESRGAN/tree/master)


## Data and Pretrained Models

```
original_image   //Sperm original image folder.
2 synchronized brightness sperm   //Super-resolution images, pre-processed images and mask files.
Run   //Program running temporary folder.
```

Click the links below to download the checkpoint for the corresponding model type. And put all pretrained models in the root directory.

- `default` or `vit_h`:[ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`:[ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`:[ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Getting Started
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. First you need to download the Segment Anything Model.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
In order to run SAM, you also need to install the following dependencies.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
### Preprocessing and segmentation

  
First, place the prepared original sperm images in the original_image folder and name them in the format of 001.jpg, 002.jpg... Then run [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master) for four times super-resolution, and save the four-fold super-resolution images in the `2 synchronized brightness sperm` folder (the corresponding image remains named the same as in `original_image` folder).

Then run the `preprocessing_step.py` for image preprocessing, preliminary segmentation and mask screening:
```
python preprocessing_step.py
```
Subsequently, select the image you need to further segment and set the value of ` j`  to the name of the image.

Then，running!
```
python sperm_segmentation_main.py
```


## References
- [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://github.com/xinntao/Real-ESRGAN/tree/master)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
