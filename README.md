# Aerial Image Enhancement Algorithms for Object Detection with Human-In-The-Loop

## Introduction
Detecting small objects is challenging especially for aerial images. Human experience can potentially provide attentional priors as additional information to enhance small object detection performance.

This algorithm is the key module for our Human-In-The-Loop aerial detection systems, in which a classification network is trained to generate an activation heatmap on the input image for a user-specified target class. The heatmap can be projected onto the input image as an additional input channel that provides a soft semantic guidance to enhance the vision features in following detection work. 
![Results](heatmap.png)
## Bases and Related Projects
### Bases
Our implementation is based on two opensource projects:
- https://github.com/machrisaa/tensorflow-vgg (we build our feature extractor on it)
- https://github.com/zhoubolei/CAM (we build our class pattern learner on it)
Both of them are following MIT License (almost no restricts to use).

### Related Projects for Experiment Baselines
Though our full aerial detection system is not fully open, we have related open projects as baselines which can be used to reproduce the comparison to our full experiments for our system.
- https://github.com/xinntao/Real-ESRGAN (it can enhance the aerial image to super-resolution)
- https://github.com/dingjiansw101/AerialDetection (it can be adapt to our enhanced input as a backbone for detection)
- https://github.com/fanq15/FSOD-code (it can be used as the primary detection tool for cutting aerial image for meaningful sub-regions)

## Requirement and Install
### General Guide
Follow the install and usage documents for Bases and Related Projects in their sites.

### Install VGG
* Install [tensorflow](https://github.com/tensorflow/tensorflow/blob/v1.0.0-rc1).
* Our code already contains the tensorflow-vgg and its models.

### Install CAM
* Our code already contains the pretrained models, so you don't need to download.
* Install [caffe](https://github.com/BVLC/caffe), compile the matcaffe (matlab wrapper for caffe), and make sure you could run the prediction example code classification.m.
* Clone the code from Github
* Run the demo code to generate the heatmap: in matlab terminal, 
```
demo
```
* Run the demo code to generate bounding boxes from the heatmap: in matlab terminal,
```
generate_bbox
```

## Demo Data
Please refer to DOTA and COCO datasets. We also built a demo dataset, you can download it from https://zenodo.org/records/19692661?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImNjNjk4OWE2LTdlY2YtNGQ5OC1hYjE2LWE3NzFmNTk5ZjdlZiIsImRhdGEiOnt9LCJyYW5kb20iOiJiOTU1YzFmYzRiNDY3NDQ4N2M5YjdkNTYyN2VjYzIzMCJ9.lwL3NpeeuZv2kjPMDG0araM0qiKKrnZ5DI0l_M0IrOBpPcDkur96vm-LuRg-NivU0Blqu46RoVkbb2qpX6bP7g

## Usage
### Enhance The Class Features
#### Train VGG
Actually we already trained the model, which can be downloaded in https://zenodo.org/records/19692661?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImNjNjk4OWE2LTdlY2YtNGQ5OC1hYjE2LWE3NzFmNTk5ZjdlZiIsImRhdGEiOnt9LCJyYW5kb20iOiJiOTU1YzFmYzRiNDY3NDQ4N2M5YjdkNTYyN2VjYzIzMCJ9.lwL3NpeeuZv2kjPMDG0araM0qiKKrnZ5DI0l_M0IrOBpPcDkur96vm-LuRg-NivU0Blqu46RoVkbb2qpX6bP7g

If you want to retrain it, just two steps:
1. Prepare data
* Download dataset, including images and labels
* Modify the path of images and labels in vgg-tensorflow-modified/get_features.py
2. Exectue trainer 
```
python vgg-tensorflow-modified/get_features.py
```
3. Test
```
python vgg-tensorflow-modified/test_vgg.py
```
#### Fine-tune ROI
1. Prepare data
Put the training, validation and test data into ./data/ROI
2. Fine-tune
```
python finetune.py
```
#### Extract Class Patterns
0. Set the input and output paths in build_patterns.py
1. Extract Patterns (in form of classificatin score heatmaps)
```
python build_patterns.py
```
2. View Test Patterns, and see /predict/my or predict/heatmap
```
python test.py
```
#### Generate Enhanced Images
0. Set the input and output paths in stod_mycluster.py and stod.py
1. Enhance Images with Patterns
```
python stod_mycluster.py
python stod.py
```
2. See enhanced images like /predict/all or /predict/VHR
### Full Detection with ROI Transformer
1. Download and install FSOD, use it to select meaningful region in images (Optional)
2. Download and install Real-ESRGAN，use it to improve the resolution of the data (Optional)
3. Enhanced the image like above
4. Download and install AerialDetection, modify its input head to more channels
5. Copy the data to the input folder for AerialDetection, use it to train and test the results

## Reference:
The code if for a paper under review

```
@inproceedings{xxxx,
    author    = {Qiu, Xingye and Chen, chenhuan and Zhang, Li},
    title     = {An Aerial Detection System with Human-In-The-Loop},
    booktitle = {The Visual Computer (Under Review)},
    year      = {2026}
}
```
## License:
The pre-trained models and the code are released for unrestricted use.
