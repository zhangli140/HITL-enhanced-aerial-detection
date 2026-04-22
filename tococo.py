"""
Created on 5/11
读取txt文件，划分训练集和测试集并且生成coco格式的json文件
@author: Wu
"""
import json
import os
import numpy as np
import pandas as pd
import re
import cv2
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# generate class name dict
class_name = ['airplane']
label_dict = {}
i = 1
for hahah in class_name:
    label_dict[i] = hahah
    i += 1
# output path
im_path = 'E:/learn/ML_DL/lab/CV/detection/human-in-the-loop/region representation/CAM-master/data/train/plane/images'
ann_path = 'E:/learn/ML_DL/lab/CV/detection/human-in-the-loop/region representation/CAM-master/data/train/plane/ground truth'
output_ann_path = 'E:/learn/ML_DL/lab/CV/detection/human-in-the-loop/region representation/CAM-master/data/train/plane/train.json'


def transform_vhr2coco(ann_path, im_path, output_ann_path):

    '''
    Param:
        ann_path txt标注所在路径
        im_path positive 图片所在路径
        out_ann_path 输出文件路径及命名
    '''

    # 初始化dataset
    datasets = dict()
    annotation_id = [0]
    datasets['images'] = []
    datasets['type'] = 'instances'
    datasets['annotations'] = []
    datasets['categories'] = []
    datasets['info'] = None
    datasets['licenses'] = None
    
    # add dataset['categories']
    for category_id, category_name in label_dict.items():
        category_item = dict()
        category_item['supercategory'] = category_name
        category_item['id'] = category_id
        category_item['name'] = category_name
        datasets['categories'].append(category_item)

    # split train test set
    train_ids = os.listdir(ann_path)
    # iter through every txt to generate train.json and test.json

    for index, ann_filename in enumerate(train_ids):  
        print(f'processing {index} th txt in {i}th dataset')
        # add dataset['images']
        img_name = ann_filename[0:-3]+'jpg'
        image = dict()
        image['id'] = index
        image['file_name'] = img_name
        print(img_name)
        img = cv2.imread(os.path.join(im_path, img_name))
        print(os.path.join(im_path, img_name))
        image['width'] = img.shape[1]
        image['height'] = img.shape[0]
        datasets['images'].append(image)

        ann_filepath = os.path.join(ann_path, ann_filename)
        ann_df = pd.read_csv(ann_filepath, header=None)
        # iter through every annotation on one image
        for _, ann in ann_df.iterrows():
            # add annotation
            x = int(ann[0][1:])
            y = int(ann[1][0:-1])
            w = int(ann[2][1:]) - x
            h = int(ann[3][0:-1]) - y
            label = int(ann[4])
            annotation_item = dict()
            annotation_item['segmentation'] = [[x, y, x, y+h, x+w, y+h, x+w, y]]
            annotation_item['image_id'] = image['id']
            annotation_item['iscrowd'] = 0
            annotation_item['bbox'] = [x, y, w, h]
            annotation_item['area'] = w * h
            annotation_item['id'] = annotation_id[0]
            annotation_id[0] = annotation_id[0] + 1
            annotation_item['category_id'] = label
            datasets['annotations'].append(annotation_item)
        json.dump(datasets, open(output_ann_path, 'w'))

if __name__ == '__main__':
    transform_vhr2coco(ann_path=ann_path, im_path=im_path, output_ann_path=output_ann_path)
