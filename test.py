# test to merge the heatmap to enhance the origional image input
# last update by ZJUHITL Team, Nov 11, 2021

import io
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import json
import os
from sklearn.cluster import SpectralClustering

# input image
LABELS_file = 'mydata.json'
path = './data/ROI/train/plane/'
path1 = './data/DOTA/'
filelist = os.listdir(path)

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=False)
    print(net)
    net.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
    net.load_state_dict(torch.load("./model/model_roi.pth"))#网络模型加载训练好的权重
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(2048,2)
    net.load_state_dict(torch.load("./model/model_resnet_roi.pth"))#网络模型加载训练好的权重
    print(net)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'


device=torch.device('cpu')
net.to(device)
net.eval()


def getsingle(feature_conv, weight_softmax, idx):
    bz, nc, h, w = feature_conv.shape #batchsize numchannel h w
    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    #计算13*13的scoremap
    cam = cam - np.min(cam)
    cam = cam/np.max(cam)
    cam_img = np.uint8(255 * cam)
    cam_img = cv2.resize(cam_img, (256, 256))
    #划分25个区域 通过阈值计算筛选得到k个区域
    regions = np.split(cam,7,axis=0)
    #print(regions)
    res = []
    for region in regions:
        res.extend(np.split(region,7,axis=1))
    S = dict()
    for i in range(49):
        S[i] = res[i].sum()
    S = sorted(S.items(),key=lambda d:d[1],reverse=True)
    ui = []
    vi = []
    z = 0
    for item in S:
        ui.append(item[0])
        z = z+1
        if z ==5:
            break
    z = 0
    S.reverse()
    for item in S:
        vi.append(item[0])
        z = z+1
        if z ==5:
            break
    #连接feature_conv中的区域
    patterns1,patterns2 = [],[]
    feature_conv = feature_conv.reshape(nc,h,w)
    # print(feature_conv.shape,cam.shape)
    #print(feature_conv.shape)
    for i in ui:
        h1, w1 = i//7, i%7
        u = feature_conv[:,h1:h1+1,w1:w1+1]
        u = u.flatten()
        patterns1.append(u.tolist())
    for i in vi:
        h1, w1 = i//7, i%7
        u = feature_conv[:,h1:h1+1,w1:w1+1]
        u = u.flatten()
        patterns2.append(u.tolist())
    return patterns1,patterns2,cam_img


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

classpatterns1 = []
classpatterns2 = []

for img in filelist[:]:
    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    # load test image
    img_pil = Image.open(path+img)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # generate class activation mapping for the top1 prediction
    # 或者直接指定某个类别index = 1 即指定plane
    patterns1, patterns2,cam = getsingle(features_blobs[0], weight_softmax, 1)
    img1 = cv2.imread(path+img)
    height, width, _ = img1.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img1 * 0.5
    #cv2.imwrite('test.jpg', result)
    classpatterns1.extend(patterns1)
    classpatterns2.extend(patterns2)
    

print(len(classpatterns1),len(classpatterns2))


C = [[],[]]

C[0] = classpatterns1
C[1] = classpatterns2

print(len(C[0]),len(C[1]))

a0 = np.array(C[0]).mean(0)
a1 = np.array(C[1]).mean(0)
print(a0,a0.shape)
print(a1,a1.shape)
################################################################################

#聚类热图
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

#load testfile
testfile = path+filelist[2]
img_pil = Image.open(testfile)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)
def getHeatMap(feature_conv,ac):
    _, nc, h, w = feature_conv.shape #batchsize numchannel h w
    #连接feature_conv中的区域
    feature_conv = feature_conv.reshape(nc,h,w)
    #print(feature_conv.shape)
    #print(feature_conv.shape)
    heatmap = np.zeros((7,7))
    ac = ac.reshape(nc,1,1)
    for i in range(49):
        h1, w1 = i//7, i%7
        u = feature_conv[:,h1:h1+1,w1:w1+1]
        for x in range(h1,h1+1):
            for y in range(w1,w1+1):
                heatmap[x][y] = np.dot(u[:,x-h1,y-w1],ac[:,x-h1,y-w1])
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    return heatmap


heatmap = getHeatMap(features_blobs[0],a0)
img = cv2.imread(testfile)
height, width, _ = img.shape
heatmap = cv2.resize(heatmap,(width, height))
print(heatmap.shape)
# heatmap = cv2.applyColorMap(cv2.resize(heatmap,(width, height)), cv2.COLORMAP_JET)
# print(heatmap.shape)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite('predict/my/mycluster_1_c0.jpg', result)



b,g,r = cv2.split(img)
combination = np.stack((b, g, r, heatmap), 2)
cv2.imwrite('predict/my/combination.png', combination)