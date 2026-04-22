# visualize enhanced patterns
# last update by ZJUHITL Team, Dec 9, 2021

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
path = './data/ROI/train/good/'
filelist = os.listdir(path)

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 1
if model_id == 1:
    net = models.squeezenet1_1(pretrained=False)
    net.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
    net.load_state_dict(torch.load("./model/model_roi.pth"))#网络模型加载训练好的权重
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
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
    #填充成15*15
    cam = np.pad(cam,((1,1),(1,1)),'edge')
    cam_img = np.uint8(255 * cam)
    cam_img = cv2.resize(cam_img, (256, 256))
    #划分25个区域 通过阈值计算筛选得到k个区域
    regions = np.split(cam,5,axis=0)
    #print(regions)
    res = []
    for region in regions:
        res.extend(np.split(region,5,axis=1))
    S = dict()
    for i in range(25):
        S[i] = res[i].sum()
    S = sorted(S.items(),key=lambda d:d[1],reverse=True)
    ui = []
    vi = []
    z = 0
    for item in S:
        ui.append(item[0])
        z = z+1
        if z ==6:
            break
    vi = ui[3:6]
    ui = ui[:3]
    #连接feature_conv中的区域
    patterns1,patterns2 = [],[]
    feature_conv = feature_conv.reshape(nc,h,w)
    #print(feature_conv.shape)
    feature_conv = np.pad(feature_conv,((0,0),(1,1),(1,1)),'edge')
    #print(feature_conv.shape)
    for i in ui:
        h1, w1 = i//5*3, i%5*3
        u = feature_conv[:,h1:h1+3,w1:w1+3]
        u = u.flatten()
        patterns1.append(u.tolist())
    for i in vi:
        h1, w1 = i//5*3, i%5*3
        u = feature_conv[:,h1:h1+3,w1:w1+3]
        u = u.flatten()
        patterns2.append(u.tolist())
    return patterns1,patterns2,cam_img

def visualize(pattern,w,h):
    pattern = pattern.reshape(-1,96)
    pattern = pattern - np.min(pattern)
    pattern = pattern / np.max(pattern)
    pattern = np.uint8(255 * pattern)
    pattern_img = cv2.applyColorMap(pattern, cv2.COLORMAP_JET)
    return pattern_img



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
pimg0 = visualize(a0,3,3)
cv2.imwrite('a0.jpg',pimg0)
print(a1,a1.shape)
pimg1 = visualize(a1,3,3)
cv2.imwrite('a1.jpg',pimg1)
cv2.waitKey(0)