# similar for clustering the heatmap (pytorch_CAM, but for showing)
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
from sklearn.preprocessing import StandardScaler
import math

# input image
LABELS_file = 'mydata.json'
path = './data/ROI/train/good/'
filelist = os.listdir(path)

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 1
if model_id == 1:
    net = models.squeezenet1_1(pretrained=False)
    net.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
    net.load_state_dict(torch.load("./model/model_roi.pth"))#网络模型加载训练好的权重 load trained roi weights to model
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

def visualize(pattern,w,h):
    pattern = pattern.reshape(-1,96)
    pattern = pattern - np.min(pattern)
    pattern = pattern / np.max(pattern)
    pattern = np.uint8(255 * pattern)
    pattern_img = cv2.applyColorMap(pattern, cv2.COLORMAP_JET)
    return pattern_img

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
    z = 0
    for item in S:
        ui.append(item[0])
        z = z+1
        if z ==3:
            break
    z = 0
    S.reverse()
    for item in S:
        ui.append(item[0])
        z = z+1
        if z ==3:
            break
    #连接feature_conv中的区域
    patterns = []
    feature_conv = feature_conv.reshape(nc,h,w)
    #print(feature_conv.shape)
    feature_conv = np.pad(feature_conv,((0,0),(1,1),(1,1)),'edge')
    #print(feature_conv.shape)
    for i in ui:
        h1, w1 = i//5*3, i%5*3
        u = feature_conv[:,h1:h1+3,w1:w1+3]
        '''检测是否有非负项
        for r in range(nc):
            for j in range(3):
                for i in range(3):
                    if u[r][j][i]<0:
                        print(u[r][j][i])
            
        '''
        u = u.flatten()
        patterns.append(u.tolist())
    return patterns,cam_img


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

classpatterns = []

for img in filelist:
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
    patterns,cam = getsingle(features_blobs[0], weight_softmax, 1)
    img1 = cv2.imread(path+img)
    height, width, _ = img1.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img1 * 0.5
    #cv2.imwrite('test.jpg', result)
    classpatterns.extend(patterns)

print(len(classpatterns))

X = np.array(classpatterns)

n_clusters=2
clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(X)
labels = clustering.labels_
print(clustering.labels_)

C = [[],[]]

for i in range(len(classpatterns)):
    C[labels[i]].append(classpatterns[i])

print(len(C[0]),len(C[1]))


a0 = np.array(C[0]).mean(0)
'''
a1 = np.array(C[1]).mean(0)
print(a0,a0.shape)
f1 = open("clusterpattern.txt",'w',encoding='utf-8',errors='ignore')
tmp = [str(a0[i]) for i in range(len(a0))]
f1.write(','.join(tmp))
f1.close()'''


from keras.layers import Input, Dense
from keras.models import Model

X = np.array(C[0])
sc = StandardScaler()
X = sc.fit_transform(X)
reduced_dim = 32
input_data = Input(shape=(X.shape[1],))
encoded1 = Dense(128, activation='relu')(input_data)
encoded2 = Dense(reduced_dim, activation='relu')(encoded1)
decoded1 = Dense(128, activation='relu')(encoded2)
decoded2 = Dense(X.shape[1], activation=None)(decoded1)

autoencoder = Model(input_data, decoded2)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

print(autoencoder.summary())

autoencoder.fit(X,X,
                epochs=1000,
                batch_size=16,
                shuffle=True)

#Encoder
encoder = Model(input_data, encoded2)
#Decoder
decoder = Model(input_data, decoded2)
a0 = a0.reshape(1,4608)
encoded_X = encoder.predict(a0)
print(encoded_X)
print(encoded_X.shape)

'''
decoded_X = decoder.predict(encoded_X)
X_decoded_ae = sc.inverse_transform(decoded_X)
print(decoded_X.shape)


def my_rmse(np_arr1,np_arr2):
    dim = np_arr1.shape
    tot_loss = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            tot_loss += math.pow((np_arr1[i,j] - np_arr2[i,j]),2)
    return round(math.sqrt(tot_loss/(dim[0]* dim[1]*1.0)),2)


error_dae = my_rmse(X,X_decoded_ae)
print(error_dae)
'''