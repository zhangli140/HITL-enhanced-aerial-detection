import os
data_dir = 'E:/learn/ML_DL/lab/CV/detection/human-in-the-loop/region representation/CAM-master/data/ROI'
save_prefix = 'roi'
epoch = 10
save_path = os.path.join(data_dir+'/model/checkpoints/', f'{save_prefix}{epoch + 1}.pth')
save_path1 = os.path.join(data_dir+'/model/', 'checkpoints')
print(save_path,save_path1)
# device=torch.device('cpu')
# net.to(device)
# net.eval()

# def visualize(pattern,w,h):
#     pattern = pattern.reshape(-1,96)
#     pattern = pattern - np.min(pattern)
#     pattern = pattern / np.max(pattern)
#     pattern = np.uint8(255 * pattern)
#     pattern_img = cv2.applyColorMap(pattern, cv2.COLORMAP_JET)
#     return pattern_img

# def getsingle(feature_conv, weight_softmax, idx):
#     bz, nc, h, w = feature_conv.shape #batchsize numchannel h w
#     cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
#     cam = cam.reshape(h, w)
#     #计算13*13的scoremap
#     cam = cam - np.min(cam)
#     cam = cam/np.max(cam)
#     #填充成15*15
#     cam_img = np.uint8(255 * cam)
#     cam_img = cv2.resize(cam_img, (256, 256))
#     #划分25个区域 通过阈值计算筛选得到k个区域
#     regions = np.split(cam,7,axis=0)
#     #print(regions)
#     res = []
#     for region in regions:
#         res.extend(np.split(region,7,axis=1))
#     S = dict()
#     for i in range(49):
#         S[i] = res[i].sum()
#     S = sorted(S.items(),key=lambda d:d[1],reverse=True)
#     ui = []
#     z = 0
#     for item in S:
#         ui.append(item[0])
#         z = z+1
#         if z ==3:
#             break
#     z = 0
#     S.reverse()
#     for item in S:
#         ui.append(item[0])
#         z = z+1
#         if z ==3:
#             break
#     #连接feature_conv中的区域
#     patterns = []
#     feature_conv = feature_conv.reshape(nc,h,w)
#     #print(feature_conv.shape)
#     # print(feature_conv.shape)
#     for i in ui:
#         h1, w1 = i//7, i%7
#         u = feature_conv[:,h1:h1+1,w1:w1+1]
#         '''检测是否有非负项
#         for r in range(nc):
#             for j in range(3):
#                 for i in range(3):
#                     if u[r][j][i]<0:
#                         print(u[r][j][i])
            
#         '''
#         u = u.flatten()
#         patterns.append(u.tolist())
#     return patterns,cam_img


# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
#    transforms.Resize((224,224)),
#    transforms.ToTensor(),
#    normalize
# ])

# classpatterns = []

# for img in filelist:
#     # hook the feature extractor
#     features_blobs = []
#     def hook_feature(module, input, output):
#         features_blobs.append(output.data.cpu().numpy())

#     net._modules.get(finalconv_name).register_forward_hook(hook_feature)

#     # get the softmax weight
#     params = list(net.parameters())
#     weight_softmax = np.squeeze(params[-2].data.numpy())

#     # load test image
#     img_pil = Image.open(path+img)
#     img_tensor = preprocess(img_pil)
#     img_variable = Variable(img_tensor.unsqueeze(0))
#     logit = net(img_variable)

#     # generate class activation mapping for the top1 prediction
#     # 或者直接指定某个类别index = 1 即指定plane
#     patterns,cam = getsingle(features_blobs[0], weight_softmax, 1)
#     img1 = cv2.imread(path+img)
#     height, width, _ = img1.shape
#     heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
#     result = heatmap * 0.3 + img1 * 0.5
#     #cv2.imwrite('test.jpg', result)
#     classpatterns.extend(patterns)

# print(len(classpatterns))

# X = np.array(classpatterns)
# n_clusters=2
# clustering = SpectralClustering(n_clusters=2,assign_labels="kmeans",random_state=0).fit(X)
# labels = clustering.labels_
# print(clustering.labels_)

# C = [[],[]]

# for i in range(len(classpatterns)):
#     C[labels[i]].append(classpatterns[i])

# # print(len(C[0]),len(C[1]))

# a0 = np.array(C[0]).mean(0)
# a1 = np.array(C[1]).mean(0)
# print(a0.shape)
# print(a1.shape)


# ################################################################################
# #聚类热图
# def getHeatMap(feature_conv,ac):
#     _, nc, h, w = feature_conv.shape #batchsize numchannel h w
#     #连接feature_conv中的区域
#     feature_conv = feature_conv.reshape(nc,h,w)
#     #print(feature_conv.shape)
#     heatmap = np.zeros((15,15))
#     ac = ac.reshape(nc,1,1)
#     for i in range(49):
#         h1, w1 = i//7, i%7
#         u = feature_conv[:,h1:h1+1,w1:w1+1]
#         for x in range(h1,h1+1):
#             for y in range(w1,w1+1):
#                 heatmap[x][y] = np.dot(u[:,x-h1,y-w1],ac[:,x-h1,y-w1])
#     heatmap = heatmap - np.min(heatmap)
#     heatmap = heatmap / np.max(heatmap)
#     heatmap = np.uint8(255 * heatmap)
#     return heatmap

# # hook the feature extractor
# features_blobs = []
# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())

# net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# # get the softmax weight
# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-2].data.numpy())


# #load testfile
# testfile = path1+'3.png'
# img_pil = Image.open(testfile)
# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
# logit = net(img_variable)

# heatmap = getHeatMap(features_blobs[0],a0)

# img = cv2.imread(testfile)
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(heatmap,(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite('predict/heatmap/dota_3_c0.jpg', result)
# #cv2.imwrite('predict/heatmap/heatmap_1_c0.jpg', result)
# heatmap = getHeatMap(features_blobs[0],a1)
# img = cv2.imread(testfile)
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(heatmap,(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite('predict/heatmap/dota_3_c1.jpg', result)
# #cv2.imwrite('predict/heatmap/heatmap_1_c1.jpg', result)
# # 
# #    
# # hook the feature extractor
# features_blobs = []
# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())

# net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# # get the softmax weight
# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-2].data.numpy())


# #load testfile
# testfile = path1+'4.png'
# img_pil = Image.open(testfile)
# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
# logit = net(img_variable)

# heatmap = getHeatMap(features_blobs[0],a0)

# img = cv2.imread(testfile)
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(heatmap,(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite('predict/heatmap/dota_4_c0.jpg', result)
# #cv2.imwrite('predict/heatmap/heatmap_1_c0.jpg', result)
# heatmap = getHeatMap(features_blobs[0],a1)
# img = cv2.imread(testfile)
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(heatmap,(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite('predict/heatmap/dota_4_c1.jpg', result)
# print('t1:',result)
# #cv2.imwrite('predict/heatmap/heatmap_1_c1.jpg', result)