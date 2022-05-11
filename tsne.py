import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pickle
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from Cross_dataset_seedvig_mmd_Hloss import InterpretableCNN_MMD
import torch
from advent import make_ADVENT_model
import scipy.io as sio
from data.read_data import read_data
import os
from dann import dann
from MCD import InterpretableCNN_MCD
from bridgeda import MDDNet
from pseudo import pseudo_label
batch_size = 50
taiwantoseed=1

xtrain,ytrain,testdatax,testdatay,testsubidx,subjnum,channelnum,samplelength,sf,_ = read_data(taiwantoseed=taiwantoseed)
results = np.zeros(subjnum)
work_dir = '/mnt/ylq/liqiang_old_ubuntu/liqiang/PycharmProjects/sharedDA/checkpoint/'
#different model here
model_name = 'our'
file_path = os.path.join(work_dir,'best_'+model_name+str(taiwantoseed))
if model_name in ['without','entropy']:
    model = InterpretableCNN_MMD().double().cuda()
if model_name == 'advent':
    model,_ = make_ADVENT_model()
    model.double().cuda()
if model_name == 'dann':
    model = dann().double().cuda()
if model_name in ['mcd','info','our']:
    model = InterpretableCNN_MCD().double().cuda()
if model_name == 'mdd':
    model = MDDNet().double().cuda()
if model_name == 'cbst':
    model = pseudo_label().double().cuda()
model.load_state_dict(torch.load(file_path)['state_dict'])
model.train(False)


with torch.no_grad():
    for i in range(1, subjnum + 1):
        testindx = np.where(testsubidx == i)[0]
        xtest = testdatax[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)
        y_test = testdatay[testindx]
        # / media / liqiang / WOOD / cuijian / sharedDA
        x_test = torch.DoubleTensor(x_test).cuda()
        answer,_,feature= model(x_test)
        if i==1:
            features = feature.cpu().numpy()
        else:
            features = np.concatenate([features,feature.cpu().numpy()],axis=0)



        # print(acc,x_test.shape)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(features)
print("Org data dimension is {}.Embedded data dimension is {}".format(features.shape[-1], X_tsne.shape[-1]))
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(64, 64))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1],'.', color=plt.cm.Set1(testdatay[i]),
             fontdict={
    'weight': 'bold', 'size': 30})
plt.xticks([])
plt.yticks([])
plt.title(model_name+' taiwantoseed='+str(taiwantoseed))
plt.show()
