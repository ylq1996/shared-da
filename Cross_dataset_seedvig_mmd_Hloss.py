# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN
"""
import torch
import os
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
import mmd
#import coral
#from InterpretableCNN import InterpretableCNN

torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)
"""
 This file performs leave-one-subject cross-subject classification on the driver drowsiness dataset.
 The data file contains 3 variables and they are EEGsample, substate and subindex.
 "EEGsample" contains 2022 EEG samples of size 20x384 from 11 subjects. 
 Each sample is a 3s EEG data with 128Hz from 30 EEG channels.

 The names and their corresponding index are shown below:
 Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, Cz, C4, T4, TP7, CP3, CPz, CP4, TP8, T5, P3, PZ, P4, T6, O1, Oz  O2
 0,    1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11, 12, 13, 14, 15, 16, 17,  18,  19,  20,  21,  22,  23,24, 25, 26, 27, 28, 29

 "subindex" is an array of 2022x1. It contains the subject indexes from 1-11 corresponding to each EEG sample. 
 "substate" is an array of 2022x1. It contains the labels of the samples. 0 corresponds to the alert state and 1 correspond to the drowsy state.
 
  The dataset can be downloaded here:
  https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687
 
  This file prints leave-one-out accuracies for each subject and the overall accuracy.
  The overall accuracy for one run is expected to be 0.7902
  
  If you have met any problems, you can contact Dr. Cui Jian at cuij0006@ntu.edu.sg
"""
class InterpretableCNN(torch.nn.Module):     
    
    def __init__(self, classes=2, sampleChannel=12, sampleLength=384 ,N1=10, d=2,kernelLength=64):
        super(InterpretableCNN, self).__init__()
        self.pointwise = torch.nn.Conv2d(1,N1,(sampleChannel,1))
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1) 
        self.activ=torch.nn.ReLU()       
        self.batchnorm = torch.nn.BatchNorm2d(d*N1,track_running_stats=False)       
        self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))         
        self.fc = torch.nn.Linear(d*N1, classes)        
        self.softmax=torch.nn.Softmax(dim=1)
        
        self.activ1=torch.nn.Tanh() 
        self.batchnorm1 = torch.nn.BatchNorm2d(1,track_running_stats=False)  

    def forward(self, inputdata):
#        intermediate = self.activ1(inputdata)
#        intermediate = self.batchnorm1(intermediate) 
                       
        intermediate = self.pointwise(inputdata) 
#       feature extractor
        intermediate = self.depthwise(intermediate) 
        intermediate = self.activ(intermediate) 
        intermediate = self.batchnorm(intermediate)          
        intermediate = self.GAP(intermediate)     
        intermediate = intermediate.view(intermediate.size()[0], -1)
#       classifier
        intermediate = self.fc(intermediate)    
        output = self.softmax(intermediate)   

        return output 
   
    
class InterpretableCNN_MMD(torch.nn.Module):     
    
    def __init__(self, classes=2, sampleChannel=12, sampleLength=384 ,N1=10, d=2,kernelLength=64):
        super(InterpretableCNN_MMD, self).__init__()
        self.pointwise = torch.nn.Conv2d(1,N1,(sampleChannel,1))
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1) 
        self.activ=torch.nn.ReLU()       
        self.batchnorm = torch.nn.BatchNorm2d(d*N1,track_running_stats=False)       
        self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))         
        self.fc = torch.nn.Linear(d*N1, classes)        
        self.softmax=torch.nn.Softmax(dim=1)
        
        self.activ1=torch.nn.Tanh() 
        self.batchnorm1 = torch.nn.BatchNorm2d(1,track_running_stats=False)  

    def forward(self, inputdata,target):
#        intermediate = self.activ1(inputdata)
#        intermediate = self.batchnorm1(intermediate) 
                       
        intermediate = self.pointwise(inputdata) 

        intermediate = self.depthwise(intermediate) 
        intermediate = self.activ(intermediate) 
        intermediate = self.batchnorm(intermediate)     
        
        
        intermediate = self.GAP(intermediate)     
        intermediate = intermediate.view(intermediate.size()[0], -1) 
        
        sourceout =intermediate         
        
        intermediate = self.fc(intermediate)    
        
        mmd_loss=0
        outputtarget=0
        
        if self.training:
            
            intermediate1 = self.pointwise(target) 
            intermediate1 = self.depthwise(intermediate1) 
            intermediate1 = self.activ(intermediate1) 
            intermediate1 = self.batchnorm(intermediate1)          
            intermediate1 = self.GAP(intermediate1)     
            intermediate1 = intermediate1.view(intermediate1.size()[0], -1)           
            
          #  mmd_loss = mmd_loss+mmd.linear_mmd2(sourceout, intermediate1)     
          #  mmd_loss =mmd.poly_mmd2(sourceout, intermediate1)
            mmd_loss = mmd.mmd_rbf_noaccelerate(sourceout, intermediate1)
          #  mmd_loss = coral.coral(sourceout, intermediate1)  
            
            intermediate1 = self.fc(intermediate1)  
            outputtarget = self.softmax(intermediate1) 
            
          
        output = self.softmax(intermediate) 
        
        
        
        
        
        

        return output, mmd_loss,outputtarget    
    
    
    
def run():
    #    load data from the file

    #########################################################################################
    filename = r'/mnt/ylq/liqiang_old_ubuntu/liqiang/PycharmProjects/sharedDA/journaltaiwanbalanced.mat'

    tmp = sio.loadmat(filename)
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    del tmp

    samplenum = label.shape[0]
    sf = 128
    #  ydata=np.zeros(samplenum)
    ydata = np.zeros(samplenum, dtype=np.longlong)
    for i in range(samplenum):
        ydata[i] = int(label[i])

    selectedchannel = [2, 6, 17, 21, 13, 15, 23, 24, 25, 27, 28, 29]

    #   channelnames=[ 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8','CP1', 'CP2', 'P1','PZ','P2','PO3' ,'POZ', 'PO4', 'O1', 'Oz','O2']
    #     #                   F7    F8                 TP7    TP8    c3     c4     p3   pz   p4                       O1', 'Oz','O2'
    #
    xtrain = np.zeros((xdata.shape[0], 12, xdata.shape[2]))
    for kk in range(12):
        xtrain[:, kk, :] = xdata[:, selectedchannel[kk], :]

    xdata = xtrain

    #############################################################################################
    #########################################################################################
    filename1 = r'/mnt/ylq/liqiang_old_ubuntu/liqiang/PycharmProjects/sharedDA/seedvigadd1.mat'
    #  filename1 = r'D:\DataBase\seed\SEED-VIG\seedvig.mat'

    tmp = sio.loadmat(filename1)
    xdata1 = np.array(tmp['EEGsample'])
    label1 = np.array(tmp['substate'])
    subIdx1 = np.array(tmp['subindex'])

    label1.astype(int)
    subIdx1.astype(int)

    del tmp

    samplenum1 = label1.shape[0]
    sf = 128
    # ydata1=np.zeros(samplenum1)
    ydata1 = np.zeros(samplenum1, dtype=np.longlong)
    for i in range(samplenum1):
        ydata1[i] = int(label1[i])

    selectedchannel = [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    #
    #   selectedchannel=[1,    1, 0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,   1,  0,  1,  0,  1,  0,  1,  0, 1,  0,  1,  0,  1,  0,  1,   1,  1,  0,  1,  0,  1,   0,  1,  1, 1, 0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  1,  1]
    #   print(np.sum(selectedchannel))

    #    tmp = sio.loadmat(filename)
    #    xtraino=np.array(tmp['EEGsampletrain'])

    xtrain = np.zeros((xdata1.shape[0], 12, xdata1.shape[2]))
    cnt = 0
    for kk in range(17):
        if selectedchannel[kk] == 1:
            xtrain[:, cnt, :] = xdata1[:, kk, :]
            cnt = cnt + 1

        ################################################################################

    xdata1 = xtrain

    channelnum = 12

    samplelength = 3
    sf = 128

    #   define the learning rate, batch size and epoches
    lr = 1e-3
    batch_size = 50
    n_epoch = 5000

    ############################################################################################################

    #   it performs leave-one-subject-out training and classfication
    #   for each iteration, the subject i is the testing subject while all the other subjects are the training subjects.

    #  change this between 0 and 1 to switch the direction of transfer
    taiwan_to_seed = 0

    if taiwan_to_seed:
        xtrain = xdata
        ytrain = ydata
        testdatax = xdata1
        testdatay = ydata1
        testsubidx = subIdx1
        sn = samplenum1

        subjnum = 12
    else:
        xtrain = xdata1
        ytrain = ydata1
        testdatax = xdata
        testdatay = ydata
        testsubidx = subIdx
        subjnum = 11
        sn = samplenum

    results = np.zeros(subjnum)

    print(xtrain.shape)

    x_train = xtrain.reshape(xtrain.shape[0], 1, channelnum, samplelength * sf)
    y_train = ytrain  # [trainindx]

    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

#       load the CNN model to deal with 1D EEG signals
  #  my_net = InterpretableCNN().double().cuda()
    ans = []
    for _ in range(1):
        cur = []
        my_net = InterpretableCNN_MMD().double().cuda()


        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.CrossEntropyLoss().cuda()

        for p in my_net.parameters():
            p.requires_grad = True

        best_acc = 0
        for epoch in range(n_epoch):
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data

                slctidx=np.random.choice(sn, labels.size()[0], replace=False)
                xtestbatch=testdatax[slctidx]
                xtestbatch = xtestbatch.reshape(xtestbatch.shape[0], 1,channelnum, samplelength*sf)
                xtestbatch  =  torch.DoubleTensor(xtestbatch).cuda()

                input_data = inputs.cuda()
                class_label = labels.cuda()

                my_net.zero_grad()
                my_net.train()

                class_output,mmdloss,targetout= my_net(input_data,xtestbatch)
                entropy = -torch.sum((torch.log(targetout)*(targetout))/(targetout.size(0)*targetout.size(1)),dim=1)
                # Hloss= torch.sum((1-entropy)*entropy)
                # #

                err_s_label = loss_class(class_output, class_label)+torch.sum(entropy)

                err=err_s_label
                err.backward()
                optimizer.step()

        #
        # optimizer = optim.SGD(my_net.parameters(), lr=1e-3)
        # for epoch in range(100):
        #     for j, data in enumerate(train_loader, 0):
        #         inputs, labels = data
        #
        #         slctidx = np.random.choice(sn, labels.size()[0], replace=False)
        #         xtestbatch = testdatax[slctidx]
        #         xtestbatch = xtestbatch.reshape(xtestbatch.shape[0], 1, channelnum, samplelength * sf)
        #         xtestbatch = torch.DoubleTensor(xtestbatch).cuda()
        #
        #         input_data = inputs.cuda()
        #
        #
        #         my_net.zero_grad()
        #         my_net.train()
        #
        #         class_output, mmdloss, targetout = my_net(input_data, xtestbatch)
        #         entropy = -torch.sum(torch.exp(targetout) * (targetout)) / (targetout.size(0) * targetout.size(1))
        #         # Hloss = (1 - entropy) * entropy
        #
        #
        #
        #         err = entropy
        #         err.backward()
        #         optimizer.step()


    ###validation
    ################################################
            my_net.train(False)
            with torch.no_grad():

                for i in range(1,subjnum+1):

                    testindx=np.where(testsubidx == i)[0]
                    xtest=testdatax[testindx]
                    x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
                    y_test=testdatay[testindx]


                    x_test =  torch.DoubleTensor(x_test).cuda()
                    answer,mmdloss,_ = my_net(x_test,x_test)
                    probs=answer.cpu().numpy()
                    preds       = probs.argmax(axis = -1)
                    acc=accuracy_score(y_test, preds)

                    # print(acc,x_test.shape)
                    results[i-1]=acc


            if np.mean(results)>best_acc:
                best_acc = np.mean(results)
                filename = os.path.join('/mnt/ylq/liqiang_old_ubuntu/liqiang/PycharmProjects/sharedDA/checkpoint',
                                        'weighted0')
                state = {
                    'state_dict': my_net.state_dict(),
                    'best_acc': best_acc
                }
                torch.save(state, filename)
            print('mean accuracy:{},best accuracy:{}'.format(np.mean(results), best_acc))
            ans.append(best_acc)
            cur.append(np.mean(results))
        print(cur)
    # print(np.mean(np.array(ans)))


if __name__ == '__main__':
    run()
    
