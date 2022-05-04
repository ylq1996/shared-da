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
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)


class InterpretableCNN_MCD(torch.nn.Module):

    def __init__(self, classes=2, sampleChannel=12, sampleLength=384, N1=10, d=2, kernelLength=64):
        super(InterpretableCNN_MCD, self).__init__()
        self.pointwise = torch.nn.Conv2d(1, N1, (sampleChannel, 1))
        self.depthwise = torch.nn.Conv2d(N1, d * N1, (1, kernelLength), groups=N1)
        self.activ = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(d * N1, track_running_stats=False)
        self.GAP = torch.nn.AvgPool2d((1, sampleLength - kernelLength + 1))
        self.fc = torch.nn.Linear(d * N1, classes)
        self.fc2 = torch.nn.Linear(d * N1, classes)

        self.softmax = torch.nn.Softmax(dim=1)
        self.softmax2 = torch.nn.Softmax(dim=1)

        self.activ1 = torch.nn.Tanh()
        self.batchnorm1 = torch.nn.BatchNorm2d(1, track_running_stats=False)

    def forward(self, inputdata):
        #        intermediate = self.activ1(inputdata)
        #        intermediate = self.batchnorm1(intermediate)

        #       feature extractor
        intermediate = self.pointwise(inputdata)
        intermediate = self.depthwise(intermediate)
        intermediate = self.activ(intermediate)
        intermediate = self.batchnorm(intermediate)
        intermediate = self.GAP(intermediate)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        #       classifier 1
        intermediate1 = self.fc(intermediate)
        output = self.softmax(intermediate1)
        #       classifier 2
        intermediate2 = self.fc2(intermediate)
        output2 = self.softmax2(intermediate2)
        return output,output2
    def stepA_parameters(self):
        for name,params in self.named_parameters():
            params.requires_grad = True

    def stepB_parameters(self):
        for name,params in self.named_parameters():
            if 'fc' not in name:
                params.requires_grad = False

    def stepC_parameters(self):
        for name, params in self.named_parameters():
            params.requires_grad = True
            if 'fc' in name:
                params.requires_grad = False

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
        my_net = InterpretableCNN_MCD().double().cuda()
        best_acc = 0
        for epoch in range(n_epoch):
            #step A
            my_net.stepA_parameters()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_net.parameters()), lr=lr,
                                  weight_decay=1e-5)


            loss_class = torch.nn.CrossEntropyLoss().cuda()


            for epoch in range(1):
                for j, data in enumerate(train_loader, 0):
                    inputs, labels = data
        #
            #
                    input_data = inputs.cuda()
                    class_label = labels.cuda()
            #
                    my_net.zero_grad()
                    my_net.train()
            #
                    output1, output2 = my_net(input_data)
                    err_s_label = loss_class(output1, class_label)+loss_class(output2, class_label)
            #
                    err = err_s_label
                    err.backward()
                    optimizer.step()
            # step B
            my_net.stepB_parameters()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_net.parameters()), lr=lr,
                                   weight_decay=1e-5)

            loss_class = torch.nn.NLLLoss().cuda()
            for epoch in range(1):
                for j, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    #
                    slctidx = np.random.choice(sn, labels.size()[0], replace=False)
                    xtestbatch = testdatax[slctidx]
                    xtestbatch = xtestbatch.reshape(xtestbatch.shape[0], 1, channelnum, samplelength * sf)
                    xtestbatch = torch.DoubleTensor(xtestbatch).cuda()
                    #
                    input_data = inputs.cuda()
                    class_label = labels.cuda()
                    #
                    my_net.zero_grad()
                    my_net.train()
                    #
                    output1, output2 = my_net(input_data)

                    err_s_label = loss_class(output1, class_label) + loss_class(output2, class_label)


                    output1, output2 = my_net(xtestbatch)
                    entropy1 = -torch.sum(torch.log(output1) * (output1), dim=1)
                    entropy2 = -torch.sum(torch.log(output2) * (output2), dim=1)
                    #infomax
                    mean_softmax1 = output1.mean(dim=0)
                    mean_softmax2 = output2.mean(dim=0)
                    info_loss = (torch.sum(-mean_softmax2*torch.log(mean_softmax2+1e-5))+torch.sum(-mean_softmax1*torch.log(mean_softmax1+1e-5)))/2
                    # discrepancy_loss = err_s_label - torch.sum(torch.abs((output2) - (output1)))
                    discrepancy_loss = err_s_label+((torch.sum(entropy1)+torch.sum(entropy2))/100-info_loss)- torch.sum(torch.abs((output2) - (output1)))


                    err = discrepancy_loss
                    err.backward()
                    optimizer.step()

            # #step C
            my_net.stepC_parameters()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_net.parameters()), lr=lr,
                                   weight_decay=1e-5)
            for epoch in range(1):
                for j, data in enumerate(train_loader, 0):
                    inputs, labels = data
                #
                    slctidx = np.random.choice(sn, labels.size()[0], replace=False)
                    xtestbatch = testdatax[slctidx]
                    xtestbatch = xtestbatch.reshape(xtestbatch.shape[0], 1, channelnum, samplelength * sf)
                    xtestbatch = torch.DoubleTensor(xtestbatch).cuda()

                    my_net.zero_grad()
                    my_net.train()

                    output1, output2 = my_net(xtestbatch)
                    entropy1 = -torch.sum(torch.log2(output1) * (output1), dim=1)
                    entropy2 = -torch.sum(torch.log2(output2) * (output2), dim=1)
                    discrepancy_loss = torch.sum(torch.abs((output2)-(output1)))+((torch.sum(entropy1)+torch.sum(entropy2))/100)

                    err = discrepancy_loss
                    err.backward()
                    optimizer.step()



        #

            ###validation
            ################################################
            my_net.train(False)
            with torch.no_grad():

                for i in range(1, subjnum + 1):
                    testindx = np.where(testsubidx == i)[0]
                    xtest = testdatax[testindx]
                    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)
                    y_test = testdatay[testindx]
                    # / media / liqiang / WOOD / cuijian / sharedDA
                    x_test = torch.DoubleTensor(x_test).cuda()
                    answer, answer2 = my_net(x_test)
                    probs = answer.cpu().numpy()
                    preds = probs.argmax(axis=-1)
                    acc = accuracy_score(y_test, preds)

                    # print(acc,x_test.shape)
                    results[i - 1] = acc

            cur_acc = np.mean(results)
            cur.append(cur_acc)
            if cur_acc > best_acc:
                best_acc = cur_acc
                filename = os.path.join('/mnt/ylq/liqiang_old_ubuntu/liqiang/PycharmProjects/sharedDA/checkpoint', 'ours')
                state = {
                    'state_dict': my_net.state_dict(),

                    'best_acc': best_acc
                }
                torch.save(state, filename)
            print('mean accuracy:{},best accuracy:{}'.format(cur_acc, best_acc))
        ans.append(best_acc)
    print(np.mean(np.array(ans)))
    print('\n')
    print(cur)


if __name__ == '__main__':
    run()