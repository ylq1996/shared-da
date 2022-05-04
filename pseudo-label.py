# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN
"""
import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
import mmd

torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)

class pseudo_label(torch.nn.Module):

    def __init__(self, classes=2, sampleChannel=12, sampleLength=384, N1=10, d=2, kernelLength=64):
        super(pseudo_label, self).__init__()
        self.pointwise = torch.nn.Conv2d(1, N1, (sampleChannel, 1))
        self.depthwise = torch.nn.Conv2d(N1, d * N1, (1, kernelLength), groups=N1)
        self.activ = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(d * N1, track_running_stats=False)
        self.GAP = torch.nn.AvgPool2d((1, sampleLength - kernelLength + 1))
        self.fc = torch.nn.Linear(d * N1, classes)

        self.softmax = torch.nn.Softmax(dim=1)

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

        return output


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
    taiwan_to_seed = 1

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

    my_net = pseudo_label().double().cuda()

    best_acc = 0
    ans = []
    for epoch in range(n_epoch):

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_net.parameters()), lr=lr,
                               weight_decay=1e-5)
        loss_class = torch.nn.CrossEntropyLoss().cuda()

        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            #
            input_data = inputs.cuda()
            class_label = labels.cuda()
            #
            my_net.zero_grad()
            my_net.train()
            #
            output1 = my_net(input_data)
            err_s_label = loss_class(output1, class_label)
            #
            err = err_s_label
            err.backward()
            optimizer.step()
        #generate pseudolabel
        my_net.eval()
        with torch.no_grad():
            entropy_list = []
            pseudo_label_list = []
            selected_index = []
            first_class = []
            second_class = []
            first_class_entropy = []
            second_class_entropy = []

            for i in range(1, subjnum + 1):
                testindx = np.where(testsubidx == i)[0]
                xtest = testdatax[testindx]
                x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)

                x_test = torch.DoubleTensor(x_test).cuda()
                answer = my_net(x_test)
                probs = answer.cpu().numpy()
                entropy = -np.sum(np.multiply(probs,np.log(probs+1e-30)),axis=-1)
                entropy_list.extend(entropy)
                pseudo_label_list.extend(probs.argmax(axis=-1))

        first_class_mask = np.argwhere(np.array(pseudo_label_list)==0)
        second_class_mask = np.argwhere(np.array(pseudo_label_list) == 1)
        #calculate class-balanced entropy threshold
        first_class.extend(np.array(entropy_list)[first_class_mask])
        second_class.extend(np.array(entropy_list)[second_class_mask])
        first_class_entropy.extend(np.array(entropy_list)[first_class_mask])
        second_class_entropy.extend(np.array(entropy_list)[second_class_mask])
        first_class_entropy = np.array(first_class_entropy)
        second_class_entropy = np.array(second_class_entropy)
        p = min(0.2 * (epoch / 50 + 1), 0.5)
        first_class_entropy_sorted = np.sort(np.squeeze(first_class_entropy))
        second_class_entropy_sorted = np.sort(np.squeeze(second_class_entropy))
        first_class_threshold = first_class_entropy[int(p * len(first_class_entropy_sorted))]
        second_class_threshold = second_class_entropy[int(p * len(second_class_entropy_sorted))]

        #select index
        first_class_index = first_class_mask[first_class_entropy < first_class_threshold]
        second_class_index = second_class_mask[second_class_entropy < second_class_threshold]
        selected_index.extend(first_class_index)
        selected_index.extend(second_class_index)



        selected_xdata = testdatax[selected_index]
        selected_xdata = selected_xdata[:,np.newaxis,:,:]
        pseudolabel = np.array(pseudo_label_list)[selected_index]

        #train with pseudo label
        my_net.train()
        new_data = np.concatenate([selected_xdata,x_train],axis=0)
        new_label = np.concatenate([pseudolabel,y_train],axis=0)
        pseudo = torch.utils.data.TensorDataset(torch.from_numpy(new_data), torch.from_numpy(new_label))
        pseudoloader = torch.utils.data.DataLoader(pseudo, batch_size=batch_size, shuffle=True)
        for j, data in enumerate(pseudoloader, 0):
            inputs, labels = data
            input_data = inputs.cuda()
            class_label = labels.cuda()
            my_net.zero_grad()
            my_net.train()
            #
            output1 = my_net(input_data)
            err_s_label = loss_class(output1, class_label)
            #
            err = err_s_label
            err.backward()
            optimizer.step()
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
                answer = my_net(x_test)
                probs = answer.cpu().numpy()
                preds = probs.argmax(axis=-1)
                acc = accuracy_score(y_test, preds)

                # print(acc,x_test.shape)
                results[i - 1] = acc

        cur_acc = np.mean(results)
        ans.append(cur_acc)
        if cur_acc > best_acc:
            best_acc = cur_acc
        print('mean accuracy:{},best accuracy:{}'.format(cur_acc, best_acc))
    print('\n')
    print(ans)
if __name__ == '__main__':
    run()