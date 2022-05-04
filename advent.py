import torch
from torch import nn
import torch.backends.cudnn as cudnn
from utils.discriminator import get_fc_discriminator
import torch.optim as optim
from utils.func import loss_calc, bce_loss
from utils.loss import entropy_loss
from utils.func import prob_2_entropy
import torch.nn.functional as F
import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
import mmd
def make_ADVENT_model():
    model = ADVENTNet()
    return model,model.parameters()
class ADVENTNet(nn.Module):
    def __init__(self, classes=2, sampleChannel=12, sampleLength=384, N1=10, d=2, kernelLength=64):
        super(ADVENTNet, self).__init__()
        self.pointwise = torch.nn.Conv2d(1, N1, (sampleChannel, 1))
        self.depthwise = torch.nn.Conv2d(N1, d * N1, (1, kernelLength), groups=N1)
        self.activ = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(d * N1, track_running_stats=False)
        self.GAP = torch.nn.AvgPool2d((1, sampleLength - kernelLength + 1))
        self.fc = torch.nn.Linear(d * N1, classes)


        self.softmax = torch.nn.Softmax(dim=1)

        self.activ1 = torch.nn.Tanh()
        self.batchnorm1 = torch.nn.BatchNorm2d(1, track_running_stats=False)






    def forward(self,inputdata):
        #       feature extractor
        featutres = self.pointwise(inputdata)
        featutres = self.depthwise(featutres)
        featutres = self.activ(featutres)
        featutres = self.batchnorm(featutres)
        featutres = self.GAP(featutres)
        featutres = featutres.view(featutres.size()[0], -1)

        #       classifier 1
        intermediate1 = self.fc(featutres)
        output = self.softmax(intermediate1)
        output = self.softmax(output)


        return output,featutres
def train_advent(model, params):

    # yi da dui
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


    ''' UDA training with advent
    '''
    # Create the model and start the training.

    device = 0
    num_classes = 2


    # SEGMNETATION NETWORK
    model.train()
    model.double().cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.double().cuda()

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.double().cuda()

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.Adam(params,
                          lr= 1e-3,
                          weight_decay=0.0005)

    # discriminators' optimizers

    optimizer_d_main = optim.Adam(d_main.parameters(), lr= 1e-3,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps


    # labels for adversarial training
    source_label = 0
    target_label = 1
    best_acc = 0
    cur = []
    for i in (range(n_epoch)):
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            slctidx = np.random.choice(sn, labels.size()[0], replace=False)
            xtestbatch = testdatax[slctidx]
            xtestbatch = xtestbatch.reshape(xtestbatch.shape[0], 1, channelnum, samplelength * sf)
            xtestbatch = torch.DoubleTensor(xtestbatch).cuda()
        # reset optimizers
            optimizer.zero_grad()

            optimizer_d_main.zero_grad()


            # UDA Training
            # only train segnet. Don't accumulate grads in disciminators
            for param in d_aux.parameters():
                param.requires_grad = False
            for param in d_main.parameters():
                param.requires_grad = False
            # train on source


            images_source, labels = inputs.to(device, dtype=torch.double), labels.to(device=device, dtype=torch.long)
            pred_src_main,pred_src_aux = model(images_source)
            loss_seg_src_main = torch.nn.CrossEntropyLoss().to(device)(pred_src_main,labels)
            loss =  loss_seg_src_main

            loss.backward()
            del loss


            # adversarial training ot fool the discriminator


            images = xtestbatch.to(device, dtype=torch.double)
            pred_trg_main,pred_trg_aux,  = model(images)
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main,dim=1)))
            loss_adv_trg_main = bce_loss(d_out_main, source_label)
            entropy = entropy_loss(pred_trg_main)
            loss = ( loss_adv_trg_main+entropy
                    )

            loss.backward()
            del loss


            # Train discriminator networks
            # enable training mode on discriminator networks

            for param in d_main.parameters():
                param.requires_grad = True

            # train with source
            pred_src_main = pred_src_main.detach()
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main,dim=1)))
            loss_d_main = bce_loss(d_out_main, source_label)/2
            loss_d_main.backward()
            del loss_d_main,pred_src_main,pred_src_aux



            # train with target

            images = xtestbatch.to(device, dtype=torch.double)
            pred_trg_main, pred_trg_aux, = model(images)

            pred_trg_main = pred_trg_main.detach()
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main,dim=1)))
            entropy = entropy_loss(pred_trg_main)
            loss_d_main = bce_loss(d_out_main, target_label)/2+entropy

            loss_d_main.backward()
            optimizer.step()
            optimizer_d_main.step()
            del loss_d_main,pred_trg_main,pred_trg_aux

            optimizer.step()
            optimizer_d_main.step()
        ###validation
        ################################################
        model.train(False)
        with torch.no_grad():

            for i in range(1, subjnum + 1):
                testindx = np.where(testsubidx == i)[0]
                xtest = testdatax[testindx]
                x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)
                y_test = testdatay[testindx]
                # / media / liqiang / WOOD / cuijian / sharedDA
                x_test = torch.DoubleTensor(x_test).cuda()
                answer,_ = model(x_test)
                probs = answer.cpu().numpy()
                preds = probs.argmax(axis=-1)
                acc = accuracy_score(y_test, preds)

                # print(acc,x_test.shape)
                results[i - 1] = acc

        cur_acc = np.mean(results)
        cur.append(cur_acc)
        if cur_acc > best_acc:
            best_acc = cur_acc
        print('mean accuracy:{},best accuracy:{}'.format(cur_acc, best_acc))
    print('\n')
    print(cur)
if __name__ == '__main__':

    model,params = make_ADVENT_model()
    train_advent(model,params)

