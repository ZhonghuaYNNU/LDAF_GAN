import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

import data
import LDAF_gan


def select_negative(realData, num_pm, num_zr, count):
    data = np.array(realData)
    n_dis_pm = np.zeros_like(data)
    n_dis_zr = np.zeros_like(data)
    all_dis_index = []
    for i in range(data.shape[0]):
        p_dis = np.where(data[i] != 0)[0]
        all_dis_index_1 = random.sample(range(data.shape[1]), count)

        for j in all_dis_index_1:
            if j not in p_dis:
                all_dis_index.append(j)

        random.shuffle(all_dis_index)
        n_dis_index_pm = all_dis_index[0: num_pm]
        n_dis_index_zr = all_dis_index[num_pm: (num_pm + num_zr)]
        n_dis_pm[i][n_dis_index_pm] = 1
        n_dis_zr[i][n_dis_index_zr] = 1
    return n_dis_pm, n_dis_zr


def main(lncCount, disCount, testSet, trainVector, trainMaskVector, epochCount, pro_ZR, pro_PM, alpha):

    # Build the generator and discriminator
    G = LDAF_gan.generator(disCount)
    D = LDAF_gan.discriminator(disCount)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    G_step = 5
    D_step = 2
    batchSize_G = 16
    batchSize_D = 16
    result_auc_sum = 0
    noise = np.random.uniform(0, 1, size=[lncCount, disCount])
    noise = torch.Tensor(noise)
    for epoch in range(epochCount):

        # ---------------------
        #  Train Generator
        # ---------------------

        for step in range(G_step):
            leftIndex = random.randint(1, lncCount - batchSize_G - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            noise_G = Variable(copy.deepcopy(noise[leftIndex:leftIndex + batchSize_G]))

            n_dis_pm, n_dis_zr = select_negative(realData, pro_PM, pro_ZR, disCount)

            ku_zp = Variable(torch.tensor(n_dis_pm + n_dis_zr))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(torch.zeros_like(realData)) * ku_zp

            fakeData = G(noise_G)
            fakeData_ZP = fakeData * (eu + ku_zp)
            fakeData_result = D(fakeData_ZP)

            # Train the discriminator
            g_loss = np.mean(np.log(1. - fakeData_result.detach().numpy() + 10e-5)) + alpha * regularization(
                fakeData_ZP, realData_zp)

            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, lncCount - batchSize_D - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_D]))
            eu = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_D]))
            noise_D = Variable(copy.deepcopy(noise[leftIndex:leftIndex + batchSize_D]))

            n_dis_pm, _ = select_negative(realData, pro_PM, pro_ZR, disCount)
            ku = Variable(torch.tensor(n_dis_pm))

            fakeData = G(noise_D)
            fakeData_ZP = fakeData * (eu + ku)

            # Train the discriminator
            fakeData_result = D(fakeData_ZP)
            realData_result = D(realData)

            d_loss = -np.mean(np.log(realData_result.detach().numpy() + 10e-5) +
                              np.log(1. - fakeData_result.detach().numpy() + 10e-5)) + 0 * regularization(fakeData_ZP,
                                                                                                          realData_zp)

            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        if (epoch % 20 == 0):
            n_user = len(testSet)
            index = 0
            auc_result_G_all = []
            label = []
            pred = []

            for testUser in range(n_user):

                data = Variable(copy.deepcopy(noise[testUser]))
                result_G = G(data.reshape(1, disCount))
                result1 = result_G.reshape(disCount).detach().numpy()
                result2 = result_G + Variable(copy.deepcopy(trainMaskVector[index]))
                result3 = result2.reshape(disCount)

                test_i = testSet[testUser]
                test = [0] * disCount
                for value in test_i:
                    test[value] = 1
                pred_i = list(result1)
                train_i = trainVector[testUser].tolist()

                for i in range(disCount - 1, -1, -1):
                    if train_i[i] == 1:
                        del test[i]
                        del pred_i[i]

                label += test
                pred += pred_i

            auc = roc_auc_score(label, pred)
            auc_result_G_all.append(auc)

            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},auc:{}'.format(epoch, epochCount, d_loss.item(), g_loss.item(), auc))
    result_auc_ave = result_auc_sum / epochs
    print(result_auc_ave)

    return

if __name__ == '__main__':
    topN = 5
    epochs = 2000
    pro_ZR = 30
    pro_PM = 30
    alpha = 0.1

    trainSet, train_lncRNA, train_disease = data.loadTrainingData("5_fold_mine_train2", " ")

    testSet, test_lncRNA, test_disease = data.loadTestData("5_fold_mine_test2", " ")

    lncCount = max(train_lncRNA, test_lncRNA)
    disCount = max(train_disease, test_disease)
    disease_List_test = list(testSet.keys())
    print(lncCount)
    print(disCount)

    trainVector, trainMaskVector, batchCount = data.to_Vectors(trainSet, lncCount, disCount, disease_List_test, "lncBased")

    result_precision = main(lncCount, disCount, testSet, trainVector, trainMaskVector, epochs, pro_ZR, pro_PM, alpha)



