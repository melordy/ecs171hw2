
# coding: utf-8

# In[6]:


import two_hidden_layer
import three_hidden_layer
import one_hidden_layer
import pandas as pd

import math
import random
import string
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
random.seed(0)

def loadFile(df):

    file = pd.read_csv(df, delimiter=',', header=None)
    dataset = file.values

    gateInput = dataset[:, 1:9]
    gateOutput = dataset[:, 9:10]

    gateOutput = [item for sublist in gateOutput for item in sublist]
    uniqueOutputList = list(np.unique(gateOutput))
    for i in range(len(gateOutput)):
        gateOutput[i] = uniqueOutputList.index(gateOutput[i]) / 10.0

    resultset = []

    for i in range(len(gateInput)):
        locallist = []
        locallist.append(list(gateInput[i]))
        locallist.append([(gateOutput[i])])

        resultset.append(locallist)

    return ((resultset))

if __name__ == "__main__":

    dataset = loadFile("data.csv")

    train_dataset = dataset[:1040]
    test_dataset = dataset[1040:]
    
    a = []
    for j in range(4):
        if j == 0:
            b = [i for i in (np.arange(0,15,3))]
            a.append(b)

        else:
            a.append([j])

    node_list = [3, 6, 9, 12]

    for i in range(3):
        for j in range(4):
            if i+1 == 1:
                nn = one_hidden_layer.NN(8, node_list[j], 1)
            elif i+1 == 2:
                nn = two_hidden_layer.NN(8 , node_list[j] , node_list[j] ,1)
            else:
                nn = three_hidden_layer.NN(8 , node_list[j] , node_list[j] , node_list[j] , 1)

            train_error_list = nn.train(train_dataset, iterations = 10)

            test_error_list = nn.test(test_dataset)

            a[i+1].append(sum(test_error_list) / len(test_error_list))


min = a[1][1]
row = 0
column = 0
for i in range(len(a)):
    if i == 0:
        continue
    for j in range(len(a[i])):
        if j==0:
            continue
        if a[i][j] < min:
            row = i
            column = j
            min = a[i][j]

for i in a:
    print (i)
#
print("Best structure:  ")

print(" \t\tNumber of hidden layers = {}".format(row) )
print("\t\tNeurons per layer= {}".format(column * 3))

