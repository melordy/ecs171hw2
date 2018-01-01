
# coding: utf-8

# In[2]:


import math
import random
import string
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')

random.seed(0)



import one_hidden_layer

def loadFile(df):

    file = pd.read_csv(df, delimiter=',', header=None)
    dataset = file.values

    gateInput = dataset[:, 1:9]
    gateOutput = dataset[:, 9:10]

    gateOutput = [item for sublist in gateOutput for item in sublist]
    uniqueOutputList = list(np.unique(gateOutput))

    # print uniqueOutputList
    for i in range(len(gateOutput)):
        gateOutput[i] = uniqueOutputList.index(gateOutput[i]) / 10.0

    resultset = []

    for i in range(len(gateInput)):
        locallist = []
        locallist.append(list(gateInput[i]))
        locallist.append([(gateOutput[i])])

        resultset.append(locallist)

    return ((resultset))

def question1():

    #train_dataset = loadFile("train_data.csv")
    #test_dataset = loadFile("test_data.csv")


    dataset = loadFile("data.csv")
    random.shuffle(dataset)

    train_dataset = dataset[:1040]
    test_dataset = dataset[1040:]
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # create a network with two input, two hidden, and one output nodes
    n = one_hidden_layer.NN(8, 3, 1)
    # train it with some patterns
    n.train(train_dataset , iterations=10)
    n.test(test_dataset)

    n.plotGraph_cyt_weight_bases(n.cyt_weights_list, n.cyt_bias_list, "output weights and bias for CYT class samples")
    n.plotGraph_error(n.cyt_train_error_list, n.cyt_test_error_list, "train vs test error for CYT class samples")

    
def question2():

    fulldataset = loadFile("data.csv")
    # create a network with two input, two hidden, and one output nodes
    n = one_hidden_layer.NN(8, 3, 1)
    # train it with some patterns
    error = n.train(fulldataset , iterations=100)
    print ("final training error value = {}".format(sum(error)/ len(error)))
    

    

if __name__ == '__main__':

    question1()
    question2()

