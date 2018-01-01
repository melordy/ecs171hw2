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


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y ** 2


class NN:
    cyt_weights_list = []
    cyt_bias_list = []
    cyt_train_error_list = []
    cyt_test_error_list = []

    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules

        # create biases
        self.bi = 2 * np.random.random((self.nh)) - 1
        self.bo = 2 * np.random.random((self.no)) - 1

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):

        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                # sum = sum + self.ai[i] * self.swi[i][j]
                sum = sum + ((self.ai[i] * self.wi[i][j]))

            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                # sum = sum + self.ah[j] * self.wo[j][k]
                sum = sum + ( (self.ah[j] * self.wo[j][k] ))
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # update output biases
        for k in range(self.no):
            change = output_deltas[k]
            self.bo[k] = self.bo[k] + N * change
            # print N*change, M*self.co[j][k]

        # update hidden layer biases
        for k in range(self.nh):
            change = hidden_deltas[k]
            self.bi[k] = self.bi[k] + N * change


        if targets[0] == 0.0:
            self.cyt_weights_list.append([i[0] for i in self.wo])
            self.cyt_bias_list.append([i for i in self.bo][0])

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2

        return error

    def test(self, patterns):
        self.cyt_test_error_list = []

        self.error_list = []
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            self.update(inputs)

            error = 0.0
            for k in range(len(targets)):
                error_change =  0.5 * (targets[k] - self.ao[k]) ** 2
                error = error + error_change
            self.error_list.append(error)

            if targets[0] == 0.0:
                self.cyt_test_error_list.append(error_change )
        return self.error_list

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=100, N=0.5, M=0):
        self.error_list = []

        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            self.cyt_train_error_list = []
            self.cyt_weights_list = []
            self.cyt_bias_list = []
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error_change = self.backPropagate(targets, N, M)
                error = error + error_change

                if targets[0] == 0.0:
                    self.cyt_train_error_list.append(error_change)

            self.error_list.append(error)

        return self.error_list



    def predict(self, patterns):
        self.error_list = []
        for p in patterns:
            inputs = p[0]
            output = self.update(inputs)
            print (p)
            print (output)



    def plotGraph_cyt_weight_bases(self, weight_list = [], biases_list = [], title = ""):
        plt.xlabel('cyt_number')
        plt.ylabel('Value')
        plt.title(title)

        ho_weights_1 = []
        ho_weights_2 = []
        ho_weights_3 = []

        for ho_weight in weight_list:
            ho_weights_1.append(ho_weight[0])
            ho_weights_2.append(ho_weight[1])
            ho_weights_3.append((ho_weight[2]))

        ho_weights_1 = [i * abs(1 / max(ho_weights_1)) for i in ho_weights_1]
        ho_weights_2 = [i * abs(1 / max(ho_weights_2)) for i in ho_weights_2]
        ho_weights_3 = [i * abs(1 / max(ho_weights_3)) for i in ho_weights_3]

        x1 = 0
        x2 = len(ho_weights_1)
        y1 = 1
        y2 = -1

        x_axis = [i for i in range(len(ho_weights_1))]


        plt.plot(
        x_axis, biases_list, label = "biases" )

        plt.plot(
            x_axis, ho_weights_1, label = "weight 1"  )

        plt.plot(
            x_axis, ho_weights_2, label = "weight 2"  )

        plt.plot(
            x_axis, ho_weights_3, label="weight 3")



        plt.axis([x1, x2, y1, y2])

        plt.legend()
        plt.show()





    def plotGraph_error(self, train_error_list = [], test_error_list = [], title = ""):
        plt.xlabel('cyt_number')
        plt.ylabel('Error Value')

        plt.title(title)

        x1 = 0
        x2 = len(train_error_list)
        y1 = min(train_error_list)
        y2 = max(train_error_list)


        if train_error_list:
            plt.plot(
                [i for i in range(len(train_error_list))], train_error_list, label="train_error")
        if test_error_list:
            plt.plot(
                [i for i in range(len(test_error_list))], test_error_list, label="test_error")

        plt.axis([x1, x2, y1, y2])

        plt.legend()
        plt.show()


