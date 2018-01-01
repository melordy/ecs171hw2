
# coding: utf-8

# In[3]:



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
    return (math.tanh(x) + 1)/2


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return (1.0 - math.tanh(y) ** 2)/2


class NN:
    weights_list = []
    bias_list = []
    error_list = []

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
        self.bi = [ 0.26411771, -0.70927067,  0.59113199]
        self.bo = [-0.45816304]

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = 0.1
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = 0.1

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):

        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni -1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]
        print ("inputs", self.ai[:8])
        print ("input_to_hidden weights", self.wi)
        print ("hidden bias", self.bi)

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni - 1):
                # sum = sum + self.ai[i] * self.swi[i][j]
                tempSum = self.ai[i] * self.wi[i][j]
                sum = sum + tempSum
                # print ("indi input * weight + bias", tempSum)

            self.ah[j] = sigmoid(sum)

        print ("hidden", self.ah)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                # sum = sum + self.ah[j] * self.wo[j][k]
                tempSum = ( (self.ah[j] * self.wo[j][k] ) + self.bo[k])
                sum = sum + tempSum
                print ("input * weight + bias", tempSum)
            print ("sum of input * weight + bias", sum)
            self.ao[k] = sigmoid(sum)
        print ("hidden_to_output", self.wo)
        print ("output bias", self.bo)
        print ("outputs", self.ao)
        return self.ao[:]

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            print ("error(actual - predicted)", error)
            output_deltas[k] = dsigmoid(self.ao[k]) * error
            print ("output_delta = ((a - tanh(predicted_out)^2)/2) * error", output_deltas[k])
        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
            print ("hidden_deltas = ((a - tanh(sum(output_delta*hidden_to_output_weights))^2)/2) * error", hidden_deltas[j])
        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change
                print ("hidden_to_output_weights_update = hidden_to_output_weights + learning_rate*output_delta*hidden", self.wo[j][k])
                self.co[j][k] = change
                # print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change
                print ("input_to_hidden_weights_update = input_to_hidden_weights + learning_rate*hidden_deltas*input",
                       self.wi[j][k])
                self.co[j][k] = change
                self.ci[i][j] = change

        # update output biases
        for k in range(self.no):
            change = output_deltas[k]
            self.bo[k] = self.bo[k] + N * change
            print ("output_biases_update = output_biases_weights + learning_rate*output_deltas",
                   self.bo[k])
            # print N*change, M*self.co[j][k]

        # update hidden layer biases
        for k in range(self.nh):
            change = hidden_deltas[k]
            self.bi[k] = self.bi[k] + N * change
            print ("input_biases_update = input_biases_weights + learning_rate*hidden_deltas",
                   self.bi[k])


        if targets[0] == 0.0:
            self.weights_list.append([i[0] for i in self.wo])
            self.bias_list.append([i for i in self.bo][0])

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + (0.5 * (targets[k] - self.ao[k]) ** 2)
        print ("total mean square error", error)

        return error

    def test(self, patterns):
        self.error_list = []
        predicted = []
        for p in patterns:
            inputs = p[0]
            # print ("inputs", inputs)
            targets = p[1]
            # print ("outputs", targets)
            self.update(inputs)
            predicted.append(self.ao)
        # print ("predicted", predicted)
        # sys.exit()

            error = 0.0
            print ("target length", len(targets))
            for k in range(len(targets)):
                error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
            self.error_list.append(error)
        return sum(self.error_list)/len(self.error_list)

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=100, N=0.2, M=0):
        self.error_list = []
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            self.weights_list = []
            self.bias_list = []
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                one_error = self.backPropagate(targets, N, M)
                error = error + one_error
            #     print ("individual_error", one_error)
            # print ("train_error", error)
            self.error_list.append(error)

        return self.error_list

    def plotGraph_weight_bases(self, weight_list = [], biases_list = []):
        plt.xlabel('cyt_number')
        plt.ylabel('Value')

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
        x_axis, biases_list, 'r^', label = "biases" )

        plt.plot(
            x_axis, ho_weights_1, 'g--', label = "weight 1"  )

        plt.plot(
            x_axis, ho_weights_2, 'g^', label = "weight 2"  )

        plt.plot(
            x_axis, ho_weights_3, 'bs', label="weight 3")



        plt.axis([x1, x2, y1, y2])

        plt.legend()
        plt.show()

    def plotGraph_error(self, train_error_list = [], test_error_list = []):
        plt.xlabel('cyt_number')
        plt.ylabel('Value')

        # if train_error_list:
        #     train_error_list = [i * abs(1 / max(train_error_list)) for i in train_error_list]
        # if test_error_list:
        #     test_error_list = [i * abs(1 / max(test_error_list)) for i in test_error_list]

        x1 = 0
        x2 = len(train_error_list)
        y1 = min(train_error_list)
        y2 = max(train_error_list)

        x_axis = [i for i in range(len(train_error_list))]

        if train_error_list:
            plt.plot(
                x_axis, train_error_list, label="train_error")
        if test_error_list:
            plt.plot(
                x_axis, test_error_list, label="test_error")

        plt.axis([x1, x2, y1, y2])

        plt.legend()
        plt.show()


def question3():
    dataset = loadFile("data.csv")

    train_dataset = dataset[:1040]

    # create a network with two input, two hidden, and one output nodes
    n = NN(8, 3, 1)


    for dataset in train_dataset:
        n.train([dataset], iterations=1)
        sys.exit()

if __name__ == '__main__':

    question3()




