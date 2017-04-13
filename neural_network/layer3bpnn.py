import math
import random
import string

random.seed(0)


# generate random number in [a, b)
def rand(a, b):
    return (b - a) * random.random() + a


# generate I*J matrix, default zero (can accelerate by Numpy)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# function sigmoid, use tanh, looks better than 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)


# hierarchy function of sigmoid, for get output(y)
def dsigmoid(y):
    return 1.0 - y ** 2


class NN:
    """ 3 layer Backpropagation Neural Network """

    def __init__(self, ni, nh, no):
        # node quantity, input/hidden/output layers
        self.ni = ni + 1  # add an offset node
        self.nh = nh
        self.no = no

        # activate all node (vector)
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weight matrix
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set to random value
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # create momentum factor matrix
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('Not match input layer node quantity!')

        # activate input layer
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # activate hidden layer
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # activate output layer
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):
        ''' BP '''
        if len(targets) != self.no:
            raise ValueError('Not match output layer node quantity!')

        # calculate error of output layer
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error of hidden layer
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output layer weight
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print(N*change, M*self.co[j][k])

        # update input layer weight
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('input layer weight:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('output layer weight:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=100000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 10000 == 0:
                print('error %-.5f' % error)


def demo():
    # demo: train NN to learn logical XOR
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    # Create a NN, input 2 node, hidden 2 node, output 1 node
    n = NN(2, 2, 1)
    n.test(pat)
    # train NN
    n.train(pat)
    # test trained result
    n.test(pat)
    # print trained weight
    n.weights()


if __name__ == '__main__':
    demo()