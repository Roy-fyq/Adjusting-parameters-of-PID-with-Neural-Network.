import numpy as np
import matplotlib.pyplot as plt
'''
activation function: tanh, sigmoid
'''
def sigmoid(x):
    return np.exp(x) / (np.exp(x) + np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x))**2
def tanh(x):
    return np.tanh(x)

class NeuralNetwork:
    def __init__(self, inputchannels, hiddenchannels, outputchannels):
        self.inputchannels = inputchannels
        self.outputchannels = outputchannels
        self.hiddenchannels = hiddenchannels
        self.w1 = np.array([[-0.2846, 0.2193, -0.5097, -1.0668],
               [-0.7484, -0.1210, -0.4708, 0.0988],
               [-0.7176, 0.8297, -1.6000, 0.2049],
               [-0.0858, 0.1925, -0.6346, 0.0347],
               [0.4358, 0.2369, -0.4564, -0.1324]]).T

        # wi1, wi2, wi3 = wi, wi, wi

        self.w2 = np.array([[1.0438, 0.5478, 0.8682, 0.1446, 0.1537],
                    [0.1716, 0.5811, 1.1214, 0.5067, 0.7370],
                    [1.0063, 0.7428, 1.0534, 0.7824, 0.6494]]).T
        self.dw2 = np.zeros((hiddenchannels, outputchannels))
        # self.w1 = np.random.randn(inputchannels, hiddenchannels)*0.01
        # self.w2 = np.random.randn(hiddenchannels, outputchannels)*0.01
        self.weights11 = self.weights12 = self.weights13 = self.w1
        self.weights21 = self.weights22 = self.weights23 = self.w2
        self.error0 = 0
        self.error1 = 0
        self.error2 = 0
        self.delta_u = 0
        self.du1 = 0
        self.output = None
        self.error = None
        self.u1, self.u2, self.u3, self.u4, self.u5 = 0, 0, 0, 0, 0
        self.y1, self.y2 = 0, 0
    
    def feedforward(self, rin, y):
        '''
        input: 1D array, shape = (inputchannels,)
        outpu: Kp, Ki, Kd
        input: target, y, error
        '''
        self.y = y
        self.error0 = rin - y
        self.input = np.array([rin, y, self.error0, 1])
        self.error = np.array([self.error0-self.error1, self.error0, self.error0+self.error2-2*self.error1])
        self.I = np.dot(self.input, self.w1)
        self.oh = tanh(self.I)
        self.output = sigmoid(np.dot(self.oh, self.w2))
        self.delta_u = np.sum(self.output * self.error)
        self.u = self.u1 + self.delta_u
        return self.u, self.output
    
    def backprop(self, lr, alpha):
        dout = 2 / (np.exp(self.output) + np.exp(-self.output)) ** 2
        dyu =  np.sign((self.y-self.y1)/(self.delta_u-self.du1+1e-6))
        delta3 = self.error0 * dyu * self.error * dout
        for i in range(self.outputchannels):
            self.dw2[:, i] = lr * delta3[i] * self.oh + alpha * (self.weights21[:, i] - self.weights22[:, i])
        self.w2 = self.weights21 + self.dw2 + alpha * (self.weights21 - self.weights22)
        # dh = 1-self.layer1**2
        do = 4/(np.exp(self.I)+np.exp(-self.I))**2
        seg = np.dot(delta3, self.w2.T)
        de2 = seg*do
        dw1 = lr*np.outer(de2,self.input).T + alpha*(self.weights11-self.weights12)
        # a = np.dot(delta3, self.w2.T)
        # delta2 = dh*a
        # dw1 = np.outer(self.input.T, delta2) * lr
        self.w1 = self.weights11 + dw1 + alpha * (self.weights11 - self.weights12)

        self.u5 = self.u4
        self.u4 = self.u3
        self.u3 = self.u2
        self.u2 = self.u1
        self.u1 = self.u
        self.y2 = self.y1
        self.y1 = self.y
        self.weights23 = self.weights22
        self.weights22 = self.weights21
        self.weights21 = self.w2
        self.weights13 = self.weights12
        self.weights12 = self.weights11
        self.weights11 = self.w1
        self.du1 = self.delta_u
        self.error2 = self.error1
        self.error1 = self.error0
if __name__ == '__main__':
    y = 0
    # target = 15
    # input = np.array([target, y, target-y, 1])
    nn = NeuralNetwork(4, 5, 3)
    # res = [y]
    ts = 0.003
    T = []
    rin_list = []
    y_list = []
    k_list = []
    # pars = []
    y1, y2 = 0, 0
    # y1 = 0
    u1, u2, u3, u4, u5 = 0, 0, 0, 0, 0
    for k in range(1000):
        t = k*ts
        T.append(t)
        rin = np.sin(1*2*np.pi*t)
        rin_list.append(rin)
        a = 1 + 1 + 0.15 * np.sin(k * np.pi / 25)
        y = (a * y1 + u1) / (1 + y1 ** 2)
        y_list.append(y)
        input = np.array([rin, y, rin-y, 1])
        u, par = nn.feedforward(rin, y)
        nn.backprop(3e-2, 5e-2)
        # u = u1 + delta_u
        # res.append(y)
        k_list.append(par)
        u1 = u
        y2 = y1
        y1 = y
    plt.subplot(2, 1, 1)
    plt.plot(rin_list, label='rin')
    plt.plot(y_list, label='y')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title('Temperature')

    # Plotting pars (kp, ki, kd)
    plt.subplot(2, 1, 2)
    plt.plot(k_list, label=['Kp', 'Ki', 'Kd'])
    plt.xlabel('Iteration')
    plt.ylabel('Parameters')
    plt.title('Plot of Parameters (kp, ki, kd)')
    plt.legend()

    # Adjusting subplot spacing
    plt.tight_layout()

    # Display the plot
    plt.show()