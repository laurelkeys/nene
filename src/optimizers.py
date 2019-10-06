import numpy as np

class Optimizer:
    ''' The optimizer's optimization policy should be implemented on its update(layers) function '''
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self, layers):
        ''' Updates the parameters (i.e. weights and biases) for each layer in the layers list '''
        raise NotImplementedError
    def init(self, layers):
        ''' Performs initializations that require knowledge about the layers list '''
        pass

class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
    def update(self, layers):
        for layer in layers[1:]:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db

class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.last_dW = None # stores the last value of dW for each layer
        self.last_db = None # stores the last value of db for each layer
    def update(self, layers):
        for l in range(1, len(layers)):
            layer = layers[l]
            self.last_dW[l] = (self.momentum * self.last_dW[l]) + (self.learning_rate * layer.dW)
            self.last_db[l] = (self.momentum * self.last_db[l]) + (self.learning_rate * layer.db)
            layer.W -= self.last_dW[l]
            layer.b -= self.last_db[l]
    def init(self, layers):
        L = len(layers)
        self.last_dW = [0 for _ in range(L)]
        self.last_db = [0 for _ in range(L)]

class Adam(Optimizer):
    ''' Adaptive Moment Estimation '''
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, correct_bias=False):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_dW = self.m_db = self.v_dW = self.v_db = None
        self.correct_bias = correct_bias
        self.t = 0
    def update(self, layers):
        if self.correct_bias:
            self.__update_w_bias_correct(layers)
        else:
            self.__update_w_o_bias_correct(layers)
    def __update_w_o_bias_correct(self, layers):
        for l in range(1, len(layers)):
            layer = layers[l]
            beta1 = self.beta1; one_minus_beta1 = 1 - beta1
            beta2 = self.beta2; one_minus_beta2 = 1 - beta2
            dW = layer.dW; db = layer.db
            self.m_dW[l] = (beta1 * self.m_dW[l]) + (one_minus_beta1 * dW)
            self.m_db[l] = (beta1 * self.m_db[l]) + (one_minus_beta1 * db)
            self.v_dW[l] = (beta2 * self.v_dW[l]) + (one_minus_beta2 * dW*dW)
            self.v_db[l] = (beta2 * self.v_db[l]) + (one_minus_beta2 * db*db)
            layer.W -= (self.learning_rate * self.m_dW[l]) / (np.sqrt(self.v_dW[l]) + self.eps)
            layer.b -= (self.learning_rate * self.m_db[l]) / (np.sqrt(self.v_db[l]) + self.eps)
    def __update_w_bias_correct(self, layers):
        self.t += 1
        for l in range(1, len(layers)):
            layer = layers[l]
            beta1 = self.beta1; one_minus_beta1 = 1 - beta1
            beta2 = self.beta2; one_minus_beta2 = 1 - beta2
            dW = layer.dW; db = layer.db
            self.m_dW[l] = (beta1 * self.m_dW[l]) + (one_minus_beta1 * dW)
            self.m_db[l] = (beta1 * self.m_db[l]) + (one_minus_beta1 * db)
            self.v_dW[l] = (beta2 * self.v_dW[l]) + (one_minus_beta2 * dW*dW)
            self.v_db[l] = (beta2 * self.v_db[l]) + (one_minus_beta2 * db*db)
            m_dW_hat = self.m_dW[l] / (1 - beta1**self.t)
            v_dW_hat = self.v_dW[l] / (1 - beta2**self.t)
            m_db_hat = self.m_db[l] / (1 - beta1**self.t)
            v_db_hat = self.v_db[l] / (1 - beta2**self.t)
            layer.W -= (self.learning_rate * m_dW_hat) / (np.sqrt(v_dW_hat) + self.eps)
            layer.b -= (self.learning_rate * m_db_hat) / (np.sqrt(v_db_hat) + self.eps)
    def init(self, layers):
        L = len(layers)
        self.m_dW = [0 for _ in range(L)]
        self.m_db = [0 for _ in range(L)]
        self.v_dW = [0 for _ in range(L)]
        self.v_db = [0 for _ in range(L)]