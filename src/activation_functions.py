import numpy as np

class ActivationFunction:
    ''' An ActivationFunction is applied to Z to get the output A, 
        but its derivative expects the value A, not Z (!):

        A == __call__(Z) and derivative(A) == derivative(__call__(Z)), 
        calling derivative(Z) will often yield WRONG results
    '''
    def __call__(self, Z):
        ''' Z.shape=(n_examples, layer_output_size) '''
        raise NotImplementedError    
    def derivative(self, A):
        ''' A.shape=(n_examples, layer_output_size) '''
        raise NotImplementedError

class Linear(ActivationFunction):
    def __call__(self, Z):
        return Z
    def derivative(self, A):
        return np.ones_like(A)

class ReLU(ActivationFunction):
    def __call__(self, Z):
        return np.maximum(0, Z)
    def derivative(self, A):
        return np.where(A > 0, 1, 0)

class Sigmoid(ActivationFunction):
    def __call__(self, Z, clip=500):
        Z = np.clip(Z, -clip, clip) # numerical stability
        return 1 / (1 + np.exp(-Z))
    def derivative(self, A):
        return A * (1 - A) # Sigmoid(Z) * (1 - Sigmoid(Z))

class SoftMax(ActivationFunction):
    def __call__(self, Z):
        exp = np.exp(Z - Z.max(axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    def derivative(self, A):
        return A * (1 - A) # SoftMax(Z) * (1 - SoftMax(Z)) FIXME
        