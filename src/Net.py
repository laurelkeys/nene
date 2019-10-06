import numpy as np
from time import time

from cost_function import CostFunction

class Net:
    def __init__(self, layers, cost_function, optimizer, weight_initialization='xavier', name=""):
        assert(isinstance(cost_function, CostFunction)), "Invalid object type for cost_function"
        
        self.name = name
        
        self.J = cost_function # cost_function(Y, Ypred)
        # obs.: cost_function.derivative is the derivative of J w.r.t. the last layer's activation values [dJ/dYpred]
        #       Ypred == self.layers[-1].A, thus [dJ/dYpred == dJ/dA^L]
        #
        #       cost_function.deltaL is the derivative of J w.r.t. the last layer's Z values [dJ/dZ^L]
        #       Z^L == self.layers[-1].Z, thus [dJ/dZ^L == dJ/dA^L . dA^L/dZ^L == dJ/dYpred . dYpred/dZ^L]
        
        self.optimizer = optimizer # obs.: the learning rate is set on the optimizer object
        
        self.layers = []
        # obs.: we don't call init for the input layer since we set it's activation values manually
        layers[0].input_size = layers[0].output_size
        self.layers.append(layers[0]) # input layer
        for l in range(1, len(layers)):            
            # sets the layer's input_size as the last layer's output_size and initializes its weights and biases
            layers[l].init(input_size=layers[l-1].output_size, weight_initialization=weight_initialization)            
            self.layers.append(layers[l]) # adds the initialized layer to the network
        
        self.optimizer.init(self.layers) # performs initializations that require knowledge about the layers list
        
        self.history = { "loss": [], "loss_val": [],  "acc": [], "acc_val": [], 
                         "best_loss": -1, "best_loss_val": -1,  "best_acc": -1, "best_acc_val": -1, 
                         "lr": self.optimizer.learning_rate }
    
    def __str__(self):
        to_str = self.name + "\n"
        to_str += "[name?] Layer (input_size, output_size) params_count\n"
        to_str += "----------------------------------------------------\n"
        for l in range(len(self.layers)):
            layer = self.layers[l]
            to_str += f"{'' if layer.name == '' else '['+layer.name+'] '}Layer_{l} "
            to_str += f"({layer.input_size}, {layer.output_size}) {layer.params_count}\n"
        return to_str
    
    # note that we use zero-based indexing here, so
    # the 1st layer is self.layers[0] and the last is self.layers[len(self.layers) - 1]
    
    def predict(self, X):
        ''' X.shape == (n_examples, self.layers[0].input_size) '''
        return np.argmax(self.__predict(X), axis=1) # prediction made from the network's one-hot encoded output
    
    def __predict(self, X):
        ''' X.shape == (n_examples, self.layers[0].input_size) 
            Returns the one-hot encoded prediction values
        '''
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        activation = X # network's input
        for l in range(1, len(self.layers)):
            Z = activation @ self.layers[l].W + self.layers[l].b
            activation = self.layers[l].g(Z)
        return activation # network's output (Ypred)
    
    def feedforward(self, X):
        ''' X.shape == (n_examples, self.layers[0].input_size) '''
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        self.layers[0].A = X # input
        for l in range(1, len(self.layers)):
            self.layers[l].feedforward(self.layers[l-1].A)
        Ypred = self.layers[-1].A # output
        return Ypred
    
    def backprop(self, X, Y, Ypred):
        ''' X.shape     == (n_examples, self.layers[0].input_size)
            Y.shape     == (n_examples, self.layers[-1].output_size)
            Ypred.shape == (n_examples, self.layers[-1].output_size)
            where Ypred is the result of feedforward(X)
            
            Note that only calling backprop doesn't actually update the network parameters
        '''
        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y.shape[1] == self.layers[-1].output_size)
        assert(Ypred.shape == Y.shape)
        
        delta = self.J.deltaL(Y, Ypred, self.layers[-1].g) # delta^L == [dJ/dZ^L]
        self.layers[-1].backprop(dZ=delta)
        for l in reversed(range(1, len(self.layers) - 1)):
            # [dJ/dZ^l == dJ/dA^l . dA^l/dZ^l], note that dJ/dA^l is dJ/dX^{l+1}
            delta = self.layers[l+1].dX * self.layers[l].g.derivative(self.layers[l].A) # delta^l == [dJ/dZ^l]
            self.layers[l].backprop(dZ=delta)
        
        # obs.: we don't backpropagate the input layer since we 
        #       manually set it's activation values A to the network's input X
    
    def __shuffle_X_Y(self, X, Y):
        m = X.shape[0] # == Y.shape[0]
        p = np.random.permutation(m)
        return X[p], Y[p]
    
    def __get_batches(self, X, Y, batch_size, shuffled):
        m = X.shape[0] # == Y.shape[0]
        n_batches = m // batch_size
        if shuffled:
            X, Y = self.__shuffle_X_Y(X, Y)
        return zip(np.array_split(X, n_batches), np.array_split(Y, n_batches))
    
    # test data
    def evaluate(self, X_test, Y_test):
        ''' X_test.shape == (n_test_samples, self.layers[0].input_size)
            Y_test.shape == (n_test_samples, self.layers[-1].output_size)
        '''
        assert(X_test.shape[0] == Y_test.shape[0])
        assert(X_test.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y_test.shape[1] == self.layers[-1].output_size)
        
        # loss/cost value for the training set
        Ypred = self.__predict(X_test) # same as self.feedforward(X_test) but doesn't change the 'cached' values
        cost = self.J(Y_test, Ypred)
        
        # calculates the values not as one-hot encoded row vectors
        target = np.argmax(Y_test, axis=1)
        prediction = np.argmax(Ypred, axis=1)
        accuracy = (prediction == target).mean()

        return cost, accuracy
    
    # training and validation data
    def train(self, X, Y, X_val, Y_val, n_epochs, batch_size, verbose=True, save_best=True):
        ''' X.shape == (n_training_samples, self.layers[0].input_size)
            Y.shape == (n_training_samples, self.layers[-1].output_size)
            
            X_val.shape == (n_validation_samples, self.layers[0].input_size)
            Y_val.shape == (n_validation_samples, self.layers[-1].output_size)
            
            For each iteration we'll have:
              n_examples = batch_size
              batch_X.shape == (n_examples, self.layers[0].input_size)
              batch_Y.shape == (n_examples, self.layers[-1].output_size)
            Thus, each epoch has ceil(n_training_samples / batch_size) iterations
            obs.: batch_X and batch_Y are rows of X and Y, and after each iteration (i.e. after going through
                  each batch) we update our network parameters (weights and biases)
            
            If n_training_samples is not divisible by batch_size the last training batch will be smaller
        '''
        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y.shape[1] == self.layers[-1].output_size)
        assert(X_val.shape[0] == Y_val.shape[0])
        assert(X_val.shape[1] == self.layers[0].output_size) # self.layers[0].input_size == self.layers[0].output_size
        assert(Y_val.shape[1] == self.layers[-1].output_size)
        
        n_training_samples = X.shape[0]
        batches_per_epoch = int(np.ceil(n_training_samples / batch_size)) # equal to the number of iterations per epoch
        
        if verbose:
            self.history = { "loss": [], "loss_val": [],  "acc": [], "acc_val": [], 
                             "best_loss": -1, "best_loss_val": -1,  "best_acc": -1, "best_acc_val": -1, 
                             "lr": self.optimizer.learning_rate }
        
        for epoch in range(n_epochs):
            if verbose:
                start_time = time()
                batch_number = 1
                
            for batch_X, batch_Y in self.__get_batches(X, Y, batch_size, shuffled=True):
                # calculates the predicted target values for this batch (with the current network parameters)
                batch_Ypred = self.feedforward(batch_X)
                
                # sets the values of dW and db, used to then update the network parameters
                self.backprop(batch_X, batch_Y, batch_Ypred)
                
                # updates each layer's parameters (i.e. weights and biases) with some flavor of gradient descent
                self.optimizer.update(self.layers)
                
                if verbose:
                    print(f"batch ({batch_number}/{batches_per_epoch})", end='\r')
                    batch_number += 1
            
            # calculate the loss/cost value for this epoch
            epoch_cost, epoch_accuracy = self.evaluate(X, Y) # training set
            epoch_cost_val, epoch_accuracy_val = self.evaluate(X_val, Y_val) # validation set
            self.history["loss"].append(epoch_cost)
            self.history["loss_val"].append(epoch_cost_val)
            self.history["acc"].append(epoch_accuracy)
            self.history["acc_val"].append(epoch_accuracy_val)
            if save_best:
                if epoch_cost < self.history["best_loss"] or self.history["best_loss"] == -1:
                    self.history["best_loss"] = epoch_cost
                if epoch_cost_val < self.history["best_loss_val"] or self.history["best_loss_val"] == -1:
                    self.history["best_loss_val"] = epoch_cost_val
                if epoch_accuracy > self.history["best_acc"] or self.history["best_acc"] == -1:
                    self.history["best_acc"] = epoch_accuracy
                if epoch_accuracy_val > self.history["best_acc_val"] or self.history["best_acc_val"] == -1:
                    self.history["best_acc_val"] = epoch_accuracy_val
            if verbose:
                print(f"epoch ({epoch+1}/{n_epochs}) "
                      f"loss: {epoch_cost:.4f}, loss_val: {epoch_cost_val:.4f} | "
                      f"acc: {epoch_accuracy:.4f}, acc_val: {epoch_accuracy_val:.4f} | "
                      f"Δt: {(time() - start_time):.2f}s")