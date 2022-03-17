import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad, learning_rate):
        raise NotImplementedError
        
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weights = np.random.rand(in_features, out_features) - 0.5
        self.bias = np.random.rand(1, out_features) - 0.5

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_grad, learning_rate):
        input_error = np.dot(output_grad, self.weights.T)
        weights_error = np.dot(self.input.T, output_grad)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_grad
        return input_error

## by yourself .Finish your own NN framework
class ACTIVITY1(_Layer):
    def __init__(self, activation, activation_deriv):
        self.activation = activation
        self.activation_deriv = activation_deriv

    def forward(self, input): 
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_grad, learning_rate):
        return self.activation_deriv(self.input) * output_grad
    
    
    