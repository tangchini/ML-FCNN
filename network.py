from layer import *
import time
import numpy as np

def _pred_convert(preds, test_y):
    ans_pred = []
    ans_test_y = []
    for i in range(len(preds)):
        ans_pred.append(np.argmax(preds[i]))
        ans_test_y.append(np.argmax(test_y[i]))
    return np.array(ans_pred), np.array(ans_test_y)

def accuracy(preds, test_y):
    pred, actual = _pred_convert(preds, test_y)
    hit = 0
    for i,j in zip(pred, actual):
        if i == j:
            hit+=1
    return hit/len(pred)

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None

    def add(self, layer):
        self.layers.append(layer)
        
    def printitss(self,a):
        print(a)
        
    def use(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            output = self.forward(output)
            result.append(output)
        return result
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)
        return error

    def fit(self, x_train, y_train, epochs, learning_rate, x_test=None, y_test=None):
        samples = len(x_train)
        
        
        for i in range(epochs):
            st = time.time()
            for j in range(samples):
                output = self.forward(x_train[j])               
                self.backward(self.loss_deriv(y_train[j], output) , learning_rate)
    
            train_err = 0   
            for j in range(samples):
                output = self.forward(x_train[j])
                train_err += self.loss(y_train[j], output)
            train_err /= samples
            train_acc = accuracy(self.predict(x_train), y_train)
            
            try:
                samples_test = len(x_test)
                test_err = 0   
                for j in range(samples_test):
                    output = self.forward(x_test[j])
                    test_err += self.loss(y_test[j], output) 
                test_err /= samples_test
                test_acc = accuracy(self.predict(x_test), y_test)
                print('epoch %d/%d  train loss=%f  test loss=%f  train accuracy=%f  test accuracy=%f  time:%f'% (i+1,epochs,train_err,test_err,train_acc,test_acc,time.time()-st))
             
            except:
                print('epoch %d/%d  train loss=%f  train accuracy=%f  time:%f'% (i+1,epochs,train_err,train_acc,time.time()-st))
              
                

        
        
