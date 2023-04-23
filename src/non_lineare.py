from body import Module
import numpy as np


class TanH (Module) :
    
    def forward(self, X):
        """ X:(batch,d)->(batch,d) """
        return np.tanh(X)

    def backward_delta(self, input, delta):
        return  (1-np.tanh(input)**2)* delta

class Sigmoide (Module):
    
    def forward(self, X):
        """ X:(batch,d)->(batch,d) """
        return 1/(1+np.exp(-X))
    
    def backward_delta(self, input, delta):
        outh = self.forward(input)
        return delta * (outh * (1 - outh))
    
class Softmax(Module):  
 
    def forward(self, X):
        #pass forward 
        e = np.exp(X)
        return e / np.sum(e, axis=1).reshape((-1, 1))

    def backward_delta(self, input, delta):
        #backward, pour la propagation
        e = np.exp(input)
        outh = e/ np.sum(e, axis=1).reshape((-1, 1))
        return delta * (outh * (1 - outh))
    
    
class ReLU(Module):
    def __init__(self,threshold=0.):
        self._threshold=threshold

    def forward(self, X):
        self._forward=self.threshold(X)
        return self._forward

    def threshold(self,input):
        return np.where(input>self._threshold,input,0.)


    def derivative_Threshold(self,input):
        #Batch x out
        return (input > self._threshold).astype(float)

    def backward_delta(self, input, delta):
        self._delta=np.multiply(delta,self.derivative_Threshold(input))
        return self._delta

