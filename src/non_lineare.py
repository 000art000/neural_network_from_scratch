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
        outh = 1 / (1 + np.exp(-input))
        return delta * (outh * (1 - outh))
 
