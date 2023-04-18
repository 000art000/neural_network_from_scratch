from body import Module
import numpy as np


class TanH (Module) :
    
    def forward(self, X):
        """ X:(batch,d)->(batch,d) """
        return np.tanh(X)

    def backward_delta(self, input, delta):
        return  delta*(1-np.tanh(input)**2)

class Sigmoide (Module):
    
    def forward(self, X):
        """ X:(batch,d)->(batch,d) """
        return 1/(1+np.exp(-X))
    
    def backward_delta(self, input, delta):
        outh = 1 / (1 + np.exp(-input))
        return delta * (outh * (1 - outh))
    
class Softmax(Module):  
 
    def forward(self,X):
        """ X:(batch,d)->(batch,d) """
        exp=np.exp(X)
        return exp/np.sum(exp,axis=1).reshape((-1,1))
    
    def backward_delta(self,X,delta):
        sof=self.forward(X)
        return sof * (1-sof) * delta  

