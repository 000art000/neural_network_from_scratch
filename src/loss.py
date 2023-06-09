from body import Loss
import numpy as np

class MSELoss (Loss):

    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return np.sum(y-yhat,axis=1)**2
    
    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -2*(y-yhat)

############################################################################################################ multi

class CELoss(Loss):
    
    def forward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert(y.shape == yhat.shape)
        return 1 - np.sum(yhat * y, axis = 1)
    
    def backward(self, y, yhat):
        #params passé en entrée sont de la bonne taille 
        assert(y.shape == yhat.shape)  
        return yhat-y
    
class LOGSOFTCELoss(Loss):
    def forward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert(y.shape == yhat.shape)
        return np.log(np.sum(np.exp(yhat), axis=1)) - np.sum(y * yhat,axis = 1)

    def backward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert(y.shape == yhat.shape)
        e = np.exp(yhat)
        return e / np.sum(e, axis=1).reshape((-1,1)) - y
    
############################################################################################################# encoder

class BCELoss(Loss):
    

    def forward(self, y, yhat):
        """
        y -> One hot encoding
        """
        return - (y*np.log(yhat + 1e-100) + (1-y)*np.log(1-yhat+ 1e-100))

    
    def backward(self, y, yhat):
        """
        y -> One hot encoding
        """
        
        return ((1-y)/(1-yhat+ 1e-100)) - (y/yhat+ 1e-100)