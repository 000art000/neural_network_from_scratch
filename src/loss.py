from body import Loss
import numpy as np

class MSELoss (Loss):

    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return (y-yhat)**2
    
    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -2*(y-yhat)
    

class BCELoss(Loss):
    
    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        yhat+=1e-9
        yhat/=yhat.sum(axis=1)
        return -(y*np.log(yhat)+(1-y)*np.log(1-yhat))

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -y/yhat +(1-y)/(1-yhat)

class CELoss(Loss):
    
    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return (-y*yhat).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -y
    
class LOGCELoss(Loss):
    
    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        yhat+=1e-9
        yhat/=yhat.sum(axis=1)
        return (-y*np.log(yhat)).sum(axis=1)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        yhat+=1e-9
        yhat/=yhat.sum(axis=1)
        return -y/yhat
    