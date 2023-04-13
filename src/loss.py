from body import Loss
import numpy as np

class MSELoss (Loss):

    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return (y-yhat)**2
    
    def backward(self, y, yhat):
        return -2*(y-yhat)
    
