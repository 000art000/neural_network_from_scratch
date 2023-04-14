from sklearn.model_selection import train_test_split
import numpy as np

class Optim:

    def __init__(self,net,loss,eps):
        self._net=net
        self._loss=loss
        self._eps=eps

    def step(self,batch_x,batch_y):

        y_hat=self.net.forward(batch_x)
        loss_back=self._loss.backward(batch_y,y_hat)
        self.net.backward(loss_back)
        self.net.update_parameters(self.eps)

        return self._loss.forward(batch_y,y_hat)
    
    def SGD(self,batch_x,batch_y,batch_size=0.8,iter=1000):
        
        d_y=batch_y.shape[0]
        assert batch_x.shape[0]==d_y

        XY=np.concatenate((batch_x,batch_y),axis=1)
        np.random.shuffle(XY)
        train_test_split(XY,)
