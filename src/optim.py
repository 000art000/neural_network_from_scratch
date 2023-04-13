
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