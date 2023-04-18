from sklearn.model_selection import train_test_split
import numpy as np

class Optim:

    def __init__(self,net ,loss,eps):
        # net : Sequentiel 
        # loss : Loss
        self._net=net
        self._loss=loss
        self._eps=eps

    def step(self,batch_x,batch_y):

        y_hat=self._net.forward(batch_x)
        delta=self._loss.backward(batch_y,y_hat)
        self._net.backward(delta)
        self._net.update_parameters(self._eps)

        return self._loss.forward(batch_y,y_hat)
    
    def SGD(self,X,Y,batch_size,epochs=20,shuffle=False):

        n=X.shape[0]

        assert Y.shape[0]==n
        
        #nombre de bloc
        nb_bloc=n//batch_size

        if nb_bloc==0 :
            nb_bloc=1

        indexs=np.arange(n)

        if shuffle :
            np.random.shuffle(indexs)

        #deviser les indexe en nb_bloc
        indexs=np.array_split(indexs,nb_bloc)

        #creer les bloc de données
        Xs,Ys=[],[]
        for ind in indexs:
            Xs.append(X[ind])
            Ys.append(Y[ind])
            
        list_loss=[]

        for _ in range(epochs) :

            for batch_x,batch_y in zip(Xs,Ys) :
                loss=self.step(batch_x,batch_y)

            list_loss.append(loss.mean())
    
        return list_loss

    def accuracy(self,x,y):
        return np.where(y == self._net.predict(x),1,0).mean()