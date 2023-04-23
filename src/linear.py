from body import Module
import numpy as np

from tools import params

class Linear (Module) :

    def __init__(self,input,output,type=0):
        
        # d d'
        self._input=input
        self._output=output

        # param
        self._parameters =params(input,output,type)
        self._biais = params(1,output,type)
        
        # gradient
        self._gradient_biais=np.zeros((1,output))
        self._gradient_param = np.zeros((input, output))
    
    def zero_grad(self):
        ## Annule gradient
        self._gradient_biais =np.zeros((1, self._output))
        self._gradient_param =np.zeros((self._input, self._output))

    def forward(self, X):
        """ X:(batch,d)->(batch,d') """

        assert X.shape[1]==self._input
        return X@self._parameters + self._biais

    def update_parameters(self, gradient_step=1e-3):
        
        self._parameters -= gradient_step * self._gradient_param
        self._biais -= gradient_step * self._gradient_biais

    def backward_update_gradient(self, input, delta):
        #met a jour les gradients

        """
            input (batch,input)
            delta (batch,output)
            derive de cout sur les param de notre module (w_h,b_h)
        """

        assert input.shape[1] == self._input
        assert delta.shape[1] == self._output
        assert input.shape[0] == delta.shape[0]

        self._gradient_param += input.T @ delta
        self._gradient_biais += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur

        """ input (batch,input)
            delta (batch,output)
            derive de cout sur nous entrer (z_h-1 entrer)
        """
        
        assert input.shape[1] == self._input
        assert delta.shape[1] == self._output

        return delta @ self._parameters.T