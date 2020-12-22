import numpy as np
import wave,json,os

steps = [
    'model',
    'activation',
    'deriv',
    'partial',
    'bptt',
    'gradient',
    'update'
]

class Dataset(object):
    """
    dataset=Dataset(epoch): epoch can be either:
      -1: instructs this program to generate and use knowledge-based LSTM weights
      0: instructs this program to generate and use pseudo-random LSTM weights
      1: instructs this program to load and use LSTM weights that have been pre-trained for 1 epoch
    ...
      100: instructs this program to load and use LSTM weights that have been pre-trained for 100 epochs

    The following are loaded from the data directory:
      dataset.train[i] = i'th training example, in the form of a numpy array
      dataset.ntrain = len(dataset.train)
    """
    def __init__(self,epoch):
        self.epoch = epoch
        # Load the training data: observation, label
        self.observation=np.genfromtxt('data/observation_train_30000.txt',dtype=np.int16)
        self.label=np.genfromtxt('data/label_train_30000.txt',dtype=np.int16)
        self.nsamps=len(self.observation)

    
    # Create the model:
    #   If epoch==-1, create knowledge-based weights that will perform the task perfectly.
    #   If epoch==0, create pseudo-random initial weights (with random.seed(0))
    #   If epoch==1, load pre-trained weights
    #
    # Task definition:
    #   For every time step t,
    #   If self.observation[t]==0, then output h[t]=0.
    #   If self.observation[t]>=1,
    #     and the preceding nonzero sample was observation[s], then output h[t]=t-s.
    #     If observation[t] is the very first nonzero sample, then output h[t]=t.
    #
    # self.model[0,:] == (bc,wc,uc) -- bias, inputweight, historyweight for the cell
    # self.model[1,:] == (bi,wi,ui) -- bias, inputweight, historyweight for input gate
    # self.model[2,:] == (bf,wf,uf) -- bias, inputweight, historyweight for forget gate
    # self.model[3,:] == (bo,wo,uo) -- bias, inputweight, historyweifht for output gate
    #
    # In order to set the "knowledge-based weights", you need to think about how an LSTM works.
    # In the following equations, * means multiplication, [] means index, () means function.
    # x[t] -- input
    # c[t] = f[t]*c[t-1] + i[t]*g(wc*x[t]+uc*h[t-1]+bc) -- cell
    # i[t] = g(wi*x[t]+ui*h[t-1]+bi) -- input gate
    # f[t] = g(wf*x[t]+uf*h[t-1]+bf) -- forget gate
    # o[t] = g(wo*x[t]+uo*h[t-1]+bo) -- output gate
    # h[t] = o[t]*c[t] -- prediction, a.k.a. hidden state, a.k.a. output
    #
    # For epoch==-1 (knowledge-based design), use the CReLU activation function:
    #        g(x) = max(0,min(1,x))  and limit the weights to -1 <= self.model[i,j] <= 1.
    # For epoch >= 0 (gradient descent), use the logistic activation function:
    #        g(x) = 1/(1+exp(-x)), and the weight values are not limited.
    def set_model(self):
        self.model = np.zeros((4,3))
        if self.epoch>0:
            # Try to load from the solutions model if it exists
            try:
                with open('solutions/model%2.2d.json'%(self.epoch)) as f:
                    self.model = np.array(json.load(f))
            # If that fails, try loading from the models directory
            except:
                with open('models/model%2.2d.json'%(self.epoch)) as f:
                    self.model = np.array(json.load(f))
        elif self.epoch==0:
            np.random.seed(0)
            self.model = np.random.normal(size=(4,3))
        elif self.epoch==-1:
            # TODO: set the weights to knowledge-based values that will perform the assigned task.
            self.model = np.zeros((4,3)) # INCORRECT!
            self.model[0,:] = (1,0,0)
            self.model[1,:] = (1,-1,0)
            self.model[2,:] = (1,0,-1)
            self.model[3,:] = (0,1,0)
            
    # Utility function: this function, self.activate(x),
    # is provided for you, in case you find it useful.
    def activate(self,x):
        if self.epoch!=-1:
            return 1/(1+np.exp(-x))
        return np.maximum(0,np.minimum(1,x))

 
    #
    # Calculate the values of the activations for every nonlinearity, in every sample
    # self.activation[t,0] == c[t], activation of the cell at t'th sample
    # self.activation[t,1] == i[t], activation of the input gate at t'th sample
    # self.activation[t,2] == f[t], activation of the forget gate at t'th sample
    # self.activation[t,3] == o[t], activation of the output gate at t'th sample
    # self.activation[t,4] == h[t], activation of the output gate at t'th sample
    #initialize c[-1]=0
    # c[t] = f[t]*c[t-1] + i[t]*g(wc*x[t]+uc*h[t-1]+bc) -- cell
    # i[t] = g(wi*x[t]+ui*h[t-1]+bi) -- input gate
    # f[t] = g(wf*x[t]+uf*h[t-1]+bf) -- forget gate
    # o[t] = g(wo*x[t]+uo*h[t-1]+bo) -- output gate
    # h[t] = o[t]*c[t] -- prediction, a.k.a. hidden state, a.k.a. output
    def set_activation(self):
        self.activation = np.zeros((self.nsamps,5))
#         c_m1 = 0
#         h_m1 = 0
        
        self.activation[0,1] = self.activate(self.model[1,0] + self.model[1,1]*self.observation[0]) #i
        self.activation[0,2] = self.activate(self.model[2,0] + self.model[2,1]*self.observation[0]) #f
        self.activation[0,3] = self.activate(self.model[3,0] + self.model[3,1]*self.observation[0]) #o
        self.activation[0,0] = self.activation[0,1]*self.activate(self.model[0,0]+self.model[0,1]*self.observation[0]) #c
        self.activation[0,4] = self.activation[0,0]*self.activation[0,3] #h
        for t in range(1,self.nsamps):
            self.activation[t,1] = self.activate(self.model[1,0] + self.model[1,1]*self.observation[t] + self.model[1,2]*self.activation[t-1,4])
            self.activation[t,2] = self.activate(self.model[2,0] + self.model[2,1]*self.observation[t] + self.model[2,2]*self.activation[t-1,4])
            self.activation[t,3] = self.activate(self.model[3,0] + self.model[3,1]*self.observation[t] + self.model[3,2]*self.activation[t-1,4])
            self.activation[t,0] = self.activation[t,2]*self.activation[t-1,0] + self.activation[t,1]*self.activate(self.model[0,0] + self.model[0,1]*self.observation[t] + self.model[0,2]*self.activation[t-1,4])
            self.activation[t,4] = self.activation[t,0]*self.activation[t,3]
        # TODO: calculate the activations
        

    # Utility function: this function, self.derivative(x), 
    # is provided for you, in case you find it useful.
    def derivative(self,x):
        if self.epoch!=-1:   # derivative of the logistic function
            return x*(1-x)  
        else:                # derivative of the CReLU
            return np.array([ 1 if (0<xt and xt<1) else 0 for xt in x])
#     def derivative(self,x):
#         if self.epoch!=-1:   # derivative of the logistic function
#             return x*(1-x)  
#         else:
#             # derivative of the CReLU
#             if (x > 0 and x < 1):
#                 return 1
#             else:
#                 return 0 
    
    
    
  
    #
    # Calculate derivative of the activation functions at each time step.
    # self.deriv[t,0] == dg(z)/dz evaluated at z=wc*x[t]+uc*h[t-1]+bc -- cell derivative
    # self.deriv[t,1] == dg(z)/dz evaluated at z=wi*x[t]+ui*h[t-1]+bi -- input gate derivative
    # self.deriv[t,2] == dg(z)/dz evaluated at z=wf*x[t]+uf*h[t-1]+bf -- forget gate derivative
    # self.deriv[t,3] == dg(z)/dz evaluated at z=wo*x[t]+uo*h[t-1]+bo -- output gate derivative
    def set_deriv(self):
        
        self.deriv = np.zeros((self.nsamps,4))
        self.deriv[0,0] = self.derivative(np.array([self.activate(self.model[0,0] + self.model[0,1]*self.observation[0] )]))
        self.deriv[0,1] = self.derivative(np.array([self.activate(self.model[1,0] + self.model[1,1]*self.observation[0] )]))
        self.deriv[0,2] = self.derivative(np.array([self.activate(self.model[2,0] + self.model[2,1]*self.observation[0] )]))
        self.deriv[0,3] = self.derivative(np.array([self.activate(self.model[3,0] + self.model[3,1]*self.observation[0] )]))
        
        for t in range(1,self.nsamps):
            self.deriv[t,0] = self.derivative(np.array([self.activate(self.model[0,0] + self.model[0,1]*self.observation[t] + self.model[0,2]*self.activation[t-1,4])]))
            self.deriv[t,1] = self.derivative(np.array([self.activate(self.model[1,0] + self.model[1,1]*self.observation[t] + self.model[1,2]*self.activation[t-1,4])]))
            self.deriv[t,2] = self.derivative(np.array([self.activate(self.model[2,0] + self.model[2,1]*self.observation[t] + self.model[2,2]*self.activation[t-1,4])]))
            self.deriv[t,3] = self.derivative(np.array([self.activate(self.model[3,0] + self.model[3,1]*self.observation[t] + self.model[3,2]*self.activation[t-1,4])]))
            
            

        # TODO: calculate the derivative

  
    #
    # Calculate the partial derivative of the error with respect to h[t],
    # Define E = (1/2)*np.average(np.square(self.activation[:,4]-self.label)).  Then
    # self.partial[t] == partial E/partial dh[t], where h[t]=self.activation[t,4]),
    # where "partial" means dE/dh[t] under the assumption that all other variables are constant
    def set_partial(self):
        self.partial = np.zeros((self.nsamps))
        for t in range(self.nsamps):
            self.partial[t] = (1/self.nsamps)*(self.activation[t,4]-self.label[t])
        # TODO: calculate the partial deriviate

  
    #
    # Back-prop through time.
    # Accumulate dE/dexcitation backward through time, via links from h[n+1] and c[n+1].
    # self.bptt[n,0] == dE/dc[t]
    # self.bptt[n,1] == dE/di[t]
    # self.bptt[n,2] == dE/df[t]
    # self.bptt[n,3] == dE/do[t]
    # self.bptt[n,4] == dE/dh[t]
    def set_bptt(self):
        self.bptt = np.zeros((self.nsamps,5))
        self.bptt[self.nsamps-1,4] = self.partial[self.nsamps-1]
        self.bptt[self.nsamps-1,0] = self.bptt[self.nsamps-1,4]*self.activation[self.nsamps-1,3]
        self.bptt[self.nsamps-1,1] = self.bptt[self.nsamps-1,0]*self.activate(self.model[0,0] + self.model[0,1]*self.observation[self.nsamps-1] + self.model[0,2]*self.activation[self.nsamps-2,4])
        self.bptt[self.nsamps-1,3] = self.bptt[self.nsamps-1,4]*self.activation[self.nsamps-1,0]
        self.bptt[self.nsamps-1,2] = self.bptt[self.nsamps-1,0]*self.activation[self.nsamps-2,0]
        
        
        for t in range(self.nsamps-2,0,-1):
            self.bptt[t,4] = self.partial[t] + self.bptt[t+1,1]*self.model[1,2]*self.deriv[t+1,1] + self.bptt[t+1,2]*self.model[2,2]*self.deriv[t+1,2] + self.bptt[t+1,3]*self.model[3,2]*self.deriv[t+1,3] + self.bptt[t+1,0]*self.activation[t+1,1]*self.model[0,2]*self.deriv[t+1,0]
            self.bptt[t,0] = self.bptt[t,4]*self.activation[t,3] + self.bptt[t+1,0]*self.activation[t+1,2]
            self.bptt[t,1] = self.bptt[t,0]*self.activate(self.model[0,0] + self.model[0,1]*self.observation[t] + self.model[0,2]*self.activation[t-1,4])
            self.bptt[t,3] = self.bptt[t,4]*self.activation[t,0]
            self.bptt[t,2] = self.bptt[t,0]*self.activation[t-1,0]
        
        self.bptt[0,4] = self.partial[0] + self.bptt[1,1]*self.model[1,2]*self.deriv[1,1] + self.bptt[1,2]*self.model[2,2]*self.deriv[1,2] + self.bptt[1,3]*self.model[3,2]*self.deriv[1,3] + self.bptt[1,0]*self.activation[1,1]*self.model[0,2]*self.deriv[1,0]
        self.bptt[0,0] = self.bptt[0,4]*self.activation[0,3] + self.bptt[1,0]*self.activation[1,2]
        self.bptt[0,1] = self.bptt[0,0]*self.activate(self.model[0,0] + self.model[0,1]*self.observation[0])
        self.bptt[0,3] = self.bptt[t,4]*self.activation[t,0]
        self.bptt[0,2] = 0
        
    # TODO: calculate the backprop through time

 
    # Calculate the gradient of error with respect to each network weight:
    # self.gradient[0,:] == d(error)/d(bias,inputweight,historyweight) for the cell
    # self.gradient[1,:] == d(error)/d(bias,inputweight,historyweight) for the input gate
    # self.gradient[2,:] == d(error)/d(bias,inputweight,historyweight) for the forget gate
    # self.gradient[3,:] == d(error)/d(bias,inputweight,historyweight) for the output gate
    def set_gradient(self):
        self.gradient = np.zeros((4,3))
        self.gradient[1,0] += self.bptt[0,1]*self.deriv[0,1]
        self.gradient[1,1] += self.bptt[0,1]*self.deriv[0,1]*self.observation[0]

            
        self.gradient[2,0] += self.bptt[0,2]*self.deriv[0,2]
        self.gradient[2,1] += self.bptt[0,2]*self.deriv[0,2]*self.observation[0]

            
        self.gradient[3,0] += self.bptt[0,3]*self.deriv[0,3]
        self.gradient[3,1] += self.bptt[0,3]*self.deriv[0,3]*self.observation[0]

            
        self.gradient[0,0] += self.bptt[0,0]*self.deriv[0,0]*self.activation[0,1]
        self.gradient[0,1] += self.bptt[0,0]*self.deriv[0,0]*self.observation[0]*self.activation[0,1]
        
        
        for t in range(1,self.nsamps):
            self.gradient[1,0] += self.bptt[t,1]*self.deriv[t,1]
            self.gradient[1,1] += self.bptt[t,1]*self.deriv[t,1]*self.observation[t]
            self.gradient[1,2] += self.bptt[t,1]*self.deriv[t,1]*self.activation[t-1,4]
            
            self.gradient[2,0] += self.bptt[t,2]*self.deriv[t,2]
            self.gradient[2,1] += self.bptt[t,2]*self.deriv[t,2]*self.observation[t]
            self.gradient[2,2] += self.bptt[t,2]*self.deriv[t,2]*self.activation[t-1,4]
            
            self.gradient[3,0] += self.bptt[t,3]*self.deriv[t,3]
            self.gradient[3,1] += self.bptt[t,3]*self.deriv[t,3]*self.observation[t]
            self.gradient[3,2] += self.bptt[t,3]*self.deriv[t,3]*self.activation[t-1,4]
            
            self.gradient[0,0] += self.bptt[t,0]*self.deriv[t,0]*self.activation[t,1]
            self.gradient[0,1] += self.bptt[t,0]*self.deriv[t,0]*self.observation[t]*self.activation[t,1]
            self.gradient[0,2] += self.bptt[t,0]*self.deriv[t,0]*self.activation[t-1,4]*self.activation[t,1]
        
        
            
        # TODO: set the gradient

  
    # Update the model weights using one epoch (one step) of gradient descent,
    #   with a learning rate (eta) equal to self.learning_rate.
    # self.update[0,:] == new (bias,inputweight,historyweight) for the cell
    # self.update[1,:] == new (bias,inputweight,historyweight) for the input gate
    # self.update[2,:] == new (bias,inputweight,historyweight) for the forget gate
    # self.update[3,:] == new (bias,inputweight,historyweight) for the output gate
    def set_update(self):
        self.update = np.zeros((4,3))
        self.learning_rate=0.2  # this is eta, the learning rate
        # TODO: set the update
        self.update = self.model - self.learning_rate*self.gradient
        

        #
        # After you've updated the weights, this part writes them to a model JSON file
        with open('models/model%2.2d.json'%(self.epoch+1),'w') as f:
            f.write(json.dumps([ list(row) for row in self.update ]))


if __name__ == '__main__':
    for epoch in range(-1,141):
        lstm=Dataset(epoch)
        print("epoch: "+str(epoch))
        for n in range(len(steps)):
            getattr(lstm,'set_'+steps[n])()
