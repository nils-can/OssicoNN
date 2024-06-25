import torch.nn as nn
from conditioning_nn_girafe_optimized import *
import FrEIA.framework as Ff
import FrEIA.modules as Fm
# from torchsummary import summary

class CinnConditional(nn.Module):
    def __init__(self, lr, n_tot):
        super().__init__()
        #Definition of the conditional and conditioning Neural Network
        self.cinn = self.build_inn(n_tot)
        self.cond_net = CondNet_girafe()

        #Define trainable parameters
        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters: #Xavier Initialization
            p.data = 0.01 * torch.randn_like(p)

        #add conditionning hyperparameters of the conditioning Neural Network 
        self.trainable_parameters += list(self.cond_net.parameters())
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self, n_tot):

        def subnet(dim_in, dim_out): #subnet in GLOVES Bock
            return nn.Sequential(nn.Linear(dim_in, 512),
                                nn.ReLU(),
                                nn.Linear(512, dim_out))

        # 4 condition each for one block
        conditions = [Ff.ConditionNode(256),
                      Ff.ConditionNode(32),
                      Ff.ConditionNode(256),
                      Ff.ConditionNode(32)]

        # Initial block which take [teff, logg, feh, abundance]
        nodes = [Ff.InputNode(n_tot)]
        
        # add other Block of the conditional Neural Network
        for k in range(4):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom , {'seed':k})) # seed for shuffle
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet, 'clamp':1.9}, # clamp of the weight
                                conditions=conditions[k])) #Here the conditions
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))
        # Output Node
        nodes.append(Ff.OutputNode(nodes[-1]))
            
        return Ff.ReversibleGraphNet(nodes + conditions) #verbose=True to see resume

    def forward(self, x, l, rev=False):
        z, jac = self.cinn.forward(x, c=self.cond_net(l), rev=rev) #Go Forward or BackWard
        return z, jac