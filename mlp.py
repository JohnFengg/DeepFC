import torch 
import torch.nn as nn 
import torch.nn.init as init

class mlp_block(nn.Module):
    """
    input_size: number of features
    neural_size:  a list that contains number of neural in input layer and each hidden layer,
                so len(neural_size)=hidden_layer+1 
    """
    def __init__(self,input_size=int,hidden_layer=int,neural_size=list,out_size=int,out_activation=False,act=None):
        super().__init__()
        if act is None:
            act=nn.Tanh()
        self.input_layer=nn.Linear(input_size,neural_size[0])
        layers=[
            self.input_layer,
            act
            ]
        for i in range(hidden_layer):
            hidden_layer_i=nn.Linear(neural_size[i],neural_size[i+1])
            layers.append(hidden_layer_i)
            layers.append(act)
        output_layer=nn.Linear(neural_size[-1],out_size)
        layers.append(output_layer)
        if out_activation:
            layers.append(act)
        self.deep=nn.Sequential(*layers)

    def forward(self,x):
        o=self.deep(x)
        return o

    
class serial_esemble_dnn(nn.Module):
    def __init__(self,input_size=int,
                 hidden_layer=int,
                 neural_size=list,
                 out_activation=False,
                 act=None,
                 num_models=int):
        super().__init__()
        self.models=nn.ModuleList([
            mlp_block(input_size,hidden_layer,neural_size,1,out_activation,act) for _ in range(num_models)
        ])

    def forward(self,x):
        o1=torch.cat([model(x) for model in self.models],dim=1)
        print(f'the shape of output in mlp is {o1.shape}')
        return o1
    