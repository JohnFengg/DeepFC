import torch 
import torch.nn as nn 
import torch.nn.init as init

class cnn_block(nn.Module):
    def __init__(self,neural_size=list,
                 hidden_layer=1,
                 kernel_size=3,
                 stride=1,
                 padding_size=1,
                 pool_size=list,
                 linear_input_size=int,
                 linear_output_size=int,
                 ):
        super().__init__()
        layers=[]
        for i in range(hidden_layer):
            conv_i=nn.Conv1d(in_channels=1,
                             out_channels=neural_size[i],
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding_size
                             )
            layers.append(conv_i)
            layers.append(nn.ReLU)
            layers.append(nn.MaxPool1d(kernel_size=pool_size[i]))
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear(linear_input_size,linear_output_size))
        layers.append(nn.Sigmoid())
        self.deep=nn.Sequential(*layers)
    #     self.init_weight()

    # def init_weight(self):
    #     for model in self.deep:
    #         if isinstance(model,nn.Linear):
    #             init.xavier_uniform_(model.weight)
    #             init.zeros_(model.bias)
    #         if isinstance(model,nn.LSTM):
    #             init.xavier_uniform_(model.weight)
    #             init.zeros_(model.bias)       
    #         if isinstance(model,nn.Conv1d):
    #             init.kaiming_normal_(model.weight)
    #             init.zeros_(model.bias)       
    
    def forward(self,x):
        o=self.deep(x)
        return o 
    
class adaptive_cnn_block(nn.Module):
    def __init__(self,neural_size=list,
                 hidden_layer=1,
                 kernel_size=3,
                 stride=1,
                 padding_size=1,
                 conv_output_size=int,
                 linear_output_size=int,
                 ):
        super().__init__()
        layers=[]
        for i in range(hidden_layer):
            conv_i=nn.Conv1d(in_channels=1,
                             out_channels=neural_size[i],
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding_size
                             )
            layers.append(conv_i)
            layers.append(nn.ReLU())
            layers.append(nn.AdaptiveAvgPool1d(output_size=conv_output_size))
        
        layers.append(nn.Flatten())
        layers.append(nn.Linear(conv_output_size*neural_size[-1],linear_output_size))
        self.deep=nn.Sequential(*layers)
    #     self.init_weight()

    # def init_weight(self):
    #     for model in self.deep:
    #         if isinstance(model,nn.Linear):
    #             init.xavier_uniform_(model.weight)
    #             init.zeros_(model.bias)
    #         if isinstance(model,nn.LSTM):
    #             init.xavier_uniform_(model.weight)
    #             init.zeros_(model.bias)       
    #         if isinstance(model,nn.Conv1d):
    #             init.kaiming_normal_(model.weight)
    #             init.zeros_(model.bias)       
    
    def forward(self,x):
        """
        x.shape -> (batch_size,sequence_size,input_size)
        would be transposed to (batch_size,channel_size,sequence_size)
        output_size -> (batch_size,linear_output_size)
        """
        x=torch.transpose(x,1,2)
        o=self.deep(x)
        return o 

class dynamic_cnn_block(nn.Module):
    def __init__(self,neural_size=int,
                 hidden_layer=1,
                 kernel_size=3,
                 stride=1,
                 padding_size=1,
                 conv_output_size=int,
                 linear_output_size=int,
                 ):
        super().__init__()
        self.neural_sz=neural_size
        self.conv=nn.Conv1d(in_channels=1,
                     out_channels=neural_size,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding_size
                     )
        self.act=nn.ReLU()
        self.pool=nn.AdaptiveAvgPool1d(output_size=conv_output_size)
        self.flatten=nn.Flatten()
        self.fc=nn.Linear(conv_output_size*neural_size,linear_output_size)

    def forward(self,x,x_mask):
        """
        x.shape -> (batch_size,sequence_size,input_size)
        would be transposed to (batch_size,channel_size,sequence_size)
        output_size -> (batch_size,linear_output_size)
        """
        x=torch.transpose(x,1,2)
        x_mask=torch.transpose(x_mask,1,2)
        x_mask_expanded=x_mask.expand(x_mask.size(0),self.neural_sz,x_mask.size(2))
        o1=self.conv(x)
        o1*=x_mask_expanded
        o2=self.act(o1)
        o3=self.pool(o2)
        pooled_mask=self.pool(x_mask)
        pooled_mask=torch.where(pooled_mask==1.0,torch.tensor(1.0),torch.tensor(0.0))
        o3*=pooled_mask.expand(pooled_mask.size(0),self.neural_sz,pooled_mask.size(2))
        o4=self.flatten(o3)
        o5=self.fc(o4)
        return o5