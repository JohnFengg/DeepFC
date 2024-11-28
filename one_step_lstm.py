import torch 
import torch.nn as nn 

class lstm_block(nn.Module):
    def __init__(self,input_size=int,neural_size=int,hidden_layer=1,out_size=int) -> None:
        super().__init__()
        self.nueral_sz=neural_size
        self.hidden_layer=hidden_layer
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=self.nueral_sz,
                          num_layers=hidden_layer,
                          batch_first=True
                          )
        self.hidden=nn.Linear(in_features=self.nueral_sz,
                              out_features=out_size)
        # self.models=[self.lstm,self.hidden]
    #     self.init_weight()

    # def init_weight(self):
    #     for model in self.models:
    #         init.xavier_uniform_(model.weight)
    #         init.zeros_(model.bias)            

    def forward(self,x,init_states=None):
        """
        x.shape -> (batch_size,sequence_size,input_size)
        """
        batch_sz,seq_sz,input_sz=x.size()
        if init_states is None:
            h_0,c_0=(
                torch.zeros(self.hidden_layer,batch_sz,self.nueral_sz).to(x.device),
                torch.zeros(self.hidden_layer,batch_sz,self.nueral_sz).to(x.device)
            )
        else:
            h_0,c_0=init_states
        o1,_=self.lstm(x,(h_0,c_0))
        o2=self.hidden(o1[:,-1,:])
        return o2

class lstm(nn.Module):
    def __init__(self,input_size,label_size,neural_size,hidden_layer) -> None:
        super().__init__()
        self.hidden_layer=hidden_layer
        self.neural_sz=neural_size
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=self.neural_sz,
                          num_layers=self.hidden_layer,
                          batch_first=True
                          )
        self.hidden=nn.Linear(in_features=self.neural_sz,
                              out_features=label_size)
        # self.models=[self.lstm,self.hidden]        
        #self.init_weight()

    # def init_weight(self):
    #     for model in self.models:
    #         init.xavier_uniform_(model.weight)
    #         init.zeros_(model.bias)            

    def forward(self,x,init_states=None):
        """
        x.shape -> (batch_size,sequence_size,input_size)
        """
        batch_sz,seq_sz,input_sz=x.size()
        if init_states is None:
            h_0,c_0=(
                torch.zeros(self.hidden_layer,batch_sz,self.neural_sz).to(x.device),
                torch.zeros(self.hidden_layer,batch_sz,self.neural_sz).to(x.device)
            )
        else:
            h_0,c_0=init_states
        o1,(h_t,c_t)=self.lstm(x,(h_0,c_0))
        o2=self.hidden(o1[:,-1,:])
        return o2,(h_t,c_t)