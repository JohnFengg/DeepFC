import torch 
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence

class direct_lstm(nn.Module):
    def __init__(self,input_size,label_size,neural_size,prediction_steps,hidden_layer=1) -> None:
        super().__init__()
        self.hidden_layer=hidden_layer
        self.neural_sz=neural_size
        self.pred_steps=prediction_steps
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=self.neural_sz,
                          num_layers=self.hidden_layer,
                          batch_first=True
                          )
        self.hidden=nn.Linear(in_features=self.neural_sz,
                              out_features=label_size*self.pred_steps)
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
        o1,_=self.lstm(x,(h_0,c_0))
        o2=self.hidden(o1[:,-1,:])
        o2=o2.view(batch_sz,self.pred_steps,-1)
        return o2


class dynamic_lstm(nn.Module):
    def __init__(self,input_size,label_size,neural_size,prediction_steps,hidden_layer=1) -> None:
        super().__init__()
        self.hidden_layer=hidden_layer
        self.neural_sz=neural_size
        self.pred_steps=prediction_steps
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=self.neural_sz,
                          num_layers=self.hidden_layer,
                          batch_first=True
                          )
        self.fc=nn.Linear(in_features=self.neural_sz,
                              out_features=label_size*self.pred_steps)        

    def forward(self,x,lengths,init_states=None):
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
        x_packed=pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        _,(h_1,__)=self.lstm(x_packed,(h_0,c_0))
        o2=self.fc(h_1[-1])   #in size of (batch_sz,neural_sz)
        o2=o2.view(batch_sz,self.pred_steps,-1)
        return o2



class iterative_lstm(nn.Module):
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

    def forward(self,x,prediction_steps,init_states=None):
        """
        x.shape -> (batch_size,sequence_size,input_size)
        """
        batch_sz,seq_sz,input_sz=x.size()
        outputs=[]

        if init_states is None:
            h_0,c_0=(
                torch.zeros(self.hidden_layer,batch_sz,self.neural_sz).to(x.device),
                torch.zeros(self.hidden_layer,batch_sz,self.neural_sz).to(x.device)
            )
        else:
            h_0,c_0=init_states

        for _ in range(prediction_steps):
            # o1,_=self.lstm(x,(h_0,c_0))
            o1,(h_0,c_0)=self.lstm(x,(h_0,c_0))      
            o2=self.hidden(o1[:,-1,:])
            tmp_o2=o2.unsqueeze(dim=1)
            outputs.append(tmp_o2)
            x=tmp_o2
            # x=torch.cat((x[:,1:,:],tmp_o2),dim=1)
        outputs=torch.cat(outputs,dim=1)
        return outputs