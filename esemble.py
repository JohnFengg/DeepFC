import torch 
import torch.nn as nn 
from .one_step_lstm import lstm_block
from .multi_step_lstm import direct_lstm,dynamic_lstm
from .mlp import mlp_block,serial_esemble_dnn
import torch.nn.init as init
from .cnn import cnn_block,adaptive_cnn_block,dynamic_cnn_block


class lstm__bias_block(nn.Module):
    """
    input_size: dictionary {'lstm':int,'bias':int}
    neural_size: dictionary {'lstm':int,'bias':list,'hidden':int}
    hidden_layer: dictionary 
    output_size: int
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=1,
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=neural_size['bias'][-1])
        self.bias=mlp_block(input_size=input_size['bias'],
                           hidden_layer=hidden_layer['bias'],
                           neural_size=neural_size['bias'])
        self.hidden=nn.Linear(in_features=neural_size['bias'][-1],
                              out_features=neural_size['hidden'])     
        self.output_layer=nn.Linear(in_features=neural_size['hidden'],
                                    out_features=output_size)  
    #     self.models=[self.hidden]
    #     self.init_weight()

    # def init_weight(self):
    #     for model in self.models:
    #         init.xavier_uniform_(model.weight)
    #         init.zeros_(model.bias) 
    
    def forward(self,x_lstm,x_bias,activate=bool):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.lstm(x_lstm)
        o2=self.bias(x_bias)
        if activate:
            o3=self.output_layer(self.act(self.hidden(o1+o2)))
        else:
            o3=self.hidden(o1+o2)
        return o3


class cnn_lstm__bias_block(nn.Module):
    """
    input_size: dictionary {'lstm':int,'bias':int,'cnn':int}
    neural_size: dictionary {'lstm':int,'bias':list,'cnn':list,'hidden':int}
    hidden_layer: dictionary {'lstm':int,'bias':int,'cnn':int}
    output_size: int
    kernal_size: int
    stride: int
    padding_size: int 
    pool_size: int
    output_size: int
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 kernal_size=None,
                 stride=None,
                 padding_size=None,
                 pool_size=list):
        super().__init__()
        self.act=nn.Tanh()
        self.cnn=cnn_block(neural_size=neural_size['cnn'],
                           hidden_layer=hidden_layer['cnn'],
                           kernel_size=kernal_size,
                           stride=stride,
                           padding_size=padding_size,
                           pool_size=pool_size,
                           linear_input_size=input_size['cnn'],
                           linear_output_size=input_size['lstm']
                           )
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=neural_size['bias'][-1])
        self.bias=mlp_block(input_size=input_size['bias'],
                           hidden_layer=hidden_layer['bias'],
                           neural_size=neural_size['bias'])
        self.hidden=nn.Linear(in_features=neural_size['bias'][-1],
                              out_features=neural_size['hidden'])     
        self.models=[self.hidden]
        self.init_weight()

    def init_weight(self):
        for model in self.models:
            init.xavier_uniform_(model.weight)
            init.zeros_(model.bias) 

    def forward(self,x_lstm,x_bias):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.cnn.forward(x_lstm)
        o1.unsqeeze(dim=2)
        o2=self.lstm.forward(o1)
        o3=self.bias.forward(x_bias)
        o4=self.act(self.hidden(o2+o3))
        return o4
    
    """
    input_size: dictionary {'lstm':int,'bias':int}
    neural_size: dictionary {'lstm':int,'bias':list,'hidden':int}
    hidden_layer: dictionary 
    output_size: int
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=neural_size['bias'][-1])
        self.bias=mlp_block(input_size=input_size['bias'],
                           hidden_layer=hidden_layer['bias'],
                           neural_size=neural_size['bias'])
        self.hidden=nn.Linear(in_features=neural_size['bias'][-1],
                              out_features=neural_size['hidden'])        
    #     self.models=[self.hidden]
    #     self.init_weight()

    # def init_weight(self):
    #     for model in self.models:
    #         init.xavier_uniform_(model.weight)
    #         init.zeros_(model.bias) 
    
    def forward(self,x_lstm,x_bias,activate=bool):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.lstm(x_lstm)
        o2=self.bias(x_bias)
        if activate:
            o3=self.act(self.hidden(o1+o2))
        else:
            o3=self.hidden(o1+o2)
        return o3


class cnn_lstm(nn.Module):
    """
    input_size: dictionary {'lstm':int,'bias':int,'cnn':int}
    neural_size: dictionary {'lstm':int,'bias':list,'cnn':list,'hidden':int}
    hidden_layer: dictionary {'lstm':int,'bias':int,'cnn':int}
    output_size: int
    kernal_size: int
    stride: int
    padding_size: int 
    pool_size: int
    output_size: int
    """
    def __init__(self,lstm_input_size=dict,
                 lstm_neural_size=dict,
                 lstm_hidden_layer=dict,
                 cnn_kernal_size=None,
                 cnn_stride=None,
                 cnn_padding_size=None,
                 cnn_pool_size=list,
                 mlp_input_size=int,
                 mlp_neural_size=list,
                 mlp_hidden_layer=int,
                 hidden_neural=int,
                 output_size=int
                 ) -> None:
        super().__init__()
        self.act=nn.Tanh()
        self.cnn_lstm=cnn_lstm__bias_block(input_size=lstm_input_size,
                                          neural_size=lstm_neural_size,
                                          hidden_layer=lstm_hidden_layer,
                                          kernal_size=cnn_kernal_size,
                                          stride=cnn_stride,
                                          padding_size=cnn_padding_size,
                                          pool_size=cnn_pool_size
                                          )
        self.mlp=mlp_block(input_size=mlp_input_size,
                           hidden_layer=mlp_hidden_layer,
                           neural_size=mlp_neural_size)
        self.hidden=nn.Linear(in_features=mlp_neural_size[-1],
                              out_features=hidden_neural)
        self.output_layer=nn.Linear(in_features=hidden_neural,
                                    out_features=output_size)
        self.models=[self.hidden,self.output_layer]
        self.init_weight()

    def init_weight(self):
        for model in self.models:
            init.xavier_uniform_(model.weight)
            init.zeros_(model.bias)   

    def forward(self,x_lstm,x_bias,x_mlp,activation=bool):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        x_mlp.shape -> (batch_size,feature_size)
        """
        o1=self.cnn_lstm(x_lstm,x_bias)
        o2=self.mlp(x_mlp)
        if activation:
            o3=self.output_layer(self.act(self.hidden(o1+o2)))
        else:
            o3=self.hidden(o1+o2)
        return o3


class adaptive_cnn_direct_lstm(nn.Module):
    def __init__(self, neural_size={'lstm':50,'cnn':[10]},
                 conv_output_size=int,
                 linear_output_sz=int,
                 lstm_input_sz=1,
                 lstm_label_size=1,
                 lstm_hidden_layer=1,
                 prediction_steps=int,
                 *args,**kwargs) -> None:
        super().__init__()
        """
        neural_size -> dictionary in format of {'cnn':list,'lstm':int}
        optional values for adaptive_cnn are [values:defaults]: 
        hidden_layer:1,kernel_size:3, stride:1, padding_size:1
        """
        self.lstm_hidden_sz=lstm_hidden_layer
        self.lstm_neural_sz=neural_size['lstm']
        self.cnn=adaptive_cnn_block(neural_size=neural_size['cnn'],
                                    conv_output_size=conv_output_size,
                                    linear_output_size=linear_output_sz,*args,**kwargs)
        self.lstm=direct_lstm(lstm_input_sz,
                              lstm_label_size,
                              self.lstm_neural_sz,
                              prediction_steps,
                              lstm_hidden_layer)


    def forward(self,x,init_states=None):
        """
        x.shape -> (batch_size,sequence_size,input_size)
        """       
        # print(f'input size for cnn: {x.shape}')
        o1=self.cnn(x)
        # print(f'output size for cnn: {o1.shape}')
        o1=o1.unsqueeze(-1)
        # print(f'input size for lstm: {o1.shape}')
        batch_sz,seq_sz,input_sz=o1.size()
        if init_states is None:
            h_0,c_0=(
                torch.zeros(self.lstm_hidden_sz, batch_sz, self.lstm_neural_sz).to(x.device),
                torch.zeros(self.lstm_hidden_sz, batch_sz, self.lstm_neural_sz).to(x.device)
            )
        else:
            h_0,c_0=init_states
        o2=self.lstm(o1,(h_0,c_0))
        return o2


class dynamic_cnn_direct_lstm(nn.Module):
    def __init__(self, neural_size={'lstm':50,'cnn':10},
                 conv_output_size=int,
                 linear_output_size=int,
                 lstm_input_size=1,
                 lstm_label_size=1,
                 lstm_hidden_layer=1,
                 prediction_steps=int,
                 *args,**kwargs) -> None:
        super().__init__()
        """
        neural_size -> dictionary in format of {'cnn':list,'lstm':int}
        optional values for adaptive_cnn are [values:defaults]: 
        hidden_layer:1,kernel_size:3, stride:1, padding_size:1
        """
        self.lstm_hidden_sz=lstm_hidden_layer
        self.lstm_neural_sz=neural_size['lstm']
        self.cnn=dynamic_cnn_block(neural_size=neural_size['cnn'],
                                   conv_output_size=conv_output_size,
                                   linear_output_size=linear_output_size,
                                   *args,**kwargs)
        self.lstm=direct_lstm(input_size=lstm_input_size,
                               label_size=lstm_label_size,
                               neural_size=neural_size['lstm'],
                               prediction_steps=prediction_steps)


    def forward(self,x,x_mask,*args,**kwargs):
        """
        x.shape -> (batch_size,sequence_size,input_size)
        """       
        # print(f'input size for cnn: {x.shape}')
        o1=self.cnn(x,x_mask)
        # print(f'output size for cnn: {o1.shape}')
        o1=o1.unsqueeze(-1)
        # print(f'input size for lstm: {o1.shape}')
        # tmp_mask=x_mask.sum(dim=2)>0
        # lengths=tmp_mask.sum(dim=1).tolist()
        o2=self.lstm(o1,*args,**kwargs)
        return o2

        
class direct_lstm__dnn__dnn_dnn(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=output_size['lstm'])
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=mlp_block(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            out_size=output_size['dnn3'])


    def forward(self,x_lstm,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.lstm(x_lstm)
        o2=self.dnn1(x_dnn1)
        o3=self.dnn2(x_dnn2)
        # print(f'the shape of unprocessed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o2=o2.sum(dim=1).squeeze(dim=1)
        o3=o3.sum(dim=1).squeeze(dim=1)
        # print(f'the shape of processed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o4=self.fc(o1+o2+o3)
        o5=self.dnn3(o4).unsqueeze(dim=-1)
        return o5

        
class direct_lstm__dnn__dnn_dnn_v2(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=output_size['lstm'])
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=mlp_block(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            out_size=output_size['dnn3'])


    def forward(self,x_lstm,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        # o1=self.lstm(x_lstm)
        o2=self.dnn1(x_dnn1)
        o3=self.dnn2(x_dnn2)
        # print(f'the shape of unprocessed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o2=o2.sum(dim=1).squeeze(dim=1)
        o3=o3.sum(dim=1).squeeze(dim=1)
        # print(f'the shape of processed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o4=self.fc(o2+o3)
        o5=self.dnn3(o4).unsqueeze(dim=-1)
        return o5


class dnn__dnn_dnn(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=mlp_block(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            out_size=output_size['dnn3'])


    def forward(self,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.dnn1(x_dnn1).sum(dim=1).squeeze(dim=1)
        o2=self.dnn2(x_dnn2).sum(dim=1).squeeze(dim=1)
        o3=self.fc(o1+o2)
        o4=self.dnn3(o3).unsqueeze(dim=-1)
        return o4

        
class direct_lstm__dnn__dnn_dnn_v3(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=output_size['lstm'])
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=mlp_block(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            out_size=output_size['dnn3'])


    def forward(self,x_lstm,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.lstm(x_lstm)
        o2=self.dnn1(x_dnn1)
        o3=self.dnn2(x_dnn2)
        # print(f'the shape of unprocessed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o2=o2.sum(dim=1).squeeze(dim=1)
        o3=o3.sum(dim=1).squeeze(dim=1)
        # print(f'the shape of processed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o4=self.fc(o1+o2+o3)
        # o5=self.dnn3(o4).unsqueeze(dim=-1)
        return o4



class direct_lstm__dnn__dnn_dnn_v4(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=output_size['lstm'])
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=mlp_block(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            out_size=output_size['dnn3'])


    def forward(self,x_lstm,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        for name,param in self.fc.named_parameters():
            print(f'layer {name} | Param {param}')
        # o1=self.lstm(x_lstm)
        o2=self.dnn1(x_dnn1)
        o3=self.dnn2(x_dnn2)
        print(f'the shape of unprocessed o2,o3 is {o2.shape},{o3.shape}')
        o2=o2.sum(dim=1).squeeze(dim=1)
        o3=o3.sum(dim=1).squeeze(dim=1)
        print(f'the shape of o2,o3 is {o2.shape},{o3.shape}')
        o4=self.fc(o2+o3)
        print(f'the shape of o4 is {o4.shape}')
        # o5=self.dnn3(o4).unsqueeze(dim=-1)
        return o4
    


class direct_lstm__dnn__dnn_dnn_v5(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=output_size['lstm'])
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=mlp_block(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            out_size=output_size['dnn3'])


    def forward(self,x_lstm,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        # o1=self.lstm(x_lstm)
        o2=self.dnn1(x_dnn1)
        # o3=self.dnn2(x_dnn2)
        # print(f'the shape of unprocessed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o2=o2.sum(dim=1).squeeze(dim=1)
        # o3=o3.sum(dim=1).squeeze(dim=1)
        # print(f'the shape of processed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o4=self.fc(o2)
        # o5=self.dnn3(o4).unsqueeze(dim=-1)
        return o4
    
class direct_lstm__dnn__dnn_dnn_v6(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=output_size['lstm'])
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=mlp_block(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            out_size=output_size['dnn3'])


    def forward(self,x_lstm,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.lstm(x_lstm)
        # o2=self.dnn1(x_dnn1)
        # o3=self.dnn2(x_dnn2)
        # print(f'the shape of unprocessed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        # o2=o2.sum(dim=1).squeeze(dim=1)
        # o3=o3.sum(dim=1).squeeze(dim=1)
        # print(f'the shape of processed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o4=self.fc(o1)
        # o5=self.dnn3(o4).unsqueeze(dim=-1)
        return o4
    

class direct_lstm__dnn__dnn_esemble_dnn(nn.Module):
    """
    input_size: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    neural_size: dictionary {'lstm':int,'dnn1':list,'dnn2':list,'dnn3':list}
    hidden_layer: dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    output_size:dictionary {'lstm':int,'dnn1':int,'dnn2':int,'dnn3':int}
    """
    def __init__(self,input_size=dict,
                 neural_size=dict,
                 hidden_layer=dict,
                 output_size=dict
                 ):
        super().__init__()
        self.act=nn.Tanh()
        self.lstm=lstm_block(input_size=input_size['lstm'],
                             neural_size=neural_size['lstm'],
                             hidden_layer=hidden_layer['lstm'],
                             out_size=output_size['lstm'])
        self.dnn1=mlp_block(input_size=input_size['dnn1'],
                           hidden_layer=hidden_layer['dnn1'],
                           neural_size=neural_size['dnn1'],
                           out_size=output_size['dnn1'],
                           out_activation=True)
        self.dnn2=mlp_block(input_size=input_size['dnn2'],
                           hidden_layer=hidden_layer['dnn2'],
                           neural_size=neural_size['dnn2'],
                           out_size=output_size['dnn2'],
                           out_activation=True)        
        self.fc=nn.Linear(in_features=output_size['dnn1'],
                              out_features=input_size['dnn3'])     
        self.dnn3=serial_esemble_dnn(input_size=input_size['dnn3'],
                            hidden_layer=hidden_layer['dnn3'],
                            neural_size=neural_size['dnn3'],
                            num_models=output_size['dnn3'])


    def forward(self,x_lstm,x_dnn1,x_dnn2):
        """
        x_lstm.shape -> (batch_size,sequence_size)
        x_bias.shape -> (batch_size,feature_size)
        """
        o1=self.lstm(x_lstm)
        o2=self.dnn1(x_dnn1)
        o3=self.dnn2(x_dnn2)
        # print(f'the shape of unprocessed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o2=o2.sum(dim=1).squeeze(dim=1)
        o3=o3.sum(dim=1).squeeze(dim=1)
        # print(f'the shape of processed o1,o2,o3 is {o1.shape},{o2.shape},{o3.shape}')
        o4=self.fc(o1)
        o5=self.dnn3(o4).unsqueeze(-1)
        print(f'the shape of processed o4,o5 in esemble is {o4.shape},{o5.shape}')
        return o5