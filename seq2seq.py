import torch
import torch.nn as nn


class LSTM_Encoder(nn.Module):
    def __init__(self,input_size,neural_size,hidden_layer):
        super().__init__()
        self.lstm=nn.LSTM(input_size,neural_size,hidden_layer,batch_first=True)

    def forward(self,X):
        """
        x.shape->(batch_sz,seq_sz,feature_sz)
        outputs.shape->(batch_sz,seq_sz,neural_sz)
        h0,c0->(hidden_layer,batch_sz,neural_sz)
        """
        outputs,(hidden, cell)=self.lstm(X)
        return outputs[:,-1,:],hidden, cell


class LSTM_Decoder(nn.Module):
    def __init__(self,input_size,neural_size,hidden_layer,output_size):
        super().__init__()
        self.lstm=nn.LSTM(input_size,neural_size,hidden_layer,batch_first=True)
        self.fc=nn.Linear(neural_size,output_size)

    def forward(self,X,h0,c0):
        """
        x.shape->(batch_sz,input_sz).unsqueeze->(batch_sz,1,input_sz)
        h0,c0.shape->(hidden_layer,batch_sz,neural_sz)
        outputs.shape->(batch_sz,1,neural_sz).squeeze->(batch_sz,neural_sz)
        o1.shape->(batch_sz,output_sz)
        """
        X=X.unsqueeze(1) 
        outputs,(h,c)=self.lstm(X,(h0,c0))
        o1=self.fc(outputs.squeeze(1))
        return o1,h,c


class LSTM_Seq2Seq(nn.Module):
    def __init__(self,input_size,
                 neural_size,
                 hidden_layer,
                 output_size,
                 pred_len,
                 device,
                 teacher_forcing_ratio=0.5,
                 ):
        """
        input_size,neural_size,hidden_layer->dict({'encoder':int,'decoder':int})
        """
        super().__init__()
        self.encoder=LSTM_Encoder(input_size['encoder'],
                                  neural_size['encoder'],
                                  hidden_layer['encoder'])
        self.decoder=LSTM_Decoder(input_size['decoder'],
                                  neural_size['decoder'],
                                  hidden_layer['decoder'],
                                  output_size)
        self.out_sz=output_size
        self.device=device
        self.pred_len=pred_len
        self.tfr=teacher_forcing_ratio

    def forward(self,X,y):
        """
        x.shape->(batch_sz,seq_sz,feature_sz)
        y.shape->(batch_sz,seq_sz,feature_sz)
        outputs->
        """
        batch_sz=X.size(0)
        encoder_out,h0,c0=self.encoder(X)
        input=y[:,0,:] #if y is not None else encoder_out

        final_out=torch.zeros(batch_sz,self.pred_len,1).to(self.device)
        for t in range(self.pred_len):
            output,h0,c0=self.decoder(input,h0,c0)
            final_out[:,t,:]=output
            if y is not None and t<self.pred_len-1:
                teacher_force=torch.rand(1).item()<self.tfr
                input=y[:,t+1,:] if teacher_force else output
            else:
                input=output
        return final_out
    
class Desc_Seq2Seq(nn.Module):
    def __init__(self,input_size,
                 neural_size,
                 hidden_layer,
                 output_size,
                 pred_len,
                 device,
                 teacher_forcing_ratio=0.5,
                 ):
        """
        input_size,neural_size,hidden_layer->dict({'encoder':int,'decoder':int})
        """
        super().__init__()
        self.encoder=LSTM_Encoder(input_size['encoder'],
                                  neural_size['encoder'],
                                  hidden_layer['encoder'])
        self.decoder=LSTM_Decoder(input_size['decoder'],
                                  neural_size['decoder'],
                                  hidden_layer['decoder'],
                                  output_size)
        self.out_sz=output_size
        self.device=device
        self.pred_len=pred_len
        self.tfr=teacher_forcing_ratio

    def forward(self,X_encode,X_decode,y):
        """
        x.shape->(batch_sz,seq_sz,feature_sz)
        y.shape->(batch_sz,seq_sz,feature_sz)
        outputs->
        """
        batch_sz=X_encode.size(0)
        encoder_out,h0,c0=self.encoder(X_encode)
        input=torch.sum(torch.sum(X_decode,dim=1),dim=1).unsqueeze(-1)
        print(f'the shape of h0,c0 and input for decoder is {h0.shape},{c0.shape},{input.shape}')
        final_out=torch.zeros(batch_sz,self.pred_len,1).to(self.device)
        for t in range(self.pred_len):
            output,h0,c0=self.decoder(input,h0,c0)
            final_out[:,t,:]=output
            if y is not None and t<self.pred_len-1:
                teacher_force=torch.rand(1).item()<self.tfr
                input=y[:,t+1,:] if teacher_force else output
            else:
                input=output
        return final_out