import torch 
import torch.nn as nn 
import numpy as np 
import time

class model_train():
    def __init__(self,epoch_num=None,
                 log_prefix='log',
                 model_prefix='model',
                 version=str,
                 data_train=None,
                 data_test=None,
                 model=None,
                 device=None,
                 loss_f=nn.MSELoss(),
                 optimizer=None):
        super().__init__()
        self.epoch_num=epoch_num
        self.log_prefix=log_prefix
        self.model_prefix=model_prefix
        self.version=version
        self.dtrain=data_train
        self.dtest=data_test
        self.model=model
        self.device=device
        self.loss=loss_f
        self.opt=optimizer

    def train(self,LSTM=False,*args,**kwargs):
        train_loss=0
        for X,mask,y in self.dtrain:
            X=X.squeeze(dim=0)
            mask=mask.squeeze(dim=0)
            y=y.squeeze(dim=0)        
            X,mask,y=X.to(self.device),mask.to(self.device),y.to(self.device)
            self.opt.zero_grad()
            if LSTM:
                out,_=self.model(X,mask,*args,**kwargs)
            else:
                out=self.model(X,mask,*args,**kwargs)
            loss=self.loss(out,y)
            loss.backward()
            self.opt.step()
            train_loss+=loss.item()*X.size(0)
        train_loss/=len(self.dtrain.dataset)
        return train_loss

    def test(self,LSTM=False,*args,**kwargs):
        val_loss=0
        with torch.no_grad():
            for X,mask,y in self.dtest:
                X=X.squeeze(dim=0)
                mask=mask.squeeze(dim=0)
                y=y.squeeze(dim=0) 
                X,mask,y=X.to(self.device),mask.to(self.device),y.to(self.device)
                if LSTM:
                    out,_=self.model(X,mask,*args,**kwargs)
                else:
                    out=self.model(X,mask,*args,**kwargs)
                loss=self.loss(out,y)
                val_loss+=loss.item()*X.size(0)
        val_loss/=len(self.dtest.dataset)
        return val_loss
    
    def run(self,LSTM=False,*args,**kwargs):
        with open(f'{self.log_prefix}/lcurve{self.version}.out','a') as f:
                f.write('#epoch\tmse_trn\tmse_test\n')
        total_check=time.time()
        for epoch in range(self.epoch_num):
            epoch_check=time.time()

            trn_loss=self.train(LSTM,*args,**kwargs)

            trn_time=time.time()-epoch_check

            val_check=time.time()

            val_loss=self.test(LSTM,*args,**kwargs)

            val_time=time.time()-val_check

            total_epoch_time=time.time()-epoch_check

            with open(f'{self.log_prefix}/train{self.version}.log','a') as f:
                f.write(f'epoch {epoch}/{self.epoch_num}, training time {trn_time:.3f} s, testing time {val_time:.3f} s, total wall time {total_epoch_time:.3f} s\n')
            with open(f'{self.log_prefix}/lcurve{self.version}.out','a') as f:
                f.write(f'{epoch}\t{trn_loss:.3e}\t{val_loss:.3e}\n')

        total_wall_time=time.time()-total_check

        torch.save(self.model,f'{self.model_prefix}/model{self.version}')

        with open(f'{self.log_prefix}/train{self.version}.log','a') as f:
                f.write(f'finish training\nwall time: {total_wall_time:.3f} s\n')

def predict(dataset,model,device,extrapolation=False,pred_len=None,LSTM=False,*args,**kwargs):
    with torch.no_grad():
        if extrapolation:
            X=dataset.unsqueeze(dim=0).to(device)
            out_list=[]
            # out_list=list(dataset)
            h_0,c_0=None,None
            for i in range(pred_len):
                # x=torch.tensor(out_list[-len(dataset):]).reshape(-1,1)
                # X=x.unsqueeze(dim=0).to(device)
                if LSTM:
                    if h_0 is None or c_0 is None:
                        out,(h_0, c_0)=model(X,*args,**kwargs)
                    else:
                        out,(h_0, c_0)=model(X,init_states=(h_0, c_0),*args,**kwargs)
                else:
                    out=model(X,*args,**kwargs)
                # out_list.append(out.cpu())
                out=out.view(-1)
                out_list.append(out.cpu())
                X=out.unsqueeze(dim=1)
            # out_list=np.array(torch.tensor(out_list)[len(dataset):])
            # out_list=torch.cat(out_list,dim=1).numpy().flatten()
            # print(len(out_list))
        else:    
            out_list=[]
            for X,mask,y in dataset:
                X,mask=X.to(device),mask.to(device)
                if LSTM:
                    out,(h_t,c_t)=model(X,*args,**kwargs)
                else:
                    out=model(X,mask,*args,**kwargs)
                out_list.append(out.squeeze(dim=-1).cpu())
                # out_list.append(out_num)
            out_list=torch.cat(out_list,dim=0).squeeze(dim=0).numpy()
    return out_list




