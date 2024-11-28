import torch 
import numpy as np
import torch.nn as nn 
import numpy as np 
import time
import os

class model_train():
    def __init__(self,epoch_num=None,
                 log_prefix='log',
                 model_prefix='model',
                 version=str,
                 data_train=None,
                 data_test=None,
                 model=None,
                 device=None,
                 loss_f=nn.MSELoss(reduction='none'),
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

    def train(self,epoch,grads_path,LSTM=False,*args,**kwargs):
        train_loss=0
        for data in self.dtrain:
            *inputs,y=data
            inputs=[input.to(self.device) for input in inputs]
            y=y.to(self.device)
            self.opt.zero_grad()
            if LSTM:
                out=self.model(*inputs,y,*args,**kwargs)
            else:
                out=self.model(*inputs,*args,**kwargs)
            # loss_per_element=self.loss(out,y)
            # loss_per_batch=loss_per_element.sum(dim=1)
            # loss=loss_per_batch.mean()
            loss=self.loss(out,y)
            loss.backward()
            self.opt.step()
            train_loss+=loss.item()*inputs[0].size(0)

        for name,param in self.model.named_parameters():
            if param.grad is not None:
                with open(f'{grads_path}/{epoch}','a') as f:
                    f.write(f'{name}\t{param.grad.max()}\t{param.grad.min()}\t{param.grad.mean()}\n')

        train_loss/=len(self.dtrain.dataset)
        return train_loss

    def test(self,epoch,LSTM=False,*args,**kwargs):
        val_loss=0
        with torch.no_grad():
            for data in self.dtest:
                *inputs,y=data
                inputs=[input.to(self.device) for input in inputs]
                y=y.to(self.device)
                if LSTM:
                    out=self.model(*inputs,y,*args,**kwargs)
                else:
                    out=self.model(*inputs,*args,**kwargs)
                # loss_per_element=self.loss(out,y)
                # loss_per_batch=loss_per_element.sum(dim=1)
                # loss=loss_per_batch.mean()
                loss=self.loss(out,y)
                val_loss+=loss.item()*inputs[0].size(0)
        val_loss/=len(self.dtest.dataset)
        return val_loss
    
    def run(self,LSTM=False,*args,**kwargs):
        with open(f'{self.log_prefix}/lcurve{self.version}.out','a') as f:
                f.write('#epoch\tmse_trn\tmse_test\n')
        if not os.path.exists(f'log/{self.version}_grads'):
            os.makedirs(f'log/{self.version}_grads')
        else:
            pass
        grads_path=f'log/{self.version}_grads'

        total_check=time.time()
        for epoch in range(self.epoch_num):
            epoch_check=time.time()
            trn_loss=self.train(epoch,grads_path,LSTM,*args,**kwargs)
            trn_time=time.time()-epoch_check

            val_check=time.time()
            val_loss=self.test(epoch,LSTM,*args,**kwargs)
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

# def predict(dataset,model,device,extrapolation=False,pred_len=None,LSTM=False,*args,**kwargs):
#     with torch.no_grad():
#         if extrapolation:
#             X=dataset.unsqueeze(dim=0).to(device)
#             out_list=[]
#             # out_list=list(dataset)
#             h_0,c_0=None,None
#             for i in range(pred_len):
#                 # x=torch.tensor(out_list[-len(dataset):]).reshape(-1,1)
#                 # X=x.unsqueeze(dim=0).to(device)
#                 if LSTM:
#                     if h_0 is None or c_0 is None:
#                         out,(h_0, c_0)=model(X,*args,**kwargs)
#                     else:
#                         out,(h_0, c_0)=model(X,init_states=(h_0, c_0),*args,**kwargs)
#                 else:
#                     out=model(X,*args,**kwargs)
#                 # out_list.append(out.cpu())
#                 out=out.view(-1)
#                 out_list.append(out.cpu())
#                 X=out.unsqueeze(dim=1)
#             # out_list=np.array(torch.tensor(out_list)[len(dataset):])
#             # out_list=torch.cat(out_list,dim=1).numpy().flatten()
#             # print(len(out_list))
#         else:    
#             out_list=[]
#             for data in dataset:
#                 *inputs,y=data
#                 inputs=[input.to(device) for input in inputs]
#                 # out=model(X,*args,**kwargs)
#                 if LSTM:
#                     out,(h_t,c_t)=model(*inputs,*args,**kwargs)
#                 else:
#                     out=model(*inputs,*args,**kwargs)
#                     # print(f'the shape of predicted output is {out.shape}')
#                 out_list.append(out.squeeze().cpu())
#                 # out_list.append(out_num)
#             # out_list=torch.cat(out_list,dim=0).numpy()
#             out_list=np.array(out_list)
#             # print(f'the shape of predicted output_list is {out_list.shape}')
#     return out_list



def predict(dataset,model,device,LSTM=False,*args,**kwargs):
    with torch.no_grad():
        out_list=[]
        for data in dataset:
            *inputs,y=data
            inputs=[input.to(device) for input in inputs]
            y=y.to(device)
            # print(f'the shape of X is {inputs[0].shape}')
            # print(f'the shape of y is {y.shape}')
            # out=model(X,*args,**kwargs)
            if LSTM:
                out=model(*inputs,y,*args,**kwargs)
            else:
                out=model(*inputs,*args,**kwargs)
                # print(f'the shape of predicted output is {out.shape}')
            out_list.append(out.squeeze().cpu())
            # out_list.append(out_num)
        # out_list=torch.cat(out_list,dim=0).numpy()
        out_list=np.array(out_list)
        # print(f'the shape of predicted output_list is {out_list.shape}')
    return out_list
