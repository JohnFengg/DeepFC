import torch 
from torch.utils.data import Dataset
import numpy as np 

class SequenceDataset(Dataset):
    def __init__(self,X,y,seq_lenth,multistep=None) -> None:
        super().__init__()
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.seq_len=seq_lenth
        self.multistep=multistep

    def __len__(self):
        if self.multistep is not None:
            lenth=len(self.y)-self.seq_len-self.multistep+1
        else:
            lenth=len(self.y)-self.seq_len
        return lenth

    def __getitem__(self, index):
        if self.multistep is not None:
            X_idx=self.X[index:index+self.seq_len]  
            y_idx=self.y[index+self.seq_len:index+self.seq_len+self.multistep]          
        else:
            X_idx=self.X[index:index+self.seq_len]
            y_idx=self.y[index+self.seq_len]
        return X_idx,y_idx

class Adaptive_SequenceDataset(Dataset):
    def __init__(self,X,y,init_steps=int,multistep=int) -> None:
        super().__init__()
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.init_steps=init_steps
        self.multistep=multistep

    def __len__(self):
        lenth=len(self.y)-self.init_steps-self.multistep+1
        return lenth

    def __getitem__(self, index):
        X_idx=self.X[0:index+self.init_steps]  
        start_point=self.init_steps+index
        y_idx=self.y[start_point:self.multistep+start_point]          
        return X_idx,y_idx

class Multi_Adaptive_SequenceDataset(Dataset):
    """
    X.shape->(sequence_sz,feature_sz)
    y.shape->(sequence_sz,feature_sz)
    """
    def __init__(self,X,y,init_steps=int,pred_steps=int,index_corrector=0,y_corrector=0) -> None:
        super().__init__()
        # self.X=torch.tensor(X,dtype=torch.float32)
        # self.y=torch.tensor(y,dtype=torch.float32)
        self.X=X
        self.y=y
        self.init_steps=init_steps
        self.pred_steps=pred_steps
        self.idx_corrector=index_corrector
        self.y_corrector=y_corrector

    def __len__(self):
        lenth=len(self.y)-self.idx_corrector-self.init_steps-self.pred_steps-self.y_corrector+1
        return lenth

    def __getitem__(self,index):
        index+=self.idx_corrector
        start_point=self.init_steps+index+self.y_corrector
        X_feature=self.X[0:index+self.init_steps][:,1:]
        y_feature=self.y[start_point:self.pred_steps+start_point][:,1:-1]

        X_seq=self.X[index:index+self.init_steps][:,-1].reshape(-1,1) 
        y_seq=self.y[start_point:self.pred_steps+start_point][:,-1].reshape(-1,1)    
        return X_feature,y_feature,X_seq,y_seq

class Multi_SequenceDataset_tmp(Dataset):
    def __init__(self,X_seqs,y_seqs,X_features,y_features,interval=1) -> None:
        super().__init__()
        self.X_seqs=X_seqs
        self.y_seqs=y_seqs
        self.X_feats=X_features
        self.y_feats=y_features
        self.interval=interval
        
    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, index):
        X_seq=self.X_seqs[index*self.interval]
        y_seq=self.y_seqs[index*self.interval]
        X_feat=self.X_feats[index*self.interval]
        y_feat=self.y_feats[index*self.interval]
        return X_seq,X_feat,y_feat,y_seq

class Multi_SequenceDataset(Dataset):
    def __init__(self,y_seqs,X_features,y_features,interval=1) -> None:
        super().__init__()
        self.y_seqs=y_seqs
        self.X_feats=X_features
        self.y_feats=y_features
        self.interval=interval
        
    def __len__(self):
        return len(self.y_seqs)

    def __getitem__(self, index):
        y_seq=self.y_seqs[index*self.interval]
        X_feat=self.X_feats[index*self.interval]
        y_feat=self.y_feats[index*self.interval]
        return X_feat,y_feat,y_seq


class SimpleDataset(Dataset):
    def __init__(self,X1,y) -> None:
        super().__init__()
        self.X1=X1
        # self.X2=X2
        self.y=y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self,index):
        X1=self.X1[index]
        # X2=self.X2[index]
        y=self.y[index]
        return X1,y
    


class Masked_SequenceDataset(Dataset):
    def __init__(self,X,y,init_steps=int,multistep=int,batch_sz=int) -> None:
        super().__init__()
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.init_steps=init_steps
        self.multistep=multistep
        self.batch_sz=batch_sz
        self.lenth=(len(self.y)-self.init_steps-self.multistep+1)//self.batch_sz+1

    def padding(self,batchs):
        max_seq_len=len(batchs[-1])
        # print(f'max_seq_len is {max_seq_len}')
        padded_batchs=torch.zeros(len(batchs),max_seq_len,self.X.size(1))
        # print(f'padded_batch_sz is {padded_batchs.size()}')
        mask_batch=torch.zeros(len(batchs),max_seq_len,self.X.size(1))
        # print(f'lenth of batchs is {len(batchs)}')
        for i in range(len(batchs)):
            seq_len=batchs[i].size(0)
            # print(seq_len)
            padded_batchs[i,:seq_len,:]=batchs[i]
            mask_batch[i,:seq_len,:]=1
        return padded_batchs,mask_batch

    def __len__(self):
        lenth=self.lenth
        return lenth

    def __getitem__(self, index):
        start_point_X=index*self.batch_sz
        # print(f'start point is {start_point_X}')
        end_point_X=np.min((start_point_X+self.batch_sz,
                           len(self.y)-self.init_steps-self.multistep+1))
        # print(f'exd point is {end_point_X}')
        X_batchs=[]
        y_batchs=[]
        for i in range(start_point_X,end_point_X):
            X_idx=self.X[0:i+self.init_steps]  
            start_point_y=self.init_steps+i
            y_idx=self.y[start_point_y:self.multistep+start_point_y]     
            X_batchs.append(X_idx)
            y_batchs.append(y_idx)
        y_batchs=torch.stack(y_batchs,dim=0)
        # print(f'y_batchs_size is {y_batchs.size()}')
        padded_batchs,mask_batch=self.padding(X_batchs)
        return padded_batchs,mask_batch,y_batchs