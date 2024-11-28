import torch 
import numpy as np 
from sklearn.model_selection import train_test_split
from HRL_FC_AItools.dataload import Multi_Adaptive_SequenceDataset
from torch.utils.data import DataLoader

class globel_decay_rate():
    def __init__(self,input_x,device=None) -> None:
        if device is not None:
            self.input=torch.Tensor(input_x).to(device)
        else:
            self.input=input_x
    def transform(self):
        """
        x is the voltage curves from time 0 hour to time N hours.
        returns a global decay rate curve for an input voltage curve.
        """
        self.init_volt=self.input[0]
        x=self.input-self.init_volt/self.init_volt
        return x 
    
    def inverse_transform(self,x):
        inversed_x=x*self.init_volt+self.init_volt
        return inversed_x

def relative_decay_rate(x=torch.Tensor,device=None):
    """
    x is the voltage sequcences from time N1 hour to time N2 hours.
    return relavtive decay rate value between N1 and N2 for a sequence piece.
    """
    if x.is_cuda is False:
        x=x.to(device)
    rdr=x[-1]-x[0]
    return rdr 

def indexed_train_test_split(X,y,valid_size=0.3):
    if X.shape[0]!=y.shape[0]:
        raise ValueError(f'The shapes of X and y are mismatched: X with {X.shape} and y with {y.shape}') 
    index=int(len(X)*(1-valid_size))
    if index==len(X):
        X_trn,X_tt=X,[]
        y_trn,y_tt=y,[]
    else:
        X_trn,X_tt=X[:index],X[index:]
        y_trn,y_tt=y[:index],y[index:]
    return index,X_trn,X_tt,y_trn,y_tt

def input_train_test_split(X_desc,y_desc,X_seqs,y_seqs,test_sz):
    length=X_desc.size(0)
    indices=np.arange(length)
    test=int(test_sz*length)
    if test>0:
        np.random.shuffle(indices)
    trn_indices = indices[test:]
    tt_indices = indices[:test] if test > 0 else []
    X_desc_tt = X_desc[tt_indices] if test > 0 else torch.empty(0, *X_desc.size()[1:], dtype=X_desc.dtype, device=X_desc.device)
    y_desc_tt = y_desc[tt_indices] if test > 0 else torch.empty(0, *y_desc.size()[1:], dtype=y_desc.dtype, device=y_desc.device)
    X_seqs_tt = X_seqs[tt_indices] if test > 0 else torch.empty(0, *X_seqs.size()[1:], dtype=X_seqs.dtype, device=X_seqs.device)
    y_seqs_tt = y_seqs[tt_indices] if test > 0 else torch.empty(0, *y_seqs.size()[1:], dtype=y_seqs.dtype, device=y_seqs.device)
    return (
        X_desc[trn_indices],
        y_desc[trn_indices],
        X_seqs[trn_indices],
        y_seqs[trn_indices],
        X_desc_tt,
        y_desc_tt,
        X_seqs_tt,
        y_seqs_tt,
    )
class position_descriptor():
    def __init__(self,device) -> None:
        self.device=device

    def middle_decay_rate(self,x,init_volt):
        """
        x.shape -> (batch_sz,sequcen_size,feature_sz)
        x is the voltage sequcences from time N1 hour to time N2 hours.
        return relavtive decay rate value between N1 and N2 for a sequence piece.
        return x.shape -> (batch_sz,feature_sz)
        """        
        x=x.to(self.device)
        init_volt=init_volt.to(self.device)
        median_val=torch.median(x,dim=1)
        mdr=median_val-init_volt/init_volt
        return mdr
    
    def average_decay_rate(self,x=torch.Tensor,init_volt=torch.float32):
        x=x.to(self.device)
        init_volt=init_volt.to(self.device)
        # print(f'the shape of init_volt is {init_volt.shape}')
        mean_val=torch.mean(x,dim=1).squeeze()
        adr=mean_val-init_volt/init_volt
        # print(f'the value of adr is {adr}')
        return adr
    
    # def outlier_detection(self,x=torch.Tensor):
    #     q1,q3=torch.quantile(x,0.25,dim=1),torch.quantile(x,0.75,dim=1)
    #     iqr=q3-q1
    #     lb,ub=q1-1.5*iqr,q3+1.5*iqr
    #     # print(lb,q1,q3,ub,iqr)
    #     outliers_mask=(x<lb)|(x>ub)
    #     # print(outliers_mask)
    #     num_outliers=outliers_mask.sum(dim=1)
    #     outliers_ratio=num_outliers/x.size(1)
    #     x_processed=torch.where(outliers_mask,torch.tensor(0.0,device=self.device),x)
    #     return outliers_ratio,x_processed
    
    def outlier_detection(self,x=torch.Tensor):
        mean=torch.mean(x)
        outliers_mask=(x>mean+0.1)|(x<mean-0.1)
        num_outliers=outliers_mask.sum(dim=1)
        outliers_ratio=num_outliers/x.size(1)
        x_processed=torch.where(outliers_mask,torch.tensor(0.0,device=self.device),x)
        return outliers_ratio,x_processed

    def quantiles(self,x=torch.Tensor):
        if x.size(1)==1:
            result=x[0]
            """
            output shape->(x.size(1),feature_sz)
            """            
        else:
            q1,q2,q3=torch.quantile(x,0.25,dim=1),torch.quantile(x,0.5,dim=1),torch.quantile(x,0.75,dim=1)
            iqr=q3-q1
            lb,ub=q1-1.5*iqr,q3+1.5*iqr
            result=torch.cat([lb,q1,q2,q3,ub],dim=0)
            """
            output shape->(5,feature_sz)
            """
        return result

    def split_sequeece(self,dimension,x=torch.Tensor):
        total_len=x.size(1)
        segment_len=total_len//dimension
        # print(f'the segment length is {total_len}/{dimension}={segment_len}')
        remanent=total_len%dimension
        if remanent/segment_len>0.5:
            segment_len+=1
        segments=[]
        for i in range(dimension):
            try:
                segments.append(x[:,i:i+segment_len,:])
            except:
                segments.append(x[:,i:,:])
        return segments

    def position_encoding(self,x=torch.Tensor,total_len=int,dimension=int,i=int):
        """
        x.shape -> (1,sequcen_sz,feature_sz)
        return x.shape -> (1,1+5*dimension,feature_sz)
        """
        x=x.to(self.device)
        weight=x.size(1)/total_len
        # outlier_ratio,x_processed=self.outlier_detection(x)
        sequences=self.split_sequeece(dimension,x)
        # features=[outlier_ratio]
        features=[]
        for seq in sequences:
            features.append(self.quantiles(seq))
        result=torch.cat(features,dim=0)*weight
        result=result.to('cpu')
        # with open(f'log/log{i}.descriptor_gen','a') as f:
        #     f.write(f'the weight is {weight}\n')            
        #     f.write(f'the shape for single_descriptor is {result.shape}\n')
        """
        output shape->(dimension*5,feature_sz)
        """
        return result

class process_data():
    def __init__(self,device,scaler,init_steps,pred_steps,total_len,dimension_X,dimension_y):
        self.scaler=scaler
        self.device=device
        self.init_steps=init_steps
        self.pred_steps=pred_steps
        self.total_len=total_len
        self.dimension_X=dimension_X
        self.dimension_y=dimension_y

    def input_gen(self,X,y,prefix,idx_corrector,y_corrector):
        descriptor_gen=position_descriptor(device=self.device)
        X_descriptor_total=[]
        y_descriptor_total=[]
        X_seqs=[]
        y_seqs=[]
        data=Multi_Adaptive_SequenceDataset(X,y,self.init_steps,self.pred_steps,idx_corrector,y_corrector)
        dataset=DataLoader(data,batch_size=1,shuffle=False)
        for X_feature,y_feature,X_seq,y_seq in dataset:
            X_descriptor=descriptor_gen.position_encoding(X_feature,self.total_len,self.dimension_X,prefix)
            y_descriptor=descriptor_gen.position_encoding(y_feature,self.pred_steps,self.dimension_y,prefix)
            X_descriptor_total.append(X_descriptor)
            y_descriptor_total.append(y_descriptor)
            X_seqs.append(X_seq)
            y_seqs.append(y_seq)
            # with open(f'log/log{i}.descriptor_gen','a') as f:
            #     f.write(f'the shape for X,y is {X_feature.shape},{y_feature.shape}\n')
        X_descriptors=torch.stack(X_descriptor_total,dim=0)
        y_descriptors=torch.stack(y_descriptor_total,dim=0)
        X_seqs,y_seqs=torch.cat(X_seqs,dim=0),torch.cat(y_seqs,dim=0)
        # with open(f'log/log{prefix}.descriptor_gen','a') as f:
        #     f.write(f'the shape for X_descriptors,y_descriptors is {X_descriptors.shape},{y_descriptors.shape}\n')
        #     f.write(f'the shape for X_seqs,y_seqs is {X_seqs.shape},{y_seqs.shape}\n')
        return X_descriptors,y_descriptors,X_seqs,y_seqs
    
    def input_data(self,X,y,test_sz,valid_sz,prefix,y_corrector):
        X=self.scaler.transform(X)        
        y[:,:-1]=X[:,:-1]
        # print(f'the shape of X_trn and test before scalering is {X.shape},{y.shape}')
        index,X__,X_vld,y__,y_vld=indexed_train_test_split(X, y,
                                            valid_size=valid_sz)
        # print(f'generating descriptors')
        x_desc,y_desc,x_seqs,y_seqs=self.input_gen(X__,y__,prefix,0,y_corrector)
        if index==len(X):
            vld_X_desc,vld_y_desc,vld_X_seqs,vld_y_seqs=torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([])
        else:
            vld_X_desc,vld_y_desc,vld_X_seqs,vld_y_seqs=self.input_gen(X,y,prefix,index,y_corrector)
        # print(f'splitting train and test data')
        trn_X_desc,trn_y_desc,trn_X_seqs,trn_y_seqs,tt_X_desc,tt_y_desc,tt_X_seqs,tt_y_seqs=input_train_test_split(x_desc,y_desc,x_seqs,y_seqs,test_sz)
        return {"X_feats_trn": trn_X_desc, "X_feats_tt": tt_X_desc,"X_feats_vld": vld_X_desc,
                "y_feats_trn": trn_y_desc, "y_feats_tt": tt_y_desc, "y_feats_vld": vld_y_desc,
                "X_seqs_trn": trn_X_seqs, "X_seqs_tt": tt_X_seqs, "X_seqs_vld": vld_X_seqs,
                "y_seqs_trn": trn_y_seqs, "y_seqs_tt": tt_y_seqs, "y_seqs_vld": vld_y_seqs}

class io():
    def __init__(self):
        pass

    def save_file(path,data):
        for key in data:
            if isinstance(data[key],torch.Tensor):
                data_key=data[key].numpy()
            else:
                data_key=data[key]
            np.save(f'{path}/{key}.npy',data_key)
    def read(path):
        data={
            "X_feats_trn": [], "X_feats_tt": [], "X_feats_vld":[],
            "y_feats_trn": [], "y_feats_tt": [], "y_feats_vld": [],
            "X_seqs_trn": [], "X_seqs_tt": [], "X_seqs_vld": [],
            "y_seqs_trn": [], "y_seqs_tt": [], "y_seqs_vld": []
            }
        for key in data:
            try:
                data[key]=torch.from_numpy(np.load(f'{path}/{key}.npy')).float()
            except:
                continue
        return data     