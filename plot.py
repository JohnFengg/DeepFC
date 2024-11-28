#!/usr/bin/env python 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

def plot_learn_curve(prefix,x=None):
    epoch=np.genfromtxt('log/lcurve%s.out'%prefix)[int(x):,0]
    train_loss=np.genfromtxt('log/lcurve%s.out'%prefix)[int(x):,1]
    test_loss=np.genfromtxt('log/lcurve%s.out'%prefix)[int(x):,2]
    err_trn=train_loss[-1]
    err_test=test_loss[-1]

    plt.figure(figsize=(5,5))
    plt.plot(epoch,train_loss,label='train_error',color='blue')
    plt.plot(epoch,test_loss,label='test_error',color='red')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title(f'error_trn:{err_trn:.2e}\nerror_test:{err_test:.2e}')
    plt.tight_layout()
    plt.savefig('%s-error.png'%prefix)

def read_grads(path,index):
    df=pd.read_csv(f'{path}/{index}',sep='\s+',header=None)
    df.columns=['label','Max','Min','Mean']
    return df 

def plot_grads(path,df,index):
    labels=df['label'].values
    Max=df['Max'].values
    Min=df['Min'].values
    Mean=df['Mean'].values
    plt.figure()
    x=np.arange(len(labels))
    width=0.35
    plt.bar(x-width/2,Max,width,label='Max',color='skyblue')
    plt.bar(x+width/2,Min,width,label='Min',color='salmon')
    plt.plot(x,Mean,label='Mean',marker='o',markersize=3,color='green',linestyle='--')
    plt.xticks(x,labels,rotation=90)
    plt.xlabel('Parameters')
    plt.ylabel('Gradient values')
    plt.title(f'Gradients visualizations {index}')
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/grads_{index}.png')

if __name__=="__main__":
    import sys
    prefix=sys.argv[1]
    x=sys.argv[2]
    # x1,x2=sys.argv[2],sys.argv[3]
    plot_learn_curve(prefix,x)

