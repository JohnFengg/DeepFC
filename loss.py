import torch.nn as nn

class TrendAwareLoss(nn.Module):
    def __init__(self, alpha=1.0,loss_f=nn.MSELoss()):
        super().__init__()
        self.alpha = alpha
        self.loss=loss_f

    def forward(self, y_pred, y_true):
        print(f'the shape of y_pred and y_true is {y_pred.shape} and {y_true.shape}')
        point_loss=self.loss(y_pred, y_true)
        delta_y_pred=y_pred[:,1:,:]-y_pred[:,:-1,:]  
        delta_y_true=y_true[:,1:,:]-y_true[:,:-1,:]  
        print(f'the shape of delta_y_pred and delta_y_true is {delta_y_pred.shape} and {delta_y_true.shape}')
        trend_loss=self.loss(delta_y_pred, delta_y_true)
        total_loss=point_loss+self.alpha*trend_loss
        return total_loss


        