import torch
from torch import nn
import torch.nn.functional as F

class ConllModel(nn.Module):
    def __init__(self,max_seq_len,emb_dim,num_labels):
        super().__init__()
        """
        input tensor: x
            of shape (batch_len,seq_len,emb_dim)
        target tensor: y
            of shape (batch_len,seq_len)

        output of model: y_pred
            of shape (batch_len,seq_len,num_labels)
        """
        self.num_labels = num_labels

        # cnn receives tensor of shape
        # (batch_len,1,seq_len,emb_dim)
        # outputs tensor of shape
        # (batch_len,out_channels,seq_len,1)
        out_channels = 100
        cnn_kwargs = {
            "in_channels":1,
            "out_channels":out_channels,
            "kernel_size":(3,emb_dim),
            "padding":(1,0),
        }        
        self.cnn = nn.Conv2d(**cnn_kwargs)

        # fc receives tensor of shape
        # (batch_len,seq_len,out_channels)
        # outputs tensor of shape
        # (batch_len,seq_len,num_labels)
        last_dim = out_channels
        self.fc = nn.Linear(last_dim,num_labels)
        
    def forward(self,x,mle_prediction=False):
        """
        x: tensor of shape (batch_len,seq_len,emb_dim)

        output: tensor of shape (batch_len,seq_len,num_labels)

        By default, no softmax is applied.

        If mle_prediction is True, a softmax is applied
        and the label with max value is picked.
        output: tensor of shape (batch_len,seq_len)

        """
        batch_len,seq_len,_ = x.size()
        
        x = x.unsqueeze(dim=1)
        x = self.cnn(x)
        x = F.relu(x)
        x = x.squeeze(dim=3)
        x = x.view(batch_len,seq_len,-1)
        x = self.fc(x)
        
        if mle_prediction is True:
            y_pred = F.softmax(y_pred,dim=2)
            y_pred = torch.argmax(y_pred,dim=2)
        else:
            y_pred = x.view(batch_len*seq_len,-1)    
        
        return y_pred
