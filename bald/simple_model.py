import torch
from torch import nn
import torch.nn.functional as F

class ConllModel(nn.Module):
    def __init__(self,max_seq_len,emb_dim,num_labels):
        super().__init__()

        self.num_labels = num_labels
        
        cnn_kwargs = {
            "in_channels":1,
            "out_channels":16,
            "kernel_size":(3,emb_dim),
            "padding":(1,0),
        }        
        self.cnn = nn.Conv2d(**cnn_kwargs)
        
        h_out = self.compute_output_dim(
            h_in = max_seq_len,
            padding = cnn_kwargs["padding"][0],
            kernel_size = cnn_kwargs["kernel_size"][0],
            )
        w_out = self.compute_output_dim(
            h_in = emb_dim,
            padding = cnn_kwargs["padding"][1],
            kernel_size = cnn_kwargs["kernel_size"][1],
            )
        last_dim = h_out*w_out*cnn_kwargs["out_channels"]
        self.fc = nn.Linear(last_dim,num_labels*max_seq_len)

    def compute_output_dim(self,h_in,padding,kernel_size):
        """
        Given input dim, padding, kernel size
        compute output dim
        (assumes stride=1,dilation=1)
        """
        out = h_in
        out += 2*padding
        out += -kernel_size
        out += 1
        return out
        
    def forward(self,x_raw,apply_softmax=False,verbose=False):
        batch_len,max_seq_len,_ = x_raw.size()
        
        x = x_raw.unsqueeze(dim=1)
        x = self.cnn(x)
        x = F.relu(x)
        x = x.view(batch_len,-1)
        x = self.fc(x)

        if verbose is True:
            print(f"Batch len is {batch_len}")
            print(f"max_seq_len is {max_seq_len}")
            print(f"num labels is {self.num_labels}")
            print(f"Tensor shape is {x.size()}")
        
        if apply_softmax is True:
            y_pred = x.view(batch_len,max_seq_len,self.num_labels)
            y_pred = F.softmax(y_pred,dim=2)
            y_pred = torch.argmax(y_pred,dim=2)
        else:
            y_pred = x.view(batch_len*max_seq_len,-1)    
        
        return y_pred
