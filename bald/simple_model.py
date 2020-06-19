from torch import nn

class ConllModel(nn.Module):
    def __init__(self,max_seq_len,emb_dim,num_labels):
        super().__init__()
        
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
        
    def forward(self,x_raw,apply_softmax=False):
        batch_len,max_seq_len,_ = x_raw.size()
        # y_target = y_raw.view(batch_len*max_seq_len)
        
        x = x_raw.unsqueeze(dim=1)
        x = self.cnn(x)
        x = nn.functional.relu(x)
        x = x.view(batch_len,-1)
        x = self.fc(x)
        y_pred = x.view(batch_len*max_seq_len,-1)
        
        if apply_softmax:
            # TODO
            pass
        
        return y_pred
