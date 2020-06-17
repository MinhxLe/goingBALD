from torch import nn

class ConllModel(nn.Module):
    def __init__(self,emb_dim,num_labels):
        super().__init__()
        
        self.cnn = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = (3,emb_dim),
            padding = (1,0),
        )
        self.fc = nn.Linear(last_dim,num_labels)
        
    def forward(self,x_batch,y_batch):
        batch_len,seq_len,emb_dim = x_batch.size()
        x_batch = x_batch.unsqueeze(dim=1)
        x = self.cnn(x_batch)
        x = nn.functional.relu(x)
        x = x.view(batch_len*seq_len,-1)
        y_pred = self.fc(x)

        if apply_softmax:
            y_pred = 