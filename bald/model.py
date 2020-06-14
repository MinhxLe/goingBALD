from torch import nn

class TestModel(nn.Module):

    def __init__(self,):
        super(TestModel,self).__init__()

        Cin = 1
        Cout = 5
        Hin = 300


        self.cnn = nn.Conv2d(
            in_channels = 1,
            out_channels = 
            )

        self.dropout_p = dropout_p
        self.num_directions = num_directions
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=emb_dim,
                                padding_idx=padding_idx)
        if num_directions == 1:
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=h_dim,
                bidirectional=False,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=h_dim,
                bidirectional=True,
                num_layers=num_layers,
                batch_first=True
            )
        
        self.new_dim = num_layers*num_directions*h_dim
        self.fc = nn.Linear(in_features=self.new_dim, out_features=out_dim)

    def forward(self, x_pad, x_len, dropout=True, apply_sigmoid=False):
        x_emb = self.emb(x_pad)
        x_pack = pack_padded_sequence(x_emb, x_len, batch_first=True, enforce_sorted=False)
        _,h_last = self.rnn(x_pack) 
        h_last = h_last.transpose(0,1).reshape(-1,self.new_dim)
        if dropout:
            h_last = F.dropout(h_last,p=self.dropout_p)
        # maybe add an activation layer here?
        y_out = self.fc(h_last)
        return y_out if not apply_sigmoid else torch.sigmoid(y_out)
