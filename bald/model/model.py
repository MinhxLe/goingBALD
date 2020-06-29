import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from bald.model.cnn import ConvNet


class CharEncoder(nn.Module):
    """
    char encoder takes in list of list of word represented by array of char idx
    and outputs word embedding
    """
    def __init__(
            self,
            num_chars: int,
            embedding_size: int,
            padding_idx: int,
            embedding_dropout: float,
            channels: List[int],
            kernel_size: int,
            cnn_dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(
            num_chars, embedding_size, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        # TODO other parameters
        self.conv_net = ConvNet(channels, kernel_size, dropout=cnn_dropout)
        self.init_weights()

    def forward(self, inputs):
        # inputs is (batch_size, seq_len)
        seq_len = inputs.size(1)
        # (batch_size, seq_len, embedding)
        embedded_chars = self.embedding_dropout(self.embedding(inputs))
        # (batch_size, embedding, seq_len)
        # we want convolution over the sequence, hence the transpose
        embedded_chars = embedded_chars.transpose(1, 2).contiguous()
        # (batch_size, conv_size, seq_len)
        output = self.conv_net(embedded_chars)
        # maxpool over entire sequence
        # (batch_size, embedding, 1)
        output = F.max_pool1d(output, seq_len)
        # (batch_size, embedding)
        return output.squeeze()

    def init_weights(self):
        nn.init.kaiming_uniform_(
            self.embedding.weight.data, mode="fan_in", nonlinearity='relu')


class WordEncoder(nn.Module):
    def __init__(
            self,
            initial_embedding_weights: torch.Tensor,
            emb_dropout: float,
            channels: List[int],
            kernel_size: int,
            cnn_dropout: float):
        # used for word2vec
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
                initial_embedding_weights, freeze=False)
        self.dropout = nn.Dropout(emb_dropout)
        # TODO, why dilated and not residual
        self.channels = channels
        self.conv_net = ConvNet(channels, kernel_size, dropout=cnn_dropout,
                add_residual=False)

    def forward(self, word_inputs, char_embedding_inputs):
        # word_inputs (batch_size, seq_len)
        # char_embedding_inputs  (batch_size, seq_len, char_embed)

        # (batch_size, sequence_len, seq_len)
        word_embedding = self.embedding(word_inputs)

        # (batch_size, seq_len, word_embed + char_embed)
        embedded = torch.cat((word_embedding, char_embedding_inputs), 2)
        # (batch_size, word_embed + char_embed, seq_len)
        embedded = embedded.transpose(1, 2).contiguous()
        # (batch_size, conv_size, seq_len)
        conv_out = self.conv_net(self.dropout(embedded))

        # (batch_size, conv_size + word_embed + char_embed, seq_len)
        output = torch.cat((conv_out, embedded), 1)

        # (batch_size, seq_len, conv_size + word_embed + char_embed)
        return output.transpose(1, 2).contiguous()


class Decoder(nn.Module):
    """
    LSTM tag decoder
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # TODO batchfirst, dropout?
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, output_dim)
        self.init_weight()

    def forward(self, inputs):
        # input (batch_size, seq_len, input_dim)
        # TODO need to initialize initial states?
        # (batch_size, seq_len, hidden_dim)
        lstm_out, self.hidden = self.lstm(inputs, None)
        # batch_size, seq_len, output_dim
        y = self.hidden2label(lstm_out)
        return y

    def init_weight(self):
        nn.init.kaiming_uniform_(
                self.hidden2label.weight.data,
                mode='fan_in',
                nonlinearity='relu')


class Model(nn.Module):
    def __init__(self, charset_size, char_embedding_size, char_channels,
                 char_padding_idx, char_kernel_size,
                 word_embedding, word_channels,
                 word_kernel_size, num_tag, dropout, emb_dropout):
        super().__init__()
        self.char_encoder = CharEncoder(
                num_chars=charset_size,
                embedding_size=char_embedding_size,
                padding_idx=char_padding_idx,
                embedding_dropout=emb_dropout,
                channels=char_channels,
                kernel_size=char_kernel_size,
                cnn_dropout=dropout)
        self.word_encoder = WordEncoder(
                initial_embedding_weights=word_embedding,
                emb_dropout=emb_dropout,
                channels=word_channels,
                kernel_size=word_kernel_size,
                cnn_dropout=dropout)
        self.drop = nn.Dropout(dropout)
        # used to figure out the encoded size
        self.char_conv_size = char_channels[-1]
        self.word_conv_size = word_channels[-1]
        self.word_embedding_size = word_embedding.size(1)
        # TODO hidden size(?)
        self.decoder = Decoder(self.char_conv_size+self.word_embedding_size+self.word_conv_size,
                               self.char_conv_size + self.word_embedding_size + self.word_conv_size,
                               num_tag, num_layers=1)

    def forward(self, word_input, char_input):
        # word input: (batch size, seq_len)
        # char input: (batch_size, seq len, word len)
        batch_size = word_input.size(0)
        seq_len = word_input.size(1)

        # (batch_size * seq_len, word_len)
        char_input_flattened = char_input.view(-1, char_input.size(2))
        # (batch_size * seq_len, char_conv_size)
        char_encoding = self.char_encoder(char_input_flattened)

        # (batch_size, seq_len, char_conv_size)
        char_encoding = char_encoding.view(batch_size, seq_len, -1)

        #(batch_size, seq_len, char_encode+word_embed+word_conv)
        word_output = self.word_encoder(word_input, char_encoding)
        # (batch_size, seq_len, n_tags)
        y = self.decoder(word_output)
        # TODO do we need this
        return F.log_softmax(y, dim=2)

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)

