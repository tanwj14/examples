import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, nhid):
        super(FNNModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.nhid = nhid
        self.vocab_size = vocab_size
        self.vocab_size1 = vocab_size + 1

        self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, nhid)
        # nn.init.xavier_normal_(self.linear1.weight)
        # self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(nhid, vocab_size+1)
        # self.decoder = nn.Linear(nhid, vocab_size)

    def wrap_input(self, input):
        '''
        Preprocess the input to fit the computation graph of FNNModel
        e.g. input = [[1, 3], 
                      [2, 4]]
             wrapped_input = [
                 [[<PAD>, 1], [<PAD>, 3], 
                 [[1, 2]], [3, 4]]
             ]
        Arguments:
            input: torch tensor with shape [seq_len, batch_size]
        Returns:
            wrapped_input: torch tensor with shape [seq_len, batch_size, model_seq_len]
        '''
        wrapped_input = []
        batch_size = input.shape[1] # Num of col (dimensions)
        context_size = input.shape[0] # Num of rows
        # print("input size:", input.shape)

        for idx in range(0, context_size):

            if idx == self.context_size-1:
                # The last time step needs no padding
                wrapped_input.append(input)
                continue

            valid_tokens = input[0:idx+1, :]
            padding = self.vocab_size * torch.ones([self.context_size - 1 - idx, batch_size], dtype=torch.int32).to(valid_tokens.device)
            # print("padding shape", padding.shape)
            # print("valid tokens", valid_tokens.shape)
            padded_tokens = torch.cat([padding, valid_tokens], dim=0)
            wrapped_input.append(padded_tokens)

        wrapped_input = torch.stack(wrapped_input, dim=0)
        wrapped_input = torch.transpose(wrapped_input, 1, 2)
        return wrapped_input
    
    def forward(self, inputs):
        # print("batch size:", inputs.shape[1])
        # print("context_size:", inputs.shape[0])

        wrapped_input = self.wrap_input(inputs)
        # print("valid? ", wrapped_input.shape[2] == inputs.shape[0])

        embeds = self.embeddings(wrapped_input)
        # print("embeds", embeds.shape)
        embeds = embeds.view(embeds.shape[0], embeds.shape[1], -1)
        # print("embeds", embeds.shape)
        # embeds = self.embeddings(inputs).view((1, -1))
        # embeds = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
        out = F.tanh(self.linear1(embeds))
        out = self.linear2(out)
        out = out.view(-1, self.vocab_size1)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters())
    #     return weight.new_zeros(2, bsz, self.nhid)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
