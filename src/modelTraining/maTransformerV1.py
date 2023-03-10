import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters

default_params = {
    'singlehead_size': 4,
    'readout_strategy': 'mean',
    'embd_kmersize': 4,
    'pooling_val': 4,
    'CNN_filters': 1,
    'CNN_filtersize': 5,
    'CNN_padding': 0,
    'input_channels': 4
}

class AttentionNet(nn.Module): #for the model that uses CNN, RNN (optionally), and MH attention
    def __init__(self, params, device=None, genPAttn=True, reuseWeightsQK=False):
        super(AttentionNet, self).__init__()

        self.SingleHeadSize = params['singlehead_size'] #SingleHeadSize
        self.readout_strategy = params['readout_strategy']
        self.kmerSize = params['embd_kmersize'] 
        self.numCNNfilters = params['CNN_filters']
        self.filterSize = params['CNN_filtersize']
        self.CNNpadding = params['CNN_padding']
        self.device = device
        self.reuseWeightsQK = reuseWeightsQK
        self.numInputChannels = params['input_channels'] #number of channels, one hot encoding
        self.genPAttn = genPAttn
 
        self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters,
                                        kernel_size=self.filterSize, padding=self.CNNpadding, bias=False),
                                        nn.ReLU())
        self.dropout1 = nn.Dropout(p=0.02)

        self.Q = nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) 
        self.K = nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) 
        self.V = nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize)

        #reuse weights between Query (Q) and Key (K)
        if self.reuseWeightsQK:
            self.K.weight = Parameter(self.Q.weight.t())	

        self.RELU = nn.ReLU()

        self.fc3 = nn.Linear(in_features=96, out_features=1)

    def attention(self, query, key, value, dropout=0.0):
        #based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, inputs):
        output = inputs
        output = self.layer1(output)
        output = self.dropout1(output)
        output = output.permute(0,2,1)
        conv_out = output.clone()
        pAttn_concat = torch.Tensor([]).to(self.device)
        attn_concat = torch.Tensor([]).to(self.device)
        query, key, value = self.Q(output), self.K(output), self.V(output)
        attnOut,p_attn = self.attention(query, key, value, dropout=0.2)
        attnOut = self.RELU(attnOut)
        attn_concat = torch.cat((attn_concat,attnOut),dim=2)
        if self.genPAttn:
            pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)
            
        if self.readout_strategy == 'normalize':
            output = output.sum(axis=1)
            output = (output-output.mean())/output.std()

        #output = self.fc3(output)	
        assert not torch.isnan(output).any()
        if self.genPAttn:
            return output, query, key, value, conv_out, pAttn_concat
        else:
            return output