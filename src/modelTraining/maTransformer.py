import numpy as np
import torch 
from torch import nn
from torch.nn.utils.rnn import pad_sequence


# Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask=None, verbose=True):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.
    
    Returns:
        output, attention_weights
    """
    
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    if verbose:
        print('q:', q)
        print('k transpose:', k.transpose(-2, -1))
        print('matmul_qk:', matmul_qk)
    
    
    # scale matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    
    if verbose:
        print('dk:', dk)
        print('scaled_attention_logits:', scaled_attention_logits)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)
    
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights


# class for the transformer model
class maTransformerBlock(nn.Module):

    def __init__(self, filter_protein_size, filter_dna_size, nb_heads, dna_alphabet_size=4, protein_alphabet_size=20, chunk_size=3):

        super(maTransformerBlock, self).__init__()
        self.conv_dna   = nn.Conv2d(1,1, (dna_alphabet_size, filter_dna_size), bias=False)
        self.conv_prot  = nn.Conv2d(1,1, (protein_alphabet_size, filter_protein_size), bias=False)
        self.nb_heads   = nb_heads
        self.chunk_size = chunk_size
        self.linear

    def forward(self, dna, prot):

        # 1 - sequence embedding using convolution
        dna_conv    = self.conv_dna(dna)
        prot_conv   = self.conv_prot(prot)

        # 2 - padd dna_conv and prot_conv to have the same size
        batch_size  = dna_conv.shape[0]
        list_dna    = list(dna_conv.reshape(batch_size, dna_conv.shape[-1]))
        list_prot   = list(prot_conv.reshape(batch_size, prot_conv.shape[-1]))
        padded      = pad_sequence(list_dna+list_prot, batch_first=True, padding_value=0)
        dna_tensor  = padded[:batch_size,:]
        prot_tensor = padded[batch_size:,:]

        # 3 - chunk dna_conv and prot_conv and stack
        # the resulting tensor will have shape (batch size, number chunks, chunk size)
        stacked_d   = torch.stack(torch.chunk(dna_tensor, dna_tensor.shape[1]//self.chunk_size, dim=1))
        stacked_p   = torch.stack(torch.chunk(prot_tensor, prot_tensor.shape[1]//self.chunk_size, dim=1))
        s_d_reshape = stacked_d.reshape(stacked_d.shape[1], stacked_d.shape[0], stacked_d.shape[2])
        s_p_reshape = stacked_p.reshape(stacked_p.shape[1], stacked_p.shape[0], stacked_p.shape[2])

        # 4 - apply dot product attention on concatenated chunks
        output, attention = scaled_dot_product_attention(s_p_reshape, s_d_reshape, s_d_reshape, None)    



# testing the scaled dot product attention
def test_scaled_dot_product_attention():
    np.random.seed(0)
    temp_k = torch.tensor(np.random.rand(1, 3, 3), dtype=torch.float32)  # (..., seq_len_k, depth)
    temp_v = torch.tensor(np.random.rand(1, 3, 3), dtype=torch.float32)  # (..., seq_len_v, depth_v)
    temp_q = torch.tensor(np.random.rand(1, 3, 3), dtype=torch.float32)  # (..., seq_len_q, depth)
    temp_mask = torch.tensor(np.random.rand(1, 3, 3), dtype=torch.float32)  # (..., seq_len_q, seq_len_k)

    print('input is:')
    print('temp_k:', temp_k)
    print('temp_v:', temp_v)
    print('temp_q:', temp_q)
    print('temp_mask:', temp_mask)


    output, attention = scaled_dot_product_attention(temp_q, temp_k, temp_v)
    print('output is:')
    print(output)  # (..., seq_len_q, depth_v)
    print('Attention weights are:')
    print(attention)  # (..., seq_len_q, seq_len_k)



if __name__ == '__main__':
    test_scaled_dot_product_attention()

