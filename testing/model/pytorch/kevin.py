from torch import nn
import torch
import math

class GEMM(nn.Module):
    # single linear layer with no bias = gemm
    def __init__(self, in_dim, out_dim):
        super(GEMM, self).__init__()
        self.model = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
    
    def forward(self, x):
        output = self.model(x)
        return output
    
class SelfAttentionVanilla(nn.Module):
    def __init__(self):
        super(SelfAttentionVanilla, self).__init__()
    
    # q k v are BNHD
    def forward(self, q, k, v):
        b, n, h, d = q.shape
        # want them to be BHND
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        p_unnormalized = q @ k.transpose(-2, -1) # B H N N
        p_unnormalized = p_unnormalized / math.sqrt(d)
        p_unnormalized = p_unnormalized - torch.max(p_unnormalized, axis=3, keepdim=True)[0]
        attention_weights = nn.functional.softmax(p_unnormalized, dim=3) # softmax over last dim
        o = attention_weights @ v # B H N D
        return o.transpose(1, 2) # B N H D
    
def test_self_attn_vanilla():
    b, n, h, d = (4, 6, 2, 64)
    q = torch.randn((b, n, h, d))
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    o_ref = nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False).transpose(1, 2)
    o = SelfAttentionVanilla()(q, k, v)
    print('o    =', o[0, 0, 0, :8], '...')
    print('o_ref=', o_ref[0, 0, 0, :8], '...')
    assert torch.allclose(o, o_ref, atol=0.0001)
    print('all close!')
    
if __name__ == '__main__':
    test_self_attn_vanilla()



