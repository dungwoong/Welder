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

# in cypress, they use L=4 independent gemms of m=n=k size
class BatchedGemm(nn.Module):
    def __init__(self, dim, L=4):
        super(BatchedGemm, self).__init__()
        self.dim = dim
        self.L = L
        self.weight = nn.Parameter(torch.randn((L, dim, dim), dtype=torch.float16))
    
    def forward(self, x):
        # x should be L, dim', dim
        return torch.bmm(x, self.weight)
    
class DualGemm(nn.Module):
    def __init__(self):
        super(DualGemm, self).__init__()
    
    def forward(self, a, b1, b2): # I should just always do this from now on
        return a @ b1 + a @ b2

class GemmAndReduction(nn.Module): # return C=AB, and rowsum(A)
    def __init__(self):
        super(GemmAndReduction, self).__init__()
    
    def forward(self, a, b):
        return a @ b, torch.sum(a, axis=-1)
    
class SelfAttentionVanilla(nn.Module):
    def __init__(self):
        super(SelfAttentionVanilla, self).__init__()
    
    # q k v are BHND. do not add transposes to the computation graph, they will mess up the fusion
    def forward(self, q, k, v):
        d = tuple(q.shape)[-1]
        p_unnormalized = q @ k.transpose(-2, -1) # B H N N
        p_unnormalized = p_unnormalized / math.sqrt(d)
        # p_unnormalized = p_unnormalized - torch.max(p_unnormalized, axis=3, keepdim=True)[0] # max_2 and subtract_3
        attention_weights = nn.functional.softmax(p_unnormalized, dim=3) # max_4, subtract_exp_5
        o = attention_weights @ v # B H N D
        return o

class SelfAttentionEasy(nn.Module):
    def __init__(self):
        super(SelfAttentionEasy, self).__init__()
    
    def forward(self, q, k, v):
        d = tuple(q.shape)[-1]
        p_unnormalized = q @ k # .transpose(-2, -1) # B H N N
        p_unnormalized = torch.exp(p_unnormalized / math.sqrt(d))
        attention_weights = p_unnormalized @ v # B H N D
        l = torch.sum(p_unnormalized, axis=3, keepdim=True)
        o = attention_weights / l # B H N D
        return o


class LayerNormLinear(nn.Module):
    def __init__(self, mnk):
        super(LayerNormLinear, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=mnk, bias=True)
        self.linear = nn.Linear(in_features=mnk, out_features=mnk, bias=False)
    
    def forward(self, x):
        x = self.ln(x)
        x = self.linear(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # learnable scale

    def forward(self, x):
        # x: (batch, ..., dim)
        norm = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        x_normed = x / (norm + self.eps)
        return self.weight * x_normed

class RMSFFNSwiGLUEasy(nn.Module):
    def __init__(self, mnk):
        super(RMSFFNSwiGLUEasy, self).__init__()
        self.rms = RMSNorm(normalized_shape=mnk)
        self.sig = nn.Sigmoid()
        self.l1 = nn.Linear(in_features=mnk, out_features=mnk, bias=False) # bias=false --> Easy
        self.l2 = nn.Linear(in_features=mnk, out_features=mnk, bias=False)
        self.next_linear = nn.Linear(in_features=mnk, out_features=mnk, bias=False) # bias=False --> easy
    
    def swiglu(self, x): # swish should have another parameter to be b * self.sig(param * b)
        a = self.l1(x)
        b = self.l2(x)
        sw = b * self.sig(b)
        return a * sw

    def forward(self, x):
        x = self.rms(x)
        x = self.swiglu(x)
        return self.next_linear(x)

# ---------------------------------------------
# TESTS
# ---------------------------------------------

def test_self_attn_vanilla():
    b, h, n, d = (4, 6, 2, 64)
    q = torch.randn((b, h, n, d))
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    o_ref = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    o = SelfAttentionVanilla()(q, k, v)
    print('o    =', o[0, 0, 0, :8], '...')
    print('o_ref=', o_ref[0, 0, 0, :8], '...')
    assert torch.allclose(o, o_ref, atol=0.0001)
    print('all close!')

def test_self_attn_easy():
    b, h, n, d = (4, 6, 2, 64)
    q = torch.randn((b, h, n, d))
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    o_ref = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    o = SelfAttentionEasy()(q, k, v)
    print('o    =', o[0, 0, 0, :8], '...')
    print('o_ref=', o_ref[0, 0, 0, :8], '...')
    assert torch.allclose(o, o_ref, atol=0.0001)
    print('all close!')

def test_rmsnorm_against_torch():
    torch.manual_seed(0)
    B, T, D = 2, 3, 4
    x = torch.randn(B, T, D)

    # Initialize both with the same weights
    torch_rms = nn.RMSNorm(D, eps=1e-8)
    my_rms = RMSNorm(D)
    with torch.no_grad():
        my_rms.weight.copy_(torch_rms.weight)

    # Compare outputs
    out_my = my_rms(x)
    out_torch = torch_rms(x)

    assert torch.allclose(out_my, out_torch, rtol=1e-5, atol=1e-6)
    print("RMSNorm matches PyTorch's nn.RMSNorm.")
    
if __name__ == '__main__':
    test_rmsnorm_against_torch()



