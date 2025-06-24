from torch import nn

class GEMM(nn.Module):
    # single linear layer with no bias = gemm
    def __init__(self, in_dim, out_dim):
        super(GEMM, self).__init__()
        self.model = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
    
    def forward(self, x):
        output = self.model(x)
        return output