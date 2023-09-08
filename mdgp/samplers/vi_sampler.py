import torch 


class VISampler(torch.nn.Module):

    def __init__(self, variational_distribution):
        super().__init__()
        self.variational_distribution = variational_distribution

    def forward(self, sample_shape: torch.Size = torch.Size([])):
        return self.variational_distribution().rsample(sample_shape=sample_shape)
