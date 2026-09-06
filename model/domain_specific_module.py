import torch.nn as nn
import torch

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)  #torch.split（tensor，split_size_or_sections，dim=1）
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class DomainModule(nn.Module):
    def __init__(self, num_domains, **kwargs):
        super(DomainModule, self).__init__()
        self.num_domains = num_domains
        self.domain = 0

    def set_domain(self, domain=0):
        assert(domain < self.num_domains), "The domain id exceeds the range (%d vs. %d)" % (domain, self.num_domains)
        self.domain = domain

class BatchNormDomain(DomainModule):
    def __init__(self, in_size, num_domains, norm_layer, **kwargs):
        super(BatchNormDomain, self).__init__(num_domains)
        self.bn_domain = nn.ModuleDict()
        for n in range(self.num_domains):
            self.bn_domain[str(n)] = norm_layer(in_size, **kwargs)

    def forward(self, x):
        out = self.bn_domain[str(self.domain)](x)
        return out

