import torch
import torch.nn as nn


class GhostBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, divider=4, momentum=0.1, track_running_stats=True):
        super().__init__()
        self.divider = divider
        self.batch_list = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=False,
                                                        track_running_stats=track_running_stats)])

        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name='std', param=torch.nn.Parameter(torch.ones(1)))

    def forward(self, x):  # TODO: Check divider range
        chunks = x.chunk(self.divider, 0)
        res = [self.batch_list(x_) for x_ in chunks]
        reshape = torch.cat(res)
        return (reshape + self.bias) * self.std


class GhostBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, divider=4, momentum=0.1, track_running_stats=True):
        super().__init__()
        self.divider = divider
        self.batch_list = nn.ModuleList([nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False,
                                                        track_running_stats=track_running_stats)])

        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name='std', param=torch.nn.Parameter(torch.ones(1)))

    def forward(self, x):  # TODO: Check divider range
        chunks = x.chunk(self.divider, 0)
        res = [self.batch_list(x_) for x_ in chunks]
        reshape = torch.cat(res)
        return (reshape + self.bias) * self.std


class GhostBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-05, divider=4, momentum=0.1, track_running_stats=True):
        super().__init__()
        self.divider = divider
        self.batch_list = nn.ModuleList([nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=False,
                                                        track_running_stats=track_running_stats)])

        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter(name='std', param=torch.nn.Parameter(torch.ones(1)))

    def forward(self, x):  # TODO: Check divider range
        chunks = x.chunk(self.divider, 0)
        res = [self.batch_list(x_) for x_ in chunks]
        reshape = torch.cat(res)
        return (reshape + self.bias) * self.std
