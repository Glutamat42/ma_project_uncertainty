import torch
import torch.nn as nn
import torch.nn.functional as F
import blitz.modules

# noinspection PyUnresolvedReferences
from src.project_enums import EnumHeadTypes  # compatibility with older saves (saved enum instead of string which is imported from here)


class PilotNet(nn.Module):
    """ Inspired by NVidia Pilotnet
    https://arxiv.org/pdf/1604.07316.pdf
    https://arxiv.org/pdf/1704.07911.pdf
    """
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        # flatten
        x = x.view(x.size(0), -1)
        return x


class SimpleRegressionHead(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1, aleatoric=False):
        super(SimpleRegressionHead, self).__init__()
        if aleatoric:
            num_labels *= 2

        self.num_labels = num_labels

        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # linear layer
        return self.linear(x)


class RegressionHead(nn.Module):
    """ Inspired by NVidia Pilotnet
    https://arxiv.org/pdf/1604.07316.pdf
    https://arxiv.org/pdf/1704.07911.pdf
    """

    def __init__(self, dim, num_labels=1, aleatoric=False, advanced_aleatoric=False):
        super(RegressionHead, self).__init__()
        if aleatoric and not advanced_aleatoric:
            num_labels *= 2

        self.aleatoric = aleatoric
        self.advanced_aleatoric = advanced_aleatoric
        self.num_labels = num_labels

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_labels)

        if self.advanced_aleatoric:
            self.fc3_std = nn.Linear(50, 10)
            self.fc4_std = nn.Linear(10, num_labels)

    def forward(self, x):
        x = self.dropout(x)

        x1 = F.elu(self.fc1(x))
        x2 = F.elu(self.fc2(x1))
        x3 = F.elu(self.fc3(x2))
        x = self.fc4(x3)

        if self.advanced_aleatoric:
            x_std = self.fc3_std(x2)
            x_std = self.fc4_std(x_std)
            x = torch.stack([x, x_std]).transpose(1, 0)

        return x


class RegressionHeadBBB(nn.Module):
    """ RegressionHead with bayes by backprop """
    def __init__(self, dim, num_labels=1, aleatoric=False):
        super(RegressionHeadBBB, self).__init__()
        if aleatoric:
            num_labels *= 2

        self.aleatoric = aleatoric

        self.dropout = nn.Dropout(0.5)

        self.fc1 = blitz.modules.BayesianLinear(dim, 100)
        self.fc2 = blitz.modules.BayesianLinear(100, 50)
        self.fc3 = blitz.modules.BayesianLinear(50, 10)
        self.fc4 = blitz.modules.BayesianLinear(10, num_labels)


    def forward(self, x):
        x = self.dropout(x)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        return x


class BackboneHeadWrapper(nn.Module):
    # TODO: clarify avgpool
    def __init__(self, backbone, head, arch, avgpool=False, n_last_blocks=4, debug=False):
        """

        Args:
            backbone: backbone / feature extractor net
            head:
            arch: backbone architecture (eg vit_tiny, vit_small, ..., resnet18, ..., nvidia, ...)
            avgpool: for vit
            n_last_blocks: for vit
        """
        super(BackboneHeadWrapper, self).__init__()
        self.backbone = backbone
        self.head = head
        self.arch = arch
        self.avgpool = avgpool
        self.n_last_blocks = n_last_blocks
        self.debug = debug

    def forward(self, x):
        if "vit" in self.arch:
            intermediate_output = self.backbone.get_intermediate_layers(x, self.n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if self.avgpool:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = self.backbone(x)

        # TODO: improve this to be more useful
        if self.debug:
            print(output.shape)
            show = False
            if self.arch == 'vit_tiny':
                backbone_frame = output.cpu().numpy()[0].reshape(12, -1)  # vit_tiny
                show = True
            if self.arch == 'vit_small':
                backbone_frame = output.cpu().numpy()[0].reshape(16, -1)  # vit_small
                show = True
            elif self.arch == 'nvidia':
                backbone_frame = output.cpu().numpy()[0].reshape(-1, 52)  # nvidia
                show = True
            elif self.arch == 'resnet18':
                backbone_frame = output.cpu().numpy()[0].reshape(-1, 16)  # resnet18
                show = True
            elif self.arch == 'resnet50':
                backbone_frame = output.cpu().numpy()[0].reshape(-1, 16)  # resnet50
                show = True
            if show:
                self.imshow('backbone output', backbone_frame)

        return self.head(output)
