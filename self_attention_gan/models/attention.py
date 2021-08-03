import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

class Self_Attn(nn.Module):
    '''
    self attention layer

    Args:
        nn (father): father class
    '''

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        input:
            x: input feature maps(b, x, w, h)

        Returns:
            out: self attention value + input feature
            attention: b, n, n (n is width*height)
        '''
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # b, (w*h), c
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # b, c, (w*h)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # x, n, n
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # b, c, n

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class eca_layer(nn.Module):
    """
    constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map 
        k_size: Adaptive selection of kernel size 
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global information 
        y = self.avg_pool(x)

        # two different branches of ECA module 
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)