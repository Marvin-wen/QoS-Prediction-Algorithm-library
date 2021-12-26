from abc import get_cache_token
from collections import OrderedDict
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size, self.out_size = in_size, out_size
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_size != self.out_size

# 用来处理short cut
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'dense': nn.Linear(self.in_size, self.out_size),
                # 'bn': nn.BatchNorm1d(self.out_size)

            })) if self.should_apply_shortcut else None

    @property
    def should_apply_shortcut(self):
        return self.in_size != self.out_size

# 来定义一个block
class ResNetBasicBlock(ResNetResidualBlock):
    def __init__(self, in_size, out_size, activation=nn.ReLU):
        super().__init__(in_size, out_size)
        self.blocks = nn.Sequential(
            nn.Linear(self.in_size, self.out_size),
            activation(), 
            nn.Linear(self.out_size, self.out_size),
        )


# 定义一个resnet层，里面会有多个block
class ResNetLayer(nn.Module):
    def __init__(self, in_size, out_size, block=ResNetBasicBlock, n=1,activation=nn.ReLU):
        super().__init__()        
        self.blocks = nn.Sequential(
            block(in_size,out_size,activation),
            *[block(out_size, 
                    out_size,activation) for _ in range(n-1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


# 由多个resnet layer组成encoder
class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by decreasing different layers with increasing features.
    """
    def __init__(self, in_size=128, blocks_sizes=[64,32,16], deepths=[2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Linear(in_size, self.blocks_sizes[0]),
            # nn.BatchNorm1d(self.blocks_sizes[0]),
            activation(),
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        self.blocks = nn.ModuleList([

            *[ResNetLayer(in_size, out_size, n=n, activation=activation, block=block) 
                for (in_size, out_size), n in zip(self.in_out_block_sizes, deepths)]       
        ])


    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    m = ResNetEncoder()
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(m))
