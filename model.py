import torch
import torch.nn as nn

ndim = 256


class MS2VEC(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.local_peak_agg = self._make_cnnblock(in_channels=1, out_channels=32, kernel_size=3, dilation=1, nlayers=2)

        self.fragment_block1 = self._make_cnnblock(in_channels=32, out_channels=64, kernel_size=3, dilation=2, nlayers=2)

        self.fragment_block2 = self._make_cnnblock(in_channels=64, out_channels=128, kernel_size=3, dilation=4, nlayers=2)

        self.fragment_block3 = self._make_cnnblock(in_channels=128, out_channels=128, kernel_size=3, dilation=8, nlayers=2)

        self.fragment_block4 = self._make_cnnblock(in_channels=128, out_channels=128, kernel_size=3, dilation=16, nlayers=2)

        self.fragment_block5 = self._make_cnnblock(in_channels=128, out_channels=128, kernel_size=3, dilation=32, nlayers=2)

        self.fragment_block6 = self._make_cnnblock(in_channels=128, out_channels=128, kernel_size=3, dilation=64, nlayers=2)

        self.pointwise_conv = nn.Conv1d(in_channels=128, out_channels=ndim, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

        self.numheads = 8
        self.global_pooling = nn.Linear(ndim//self.numheads, 1)
        self.alpha = nn.Parameter(torch.randn(self.numheads, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn((2449, self.numheads, 1)), requires_grad=True)

        self.MLPBlock = nn.Sequential(
                        nn.Linear(ndim, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Linear(512, 128)
                        )

    def _make_cnnblock(self, in_channels, out_channels, kernel_size, dilation, nlayers):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation*int((kernel_size-1)/2), dilation=dilation))
        layers.append(nn.SELU())

        for _ in range(nlayers - 1):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation*int((kernel_size-1)/2), dilation=dilation))
            layers.append(nn.SELU())

        return nn.Sequential(*layers)

    def forward(self, fragment_feature):
        fragment_feature = fragment_feature.reshape(-1, 1, fragment_feature.shape[-1])
        fragment_feature = self.local_peak_agg(fragment_feature)

        fragment_embedding = self.fragment_block1(fragment_feature)
        fragment_embedding = self.fragment_block2(fragment_embedding)
        fragment_embedding = self.fragment_block3(fragment_embedding)

        fragment_embedding = self.fragment_block4(fragment_embedding)
        fragment_embedding = self.fragment_block5(fragment_embedding)
        fragment_embedding = self.fragment_block6(fragment_embedding)
        fragment_embedding = self.pointwise_conv(fragment_embedding)

        fragment_embedding = multhead_trans(fragment_embedding, self.numheads)

        pooling_weight = self.softmax(self.alpha * self.global_pooling(fragment_embedding) + self.bias)
        fragment_embedding = torch.sum(pooling_weight * fragment_embedding, dim=1)
        fragment_embedding = fragment_embedding.reshape(fragment_embedding.shape[0], -1)

        fragment_embedding = self.MLPBlock(fragment_embedding)
        reg_loss = torch.sum(torch.square(fragment_embedding)) / fragment_embedding.shape[0]
        return fragment_embedding, reg_loss


def multhead_trans(X, num_heads):
    X = X.reshape(X.shape[0], num_heads, -1, X.shape[2])
    X = X.permute(0, 3, 1, 2)
    return X
