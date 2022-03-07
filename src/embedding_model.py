#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [
    'EmbeddingModel'
]

from pytorch_model_summary import summary
import torch
from torch import Tensor
from typing import Tuple, Union, Optional
from torch.nn import Module, Conv2d, Flatten, Sequential, BatchNorm2d, MaxPool2d, Dropout2d, Linear
from torch.nn.functional import normalize
from torch.nn.modules.activation import ReLU


class EmbeddingModel(Module):
    def __init__(self,
                 cnn_channels_out_1: int,
                 cnn_kernel_1: Union[Tuple[int], int],
                 cnn_stride_1: Union[Tuple[int], int],
                 cnn_padding_1: Union[Tuple[int], int],
                 pooling_kernel_1: Union[Tuple[int], int],
                 pooling_stride_1: Union[Tuple[int], int],
                 cnn_channels_out_2: int,
                 cnn_kernel_2: Union[Tuple[int], int],
                 cnn_stride_2: Union[Tuple[int], int],
                 cnn_padding_2: Union[Tuple[int], int],
                 pooling_kernel_2: Union[Tuple[int], int],
                 pooling_stride_2: Union[Tuple[int], int],
                 cnn_channels_out_3: int,
                 cnn_kernel_3: Union[Tuple[int], int],
                 cnn_stride_3: Union[Tuple[int], int],
                 cnn_padding_3: Union[Tuple[int], int],
                 pooling_kernel_3: Union[Tuple[int], int],
                 pooling_stride_3: Union[Tuple[int], int],
                 linear_input: int,
                 output_embeddings_length: int,
                 dropout: float) -> None:
        super().__init__()

        self.block_1 = Sequential(
            Conv2d(in_channels=1,
                   out_channels=cnn_channels_out_1,
                   kernel_size=cnn_kernel_1,
                   stride=cnn_stride_1,
                   padding=cnn_padding_1),
            ReLU(),
            BatchNorm2d(cnn_channels_out_1),
            MaxPool2d(kernel_size=pooling_kernel_1,
                      stride=pooling_stride_1),
            Dropout2d(dropout))

        self.block_2 = Sequential(
            Conv2d(in_channels=cnn_channels_out_1,
                   out_channels=cnn_channels_out_2,
                   kernel_size=cnn_kernel_2,
                   stride=cnn_stride_2,
                   padding=cnn_padding_2),
            ReLU(),
            BatchNorm2d(cnn_channels_out_2),
            MaxPool2d(kernel_size=pooling_kernel_2,
                      stride=pooling_stride_2))

        self.block_3 = Sequential(
            Conv2d(in_channels=cnn_channels_out_2,
                   out_channels=cnn_channels_out_3,
                   kernel_size=cnn_kernel_3,
                   stride=cnn_stride_3,
                   padding=cnn_padding_3),
            ReLU(),
            BatchNorm2d(cnn_channels_out_3),
            MaxPool2d(kernel_size=pooling_kernel_3,
                      stride=pooling_stride_3))

        self.flatten = Flatten()
        self.dense = Linear(in_features=linear_input, out_features=output_embeddings_length)

    def forward(self,
                X: Tensor) -> Tensor:
        h = X.float()

        h = self.block_1(h)
        h = self.block_2(h)
        h = self.block_3(h)

        h = self.flatten(h)
        h = self.dense(h)

        h = normalize(h, dim=-1, p=2)

        return h


def main():
    cnn_channels_out_1 = 16
    cnn_kernel_1 = 5
    cnn_stride_1 = 2
    cnn_padding_1 = 2
    pooling_kernel_1 = 3
    pooling_stride_1 = 1

    cnn_channels_out_2 = 32
    cnn_kernel_2 = 5
    cnn_stride_2 = 2
    cnn_padding_2 = 2
    pooling_kernel_2 = 3
    pooling_stride_2 = 1

    cnn_channels_out_3 = 32
    cnn_kernel_3 = 3
    cnn_stride_3 = 1
    cnn_padding_3 = 1
    pooling_kernel_3 = 3
    pooling_stride_3 = 2

    linear_input = 49920
    output_embeddings_length = 4096
    dropout = .25

    model = EmbeddingModel(cnn_channels_out_1=cnn_channels_out_1,
                           cnn_kernel_1=cnn_kernel_1,
                           cnn_stride_1=cnn_stride_1,
                           cnn_padding_1=cnn_padding_1,
                           pooling_kernel_1=pooling_kernel_1,
                           pooling_stride_1=pooling_stride_1,
                           cnn_channels_out_2=cnn_channels_out_2,
                           cnn_kernel_2=cnn_kernel_2,
                           cnn_stride_2=cnn_stride_2,
                           cnn_padding_2=cnn_padding_2,
                           pooling_kernel_2=pooling_kernel_2,
                           pooling_stride_2=pooling_stride_2,
                           cnn_channels_out_3=cnn_channels_out_3,
                           cnn_kernel_3=cnn_kernel_3,
                           cnn_stride_3=cnn_stride_3,
                           cnn_padding_3=cnn_padding_3,
                           pooling_kernel_3=pooling_kernel_3,
                           pooling_stride_3=pooling_stride_3,
                           linear_input=linear_input,
                           output_embeddings_length=output_embeddings_length,
                           dropout=dropout)
    print(summary(model, torch.rand((4, 1, 256, 431))))


if __name__ == '__main__':
    main()

# EOF
