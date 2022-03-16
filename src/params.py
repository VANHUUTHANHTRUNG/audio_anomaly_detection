#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [
    'params',
    'SAMPLED_FEATURE_OPTIONS'
]

params = dict(
    cnn_channels_out_1=256,
    cnn_kernel_1=5,
    cnn_stride_1=2,
    cnn_padding_1=2,
    pooling_kernel_1=3,
    pooling_stride_1=1,

    cnn_channels_out_2=256,
    cnn_kernel_2=5,
    cnn_stride_2=2,
    cnn_padding_2=2,
    pooling_kernel_2=3,
    pooling_stride_2=1,

    cnn_channels_out_3=256,
    cnn_kernel_3=3,
    cnn_stride_3=1,
    cnn_padding_3=1,
    pooling_kernel_3=3,
    pooling_stride_3=2,

    linear_input=399360,
    output_embeddings_length=16,
    dropout=.25
)

SAMPLED_FEATURE_OPTIONS = [5, 8, 10]

# EOF
