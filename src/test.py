#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [

]

from auc import auc
from data_handling import SingleMachineDataset
from train import EmbeddingModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import PairwiseDistance
from common import CHECKPOINT, get_files_from_dir_with_pathlib


def infer(device: str, model: EmbeddingModel, train_dataset: SingleMachineDataset, test_iterator: DataLoader):
    """
    Using random training samples as reference, calculate the anomaly score and return
    the best threshold for anomaly detection

    :param device: str,
        The device used to train the model (Either CPU or GPU)
    :param model: EmbeddingModel,
        Model used for embedding
    :param train_dataset: SingleMachineDataset,
        Training dataset of a particular machine
    :param test_iterator: DataLoader,
        Dataloader of testing dataset of a particular machine
    """
    anomaly_scores = []
    ground_truths = []

    for i, batch in test_iterator:
        feature, label = batch
        feature.to(device)

        # Embed the feature
        feature_embedded = model(feature)

        # Samples to infer
        inferring_samples = train_dataset.sample_to_infer()

        # Save the distances into a list
        distances = []

        # Distance of the features from the sample
        for inferring_sample in inferring_samples:
            inferring_sample_embedded = model(inferring_sample)
            distances.append(PairwiseDistance(feature_embedded, inferring_sample_embedded))

        # Add anomaly scores (as the mean of the distances)
        anomaly_scores.append(np.mean(np.array(distances)))
        ground_truths.append(label)

    return auc(np.array(anomaly_scores), np.array(ground_truths))


def main():
    newest_checkpoint_filename = get_files_from_dir_with_pathlib(CHECKPOINT)
    checkpoint = torch.load(newest_checkpoint_filename)
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

    learning_rate = 0.1
    epochs = 10

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

    model.load_state_dict(checkpoint)
    model.eval()



if __name__ == '__main__':
    main()

# EOF
