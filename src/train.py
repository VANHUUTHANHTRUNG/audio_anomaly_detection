# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:23:48 2022

@author: qcirma
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from embedding_model import EmbeddingModel
from data_handling import SingleMachineDataset, get_dataloader, get_dataset
from common import DEV_DATA_MAC
from copy import deepcopy
from auc import auc


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
        infering_samples = train_dataset.sample_to_infer()

        # Save the distances into a list
        distances = []

        # Distance of the features from the sample
        for infering_sample in infering_samples:
            infering_sample_embedded = model(infering_sample)
            distances.append(nn.PairwiseDistance(feature_embedded, infering_sample_embedded))
        
        # Add anomaly scores (as the mean of the distances)
        anomaly_scores.append(np.mean(np.array(distances)))
        ground_truths.append(label)
    
    return auc(np.array(anomaly_scores), np.array(ground_truths))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
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
    epochs = 100

    # Instantiate our DNN
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


    model = model.to(device)
    
    # Load data
    train_dataset = get_dataset(DEV_DATA_MAC, 'fan', 'train')
    test_dataset = get_dataset(DEV_DATA_MAC, 'fan', 'test')
    train_iterator = get_dataloader(train_dataset, 4, False, False)
    test_iterator = get_dataloader(test_dataset, 1, True, False)
    
    # Loss function & optimizer
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        for batch in train_iterator:
            optimizer.zero_grad()

            anchor, positive, negative = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_embedded = model(anchor)
            positive_embedded = model(positive)
            negative_embedded = model(negative)

            loss = triplet_loss(anchor_embedded, positive_embedded, negative_embedded)
            loss.backward()
            # print(f"{epoch}, {loss.item()}")

    trained_model = deepcopy(model)
    anomaly_threshold = infer(device, model, train_dataset, test_iterator)
    print("Anomaly threshold: ", anomaly_threshold)

if __name__ == '__main__':
    main()
