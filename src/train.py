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
from common import DEV_DATA_MAC, DEV_DATA, CHECKPOINT
from copy import deepcopy
from auc import auc
import time




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
    epochs = 10

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
    train_dataset = get_dataset(DEV_DATA, 'fan', 'train')
    test_dataset = get_dataset(DEV_DATA, 'fan', 'test')
    train_iterator = get_dataloader(dataset=train_dataset,
                                    batch_size=4,
                                    shuffle=False,
                                    drop_last=True)
    test_iterator = get_dataloader(dataset=test_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   drop_last=True)

    # Loss function & optimizer
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("------------------------Start training------------------------")
    # Train the model
    for epoch in range(epochs):
        loss_training = []
        model.train()
        for batch in train_iterator:
            optimizer.zero_grad()

            anchor, positive, negative = batch

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor = torch.reshape(anchor, (4, -1, 256, 431))
            positive = torch.reshape(positive, (4, -1, 256, 431))
            negative = torch.reshape(negative, (4, -1, 256, 431))

            anchor_embedded = model(anchor)
            positive_embedded = model(positive)
            negative_embedded = model(negative)

            loss = triplet_loss(anchor_embedded, positive_embedded, negative_embedded)
            loss.backward()

            optimizer.step()

            loss_training.append(loss.item())

        loss_training_mean = torch.Tensor(loss_training).mean()
        print(f"epoch: {epoch}, loss: {loss_training_mean}")
        print("------------------------------------------")

    file_name = 'state_dict_model' + time.strftime("%Y%m%d-%H%M%S") + '.pt'
    saved_model_path = Path(CHECKPOINT, file_name)
    torch.save(model, saved_model_path)
    print("Saved model in ", saved_model_path)
    # trained_model = deepcopy(model)
    # anomaly_threshold = infer(device, model, train_dataset, test_iterator)
    # print("Anomaly threshold: ", anomaly_threshold)


if __name__ == '__main__':
    main()
