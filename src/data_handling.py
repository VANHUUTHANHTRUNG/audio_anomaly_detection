#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [
    'SingleMachineDataset',
    'get_dataloader',
    'get_dataset'
]

from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import Tensor

from common import DEV_DATA


class SingleMachineDataset(Dataset):
    def __init__(self,
                 data_name: str,
                 machine_type: str,
                 data_parent_dir: Union[str, Path],
                 use_add_data: Optional[bool] = False
                 ) -> None:
        """

        :param data_name: indicate if the data set is used to train or test
        :param machine_type: samples belongs to what machine type
        :param data_parent_dir: parent dir for all machine types
        """
        super().__init__()
        self.data_name = data_name
        self.df_name = 'df_test.pkl' if data_name == 'test' else 'df_train.pkl'
        self.machine_type = machine_type
        self.data_parent_dir = data_parent_dir

        input_df_path = Path.joinpath(data_parent_dir,
                                      machine_type,
                                      'df_test' if data_name == 'test' else 'df_train',
                                      self.df_name)
        self.dataset = self._load_data(input_df_path)
        self.transform = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def _load_data(input_df_path: Union[str, Path]) -> pd.DataFrame:
        """

        :type input_df_path: specify location of pre-read and saved df (in pickle format)
        """
        dataset = pd.read_pickle(input_df_path)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self,
                    item: int) -> Union[Tuple[Tensor, Tensor, Tensor],
                                        Tuple[Tensor, str, str]]:
        if self.data_name == 'train':
            # return triplet of df read from df_train.pkl
            anchor_df = self.dataset.iloc[item]
            positive_df = self.dataset.loc[(self.dataset['section'] == anchor_df['section']) &
                                           (self.dataset['id'] != anchor_df['id'])].sample()
            negative_df = self.dataset.loc[self.dataset['section'] != anchor_df['section']].sample()

            anchor = Tensor(anchor_df['features'])
            positive = Tensor(positive_df['features'].values[0].astype(np.float64))
            negative = Tensor(negative_df['features'].values[0].astype(np.float64))
            return anchor, positive, negative

        elif self.data_name == 'test':
            # return only one row from df_test.pkl, use section to randomly pick from train data later
            test_df = self.dataset.iloc[item]
            return Tensor(test_df['features']), test_df['section'], test_df['label']
        else:
            raise Exception('data name can only be either train or test')

    def sample_to_infer(self) -> pd.DataFrame:
        """ return a list of samples from source sections which will be used to calculate the similarity to the input

        :rtype: pd.DataFrame
        """
        if self.data_name == 'test':
            raise Exception('sample_to_infer can be used only on train data set')
        else:
            samples = pd.DataFrame([], columns=self.dataset.columns)
            for section in (self.dataset['section'].unique()):
                samples = pd.concat([samples,
                                     self.dataset.loc[self.dataset['section'] == section].sample(10)],
                                    ignore_index=True)
            return samples


def get_dataset(data_parent_dir: Union[str, Path],
                machine_type: str,
                data_name: str,
                use_add_data=False
                ) -> Dataset:
    dataset = SingleMachineDataset(data_name=data_name,
                                   machine_type=machine_type,
                                   data_parent_dir=data_parent_dir,
                                   use_add_data=use_add_data)
    return dataset


def get_dataloader(dataset: SingleMachineDataset,
                   batch_size: int,
                   shuffle: bool,
                   drop_last: bool) -> DataLoader:
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=1)


def main():
    dataset = get_dataset(data_parent_dir=DEV_DATA,
                          machine_type='fan',
                          data_name='train',
                          use_add_data=False)
    print(len(dataset))
    sample_train = dataset.__getitem__(9)
    print(type(sample_train))
    dataset = get_dataset(data_parent_dir=DEV_DATA,
                          machine_type='fan',
                          data_name='test',
                          use_add_data=False)
    print(len(dataset))
    sample_test = dataset.__getitem__(3)
    print(type(sample_test))


if __name__ == '__main__':
    main()

# EOF
