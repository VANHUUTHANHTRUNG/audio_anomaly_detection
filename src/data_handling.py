#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'cptrva'
__docformat__ = 'reStructuredText'
__all__ = [
    'SingleMachineDataset',
    'get_dataloader',
    'get_dataset'
]

from torch.utils.data import DataLoader, Dataset
from typing import Union, Optional, Tuple, Dict, List
from pathlib import Path
import numpy as np
import pandas as pd
from common import DEV_DATA, ADD_DATA, DEV_DATA_MAC


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
                    item: int) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                        Tuple[np.ndarray, str, str]]:
        if self.data_name == 'train':
            # return triplet of df read from df_train.pkl
            anchor_df = self.dataset.iloc[[item]]
            positive_df = self.dataset.loc[self.dataset['section'] == anchor_df['section'] &
                                           self.dataset['id'] != anchor_df['id']].sample()
            negative_df = self.dataset.loc[self.dataset['section'] != anchor_df['section']].sample()
            return anchor_df['features'], positive_df['features'], negative_df['features']
        elif self.data_name == 'test':
            # return only one row from df_test.pkl, use section to randomly pick from train data later
            test_df = self.dataset.iloc[[item]]
            return test_df['features'], test_df['section'], test_df['label']
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
                   drop_last: bool):
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=1)


def main():
    dataset = get_dataset(data_parent_dir=DEV_DATA,
                          machine_type='fan',
                          data_name='train',
                          use_add_data=False)
    print(len(dataset))


if __name__ == '__main__':
    main()

# EOF
