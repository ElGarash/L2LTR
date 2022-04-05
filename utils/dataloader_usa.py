import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import scipy.io as sio

from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse


class TrainDataloader(Dataset):
    def __init__(self, args):
        
        self.polar = args.polar
        self.img_root = args.dataset_dir

        self.train_list = self.img_root + 'splits/train-19zl.csv'

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        #print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                if self.polar:
                    item1 = self.img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')
                else:
                    item1 = self.img_root + data[0]

                item2 = self.img_root + data[1]

                self.id_list.append([item1, item2, pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)

        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)

    def __getitem__(self, idx):

        
        x = Image.open(self.id_list[idx][1]).convert('RGB')
        x = self.transform(x)
        
        y = Image.open(self.id_list[idx][0]).convert('RGB')
        if self.polar:
            y = self.transform(y)
        else:
            y = self.transform_1(y)

        return x, y

    def __len__(self):
        return len(self.id_list)

class TestDataloader(Dataset):
    def __init__(self, args):
        """Return dictionary of aerials path key and tuple of ground dir path, number of taken aerials and number of taken grounds"""
        grds_root_path = "/kaggle/input"
        aerials_root_path = '/kaggle/input/polar-transformation-0-5000/polar_aerial_images/'

        aerial_dirs = os.listdir(aerials_root_path)
        grd_parts = [f'frame-extraction-{i}-{i+1}' for i in range(0, 19, 2)]
        aerial_files_path = []
        ground_files_path = []

        for grd_part in grd_parts:
            part_path = f'{grds_root_path}/{grd_part}/frames'
            for simple_dir in os.listdir(part_path): 

                if simple_dir in aerial_dirs:
                    aerial_path = os.path.join(aerials_root_path, simple_dir)
                    grd_path = os.path.join(part_path, simple_dir)
                    aerial_dir = sorted(os.listdir(aerial_path))
                    ground_dir = sorted(os.listdir(grd_path))
                    num_ground = len(ground_dir)
                    num_aerial = len(aerial_dir)

                    for i in range(num_aerial):
                        aerial_file_path = os.path.join(aerial_path, aerial_dir[i])
                        grd_file_path = [os.path.join(grd_path, ground_dir[j]) for j in range(i*5, min(i*5+5, num_ground))]
                        aerial_files_path.append(aerial_file_path)
                        ground_files_path.append(grd_file_path)

        i = 0
        while i < len(ground_files_path):
            if len(ground_files_path[i]) < 1:
                del ground_files_path[i]
                del aerial_files_path[i]
            else:
                i +=1


        ground_files_path = [path[0] for path in ground_files_path] # Retrieve the first ground image and drop the remaining

        SELECTED_INDICES_1 =  [210, 211, 212, 290, 305, 319, 330, 355, 400, 505, 740, 800, 840, 900, 870, 935, 960, 965, 967, 990, 1006, 1020, 1095, 1135, 1200, 1204, 1218, 1229, 1297, 1305, 1311, 1355, 1380, 1382, 1497, 1500, 1585, 1595, 1600, 1900, 1960, 1980, 1995, 2020, 2050, 2210, 2220, 2225, 2280, 2400, 2395, 2437, 2545, 2705, 3010, 3025, 3080, 3110, 3235, 3505, 3870, 4400, 4410, 5500, 6010]
        SELECTED_INDICES_2 =  [7536, 7566, 7596, 7616, 7621, 7626, 7641, 7651, 7661, 7676, 7691, 7706, 7711, 7726, 7741, 7751, 7766, 7796, 7816, 7836, 7846, 7906, 7956, 7996, 8156, 8166, 8231, 8321, 8351, 8481, 8596, 8626, 8646, 8691, 8761, 8781, 8836, 8876, 8896, 8916, 8951, 9011, 9121, 9201, 9336, 9346, 9381, 9571, 9586, 9596, 9601, 9666, 9736, 10166, 12461]
        SELECTED_INDICES = SELECTED_INDICES_1 + SELECTED_INDICES_2
        SELECTED_AERIAL_POLAR = [aerial_files_path[index] for index in SELECTED_INDICES]
        SELECTED_GROUND = [ground_files_path[index] for index in SELECTED_INDICES]

        self.polar = args.polar

        self.transform = transforms.Compose(
            [transforms.Resize((args.img_size[0], args.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.transform_1 = transforms.Compose(
            [transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))] )

        self.__cur_test_id = 0  # for training
        self.id_test_list = (SELECTED_AERIAL_POLAR, SELECTED_GROUND)
        self.id_test_idx_list = [index for index in range(len(self.id_test_list[0]))]

        self.test_data_size = len(self.id_test_list[0])
        print(f'Number of polar aerial images {len(self.id_test_list[0])}')
        print(f'Number of ground images {len(self.id_test_list[1])}')
        print('polar', args.polar)



    def __getitem__(self, idx):
        
        x = Image.open(self.id_test_list[1][idx]).convert('RGB')
        
        x = self.transform(x)

        y = Image.open(self.id_test_list[0][idx]).convert('RGB')

        if self.polar:
            y = self.transform(y)
        else:
            y = self.transform_1(y)

        return x, y

    def __len__(self):
        return len(self.id_test_list[0])

