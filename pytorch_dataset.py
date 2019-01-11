from torch.utils.data import Dataset
import utils
import pandas as pd
import numpy as np
import torch

# create dataset class
class ProteinDataset(Dataset):
    def __init__(self,dirpath,fnames):
        self.dirpath = dirpath
        # train data labels
        self.labels = pd.read_csv(utils.LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]
        self.fnames=fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self,index):
        filename=self.fnames[index]
        img = utils.open_rgby(self.dirpath, filename, train=True,stats=utils.stats)
        label = self.labels.loc[filename]['Target']
        label = np.eye(len(utils.name_label_dict))[label].sum(axis=0)
        img = torch.from_numpy(img.transpose((2,0,1)))
        label = torch.from_numpy(label).float()
        return img,label
