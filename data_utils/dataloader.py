# *_*coding:utf-8 *_*
import os
import torch
import numpy as np
from torch.utils.data import Dataset

from pytorch3d.ops import ball_query

def pc_normalize(pc,centroid=None,m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid

    if m is None:        
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))        
    pc = pc / m

    return pc, centroid ,m

def ARPD(pc_full, pc_ds, npoints_abs=1000, npoints_relative = 29, radius=0.05):
    """
        Input:
            pc_full: raw point data, [B, N, 3]
            pc_ds: input points position data, [B, npoints, 3]
        Return:
            arpd: absolute and relateive position data, [B, N, 3 + npoints_relative]
    """
    if pc_full.shape[0] > 100000:
        choice = np.random.choice(pc_full.shape[0], 100000, replace=False)
        pc_full = pc_full[choice]
    pc_full = torch.Tensor(np.expand_dims(pc_full,axis = 0))
    pc_ds = torch.Tensor(np.expand_dims(pc_ds,axis = 0))

    point_bq = ball_query(pc_ds,pc_full,K=npoints_relative,radius = radius)[2][0].numpy() #[30000,16,3]
    idx_zero = np.argwhere((point_bq==[0.0,0.0,0.0]).all(axis=2))
    pc_ds = pc_ds.numpy()[0] #[30000, 3]

    point_bq -= pc_ds[:,None,:]
    for idx in idx_zero:
        dim1,dim2 = idx[0],idx[1]
        point_bq[dim1,dim2] = np.zeros(3)
    ct = np.concatenate((pc_ds,point_bq.reshape(30000,-1)),axis=1)
    return ct

class RibSegDataset(Dataset):
    def __init__(self,root , npoints=30000, split='train', flag_cl = False, flag_arpe = False):
        self.npoints = npoints
        self.root = root
        self.flag_cl = flag_cl
        self.flag_arpe = flag_arpe

        train_ids = set([x for x in os.listdir(self.root+'/train')])
        val_ids = set([x for x in os.listdir(self.root+'/val')])
        test_ids = set([x for x in os.listdir(self.root+'/test')])

        if split == 'trainval':
            self.datapath = [self.root+'/train/'+fn for fn in train_ids]+[self.root+'/val/'+fn for fn in val_ids]
        elif split == 'train':
            self.datapath = [self.root+'/train/'+fn for fn in train_ids]
        elif split == 'val':
            self.datapath = [self.root+'/val/'+fn for fn in val_ids]
        elif split == 'test':
            self.datapath = [self.root+'/test/'+fn for fn in test_ids]
        else:
            self.datapath=[]
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)
    
    def __getitem__(self, index):
        fn = self.datapath[index]
        data = np.load(fn)
        pc = data['ct'].astype(np.float32)
        pc[:,:3], centroid, m = pc_normalize(pc[:,:3])
        
        ct, label = pc[:,:3], pc[:,3]
        
        choice = np.random.choice(ct.shape[0],self.npoints,replace=False)
        ct,label = ct[choice],label[choice]
        
        if self.flag_arpe:
            ct = ARPD(pc[:,:3],ct)

        if self.flag_cl:
            cl = data['cl'].astype(np.float32)
            cl = cl.reshape(-1, 3)
            idx_list = []
            for i in range(24):
                if cl[i * 500][0] != -1:
                    idx_list.append(i)
            idx = []
            for x in idx_list:
                for a in range(x * 500, x * 500 + 500):
                    idx.append(a)
            if idx:
                cl[idx], _, _ = pc_normalize(cl[idx], centroid, m)
            cl = cl.reshape(-1, 500, 3)
            return ct, label, cl
        return ct, label

    def __len__(self):
        return len(self.datapath)
