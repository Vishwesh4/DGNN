'''
Code modified based on https://github.com/mahmoodlab/Patch-GCN
'''
from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch_geometric
from torch.utils.data import Dataset

from .BatchWSI import BatchWSI

class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'path',
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, eps=1e-6,data_dir=None,
        add_pe=False, add_edge_attr=False):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = data_dir
        self.add_pe = add_pe
        self.add_edge_attr = add_edge_attr

        slide_data = pd.read_csv(csv_path, index_col=0, low_memory=False)

        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        
        
        #to bring under same bucket for comparison, some cases dont have good slides due to which no tumorbed is detected, hence
        # no density measure, due to which our model wont work, we exclude those specific cases
        onco_code = Path(csv_path).stem.split("_")[1].upper()
        if "knn" in Path(self.data_dir).stem:
            slidecases_list = list((Path(self.data_dir)).glob("*.pt"))
        else:
            slidecases_list = list((Path(self.data_dir).parent/f"DGNN_graphs/knn_no_sample_{onco_code}").glob("*.pt"))
        slide_avoid = []
        for paths in tqdm(slidecases_list):
            data = torch.load(paths)
            try:
                dens = data.density
            except:
                slide_avoid.append(paths.name)
        if len(slidecases_list)==0:
            print("The case ids maybe different from those used by DGNN. Please check slidecases_list in Generic_WSI_Survival_Dataset class...")
            slidecases_list = list((Path(self.data_dir)).glob("*.pt"))
        slidecases_list = [slide_cases for slide_cases in slidecases_list if slide_cases.name not in slide_avoid]
        # else:
            # slidecases_list = list(Path(self.data_dir).glob("*.pt"))
        print(len(slidecases_list))
        if mode!="cluster": 
            availableslides = ["-".join(slidecases.stem.split("-")[:3]) for slidecases in slidecases_list]
            slide_data = slide_data.loc[slide_data["case_id"].isin(availableslides)]
        else:
            availableslides = [slidecases.stem.split("_cluster")[0] for slidecases in slidecases_list]
            slide_data = slide_data.loc[slide_data["case_id"].isin(availableslides)]

        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs, PatchGCN)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        if not label_col:
            label_col = 'survival'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col
        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        #self.slide_data = slide_data
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        #disc_labels, bins = pd.cut(uncensored_df[label_col], bins=n_bins, right=False, include_lowest=True, labels=np.arange(n_bins), retbins=True)
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:11]
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()


        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None, process=True):
        if process:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True)
        else:
            split = all_splits

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode,
                                  data_dir=self.data_dir, label_col=self.label_col,
                                  patient_dict=self.patient_dict, num_classes=self.num_classes,
                                  add_pe=self.add_pe, add_edge_attr=self.add_edge_attr)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None, train_case_ids=[], val_case_ids=[]):
        if from_id:
            # raise NotImplementedError
            train_split = self.get_split_from_df(all_splits=train_case_ids,process=False)
            val_split = self.get_split_from_df(all_splits=val_case_ids,process=False)
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')

        return train_split, val_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, mode: str='path', add_pe:bool=False, add_edge_attr:bool=False, **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(mode=mode, add_pe=add_pe, add_edge_attr=add_edge_attr,**kwargs)
        self.mode = mode
        self.use_h5 = False
        self.add_pe = add_pe
        self.add_edge_attr = add_edge_attr

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]


        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    if len(list(Path(data_dir).glob("*_featvec.pt"))) > 0:
                        string_comp = "featvec"
                    elif len(list(Path(data_dir).glob("*_graph.pt"))) > 0:
                        string_comp = "graph"
                    else:
                        raise ValueError("Given path does not have appropriate set of files")
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, '{}_{}.pt'.format(slide_id.rstrip('.svs').split(".")[0],string_comp))
                        try:
                            wsi_bag = torch.load(wsi_path)
                        except:
                            print(f"slide not available: {slide_id}")
                            continue
                        if isinstance(wsi_bag,dict):
                            wsi_bag = wsi_bag["feat"]
                        if isinstance(wsi_bag,torch_geometric.data.Data):
                            wsi_bag = wsi_bag.x
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, label, event_time, c, case_id)
                elif self.mode == 'hvt':
                    path_features = []
                    for slide_id in slide_ids:
                        for bag_id in range(2):
                            wsi_path = os.path.join(data_dir, '{}_feature_{}.pt'.format(slide_id.rstrip('.svs').split(".")[0],bag_id))
                            wsi_bag = torch.load(wsi_path)
                            path_features.append(wsi_bag)
                    return (path_features, label, event_time, c, case_id)

                elif self.mode == 'cluster':
                    bag_path = os.path.join(data_dir,'{}_cluster.pt'.format(case_id))
                    bag = torch.load(bag_path)
                    path_features = bag["feat"]
                    cluster_ids = torch.tensor(bag["cluster_ids"])
                    return (path_features, cluster_ids, label, event_time, c, case_id)

                elif self.mode == 'graph':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, '{}_graph.pt'.format(slide_id.rstrip('.svs').split(".")[0]))
                        try:
                            wsi_bag = torch.load(wsi_path)
                            if self.add_edge_attr:
                                dens = wsi_bag.density
                        except:
                            print(f"slide not available: {slide_id}")
                            continue
                        if self.add_pe:
                            wsi_bag.x = wsi_bag.random_walk_pe
                        if self.add_edge_attr:
                            wsi_bag.edge_attr=torch.cat((wsi_bag.density[wsi_bag.edge_index][0,:,:],wsi_bag.density[wsi_bag.edge_index][1,:,:]),dim=-1)
                        if len(wsi_bag.x.shape)==3:
                            wsi_bag.x = wsi_bag.x[:,0,:]
                        path_features.append(wsi_bag)

                    if "edge_latent" in wsi_bag.keys:
                        path_features = BatchWSI.from_data_list(path_features, update_cat_dims={'edge_latent': 1})
                    else:
                        path_features = BatchWSI.from_data_list(path_features, exclude_keys={'random_walk_pe'})
                    return (path_features, label, event_time, c, case_id)
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, data_dir=None,
                  label_col=None, patient_dict=None, num_classes=2, add_pe=False, add_edge_attr=False):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.add_pe = add_pe
        self.add_edge_attr = add_edge_attr
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        
        if os.path.isfile(os.path.join(data_dir, 'fast_cluster_ids.pkl')):
            with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
                self.fname2ids = pickle.load(handle)

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

    def __len__(self):
        return len(self.slide_data)
