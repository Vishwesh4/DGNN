'''
Code modified based on https://github.com/mahmoodlab/Patch-GCN
'''
from __future__ import print_function

import time
import argparse
import os
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from Trainer.datasets.dataset_survival import Generic_MIL_Survival_Dataset

from Trainer.utils.file_utils import save_pkl
from Trainer.utils.core_utils import train
from Trainer.utils.utils import get_custom_exp_code

### PyTorch Imports
import torch

def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_val_cindex = []
    folds = np.arange(start, end)

    ### Start 5-Fold CV Evaluation.
    for i in folds:
        start = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue

        ### Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        ### Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            val_latest, cindex_latest = train(datasets, i, args)
            latest_val_cindex.append(cindex_latest)

        ### Write Results for Each Split to PKL
        save_pkl(results_pkl_path, val_latest)
        end = timer()
        print('Fold %d Time: %f seconds' % (i, end - start))

    ### Finish 5-Fold CV Evaluation.
    if args.task_type == 'survival':
        results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'

    results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))
    print(results_latest_df)
    print(f"Final Cross Fold Result: {results_latest_df['val_cindex'].mean()} ± {results_latest_df['val_cindex'].std()}")


### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir', type=str, default='/localdisk3/ramanav/TCGA_processed/DGNN_graphs/knn_no_sample_BRCA', help='data directory')
parser.add_argument('--seed',            type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k',               type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',         type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',           type=int, default=-1, help='End fold (Default: -1, first fold)')
# parser.add_argument('--results_dir',     type=str, default='/home/vramanathan/scratch/amgrp/TCGA_Results/Patch-GCN', help='Results directory (Default: ./results)')
parser.add_argument('--results_dir',     type=str, default='/localdisk3/ramanav/Results/TCGA_Results/5foldcv/', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_brca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',       action='store_true', default=True, help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--testing',         action='store_true', default=False, help='debugging tool')

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['deepset', 'amil', 'mifcn', 'dgc', 'patchgcn', 'tmil', 'dgnn','hvtsurv', 'gtn_orig'], default='dgnn', help='Type of model (Default: mcat)')
parser.add_argument('--model_name',     type=str, default="dgnn", help="dgnn model name")
parser.add_argument('--mode',            type=str, choices=['path','cluster', 'graph','hvt'], default='graph', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--num_gcn_layers',  type=int, default=2, help = '# of GCN layers to use.')
parser.add_argument('--nystrom_heads',  type=int, default=1, help = '# of nystrom heads to use.')
parser.add_argument('--nystrom_landmarks',  type=int, default=128, help = '# of nystrom landmarks to use.')
parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.")
parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--hidden_dim',        type=int, default=128, help = '# features for hidden layers')
parser.add_argument('--add_pe',        action='store_true', default=False, help='Enable postional encoding')
parser.add_argument('--add_edge_attr',        action='store_true', default=False, help='Enable edge attribute')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
#For hvtsurv we assume batch size to be 1
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',              type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg',             type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 512
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_name': args.model_name,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt,
            'num_gcn_layers': args.num_gcn_layers, 
            'nystrom_heads': args.nystrom_heads,  
            'nystrom_landmarks': args.nystrom_landmarks,
            'hidden_dim': args.hidden_dim,
            'add_pe': args.add_pe,
            'add_edge_attr': args.add_edge_attr,
            } 
print('\nLoad Dataset')

if 'survival' in args.task:
    args.n_classes = 4
    study = '_'.join(args.task.split('_')[:2])
    if study == 'tcga_kirc' or study == 'tcga_kirp':
        combined_study = 'tcga_kidney'
    elif study == 'tcga_luad' or study == 'tcga_lusc':
        combined_study = 'tcga_luad'
    else:
        combined_study = study
    study_dir = '%s_20x_features' % combined_study
    dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, combined_study),
                                           mode = args.mode,
                                           data_dir = args.data_root_dir,
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[],
                                           add_pe=args.add_pe,
                                           add_edge_attr=args.add_edge_attr)
else:
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed) + time.strftime("_%d%b_%H_%M_%S", time.localtime()))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

### Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'data_root_dir':args.data_root_dir})
settings.update({'split_dir': args.split_dir})
settings.update({'results_dir': args.results_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))