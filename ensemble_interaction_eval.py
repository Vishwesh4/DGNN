"""
This script performs ensemble of different tissue interation graphs
"""

from __future__ import print_function

import argparse
import os

import torch
import numpy as np
from sksurv.metrics import concordance_index_censored
from pathlib import Path

### Internal Imports
from Trainer.datasets.dataset_survival import Generic_MIL_Survival_Dataset
from Trainer.datasets.BatchWSI import BatchWSI
from Trainer.utils.core_utils import *
from Trainer.utils.utils import get_custom_exp_code

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

def ensemble_models(risk_predictions, c, event_time):
    # Here just performs ensembling by weighted averaging
    print(risk_predictions, c , event_time)
    temp = risk_predictions[:-1]
    fullmodel = risk_predictions[-1]
    #In case some interaction is missing in a case, however that does not occur if graphs are built carefully
    n_len = len(temp[temp<1])
    return (np.sum(temp[temp<1]) + 4*fullmodel)/(4+n_len)


def model_prediction(model_list, data_list):
    """
    Model prediction for each case across multiple graph interaction model. If no graph then 
    defualted to a very high number which will be filtered out later
    """
    risk_predictions = []
    with torch.no_grad():
        for i in range(len(model_list)):
            if data_list[i] is not None:
                data_WSI = data_list[i]
                data_WSI.x = data_WSI.x.to(device)
                data_WSI.edge_index = data_WSI.edge_index.to(device)
                if data_WSI.edge_attr is not None:
                    data_WSI.edge_attr = data_WSI.edge_attr.to(device)
                hazards, survival, Y_hat, _, _ = model_list[i](x_path=data_WSI)
                risk = -torch.sum(survival, dim=1).cpu().numpy().item()
            else:
                risk = 1000000
            risk_predictions.append(risk)
    return np.array(risk_predictions)

def load_graphs(args, case_id, data_dir):
    """
    Loads multiple interaction graphs for a single case and stores it in a list
    """
    path_features = []
    slide_paths = list(Path(data_dir).glob(f"{case_id}*_graph.pt"))
    if len(slide_paths)>0:
        for wsi_path in slide_paths:
            wsi_bag = torch.load(wsi_path)
            try:
                dens = wsi_bag.density
            except:
                continue
            if args.add_pe:
                wsi_bag.x = wsi_bag.random_walk_pe
            if args.add_edge_attr:
                #try 3 : concat densities 
                wsi_bag.edge_attr=torch.cat((wsi_bag.density[wsi_bag.edge_index][0,:,:],wsi_bag.density[wsi_bag.edge_index][1,:,:]),dim=-1)
            path_features.append(wsi_bag)
        path_features = BatchWSI.from_data_list(path_features)
    else:
        path_features = None
    return path_features

def single_fold_computation(args, val_loader, model_list, data_dir_list):
    """
    Calculates cindex for single fold
    """
    all_risk_scores = np.zeros((len(val_loader)))
    all_censorships = np.zeros((len(val_loader)))
    all_event_times = np.zeros((len(val_loader)))
    for batch_idx, (data_WSI, label, event_time, c, case_id) in enumerate(val_loader):
        data_graphs = []
        for data_dirs in data_dir_list:
            data_graphs.append(load_graphs(args, case_id[0], data_dirs))

        label = label.to(device)
        risk_predictions = model_prediction(model_list, data_graphs)
        risk = ensemble_models(risk_predictions, c.item(), event_time)
        if np.isnan(risk):
            print("all risks are empty which may lead to wrong concordance")
            continue
        event_time = event_time.item()
        c = c.item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return c_index

def get_args():
    parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
    ### Checkpoint + Misc. Pathing Parameters
    parser.add_argument('--data_root_dir', type=str, default='./dataset/TCGA_processed/knn_no_sample_BRCA/', help='data directory')
    parser.add_argument('--seed',            type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k',               type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--k_start',         type=int, default=-1, help='Start fold (Default: -1, last fold)')
    parser.add_argument('--k_end',           type=int, default=-1, help='End fold (Default: -1, first fold)')
    parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')
    parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
    parser.add_argument('--split_dir',       type=str, default='tcga_brca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
    parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
    parser.add_argument('--overwrite',       action='store_true', default=True, help='Whether or not to overwrite experiments (if already ran)')
    parser.add_argument('--testing',         action='store_true', default=False, help='debugging tool')
    parser.add_argument('--auxillary_training', action='store_true', default=False, help='If want to load for pretrained model')

    ### Model Parameters.
    parser.add_argument('--model_type',      type=str, choices=['deepset', 'amil', 'mifcn', 'dgc', 'patchgcn', 'tmil', 'dgnn','hvtsurv', 'gtn_orig'], default='gtn', help='Type of model (Default: mcat)')
    parser.add_argument('--model_name',     type=str, default="dgnn", help="dgnn model name")
    parser.add_argument('--mode',            type=str, choices=['path', 'cluster', 'graph','hvt'], default='graph', help='Specifies which modalities to use / collate function in dataloader.')
    # parser.add_argument('--num_gcn_layers',  type=int, default=4, help = '# of GCN layers to use.')
    parser.add_argument('--num_gcn_layers',  type=int, default=2, help = '# of GCN layers to use.')
    parser.add_argument('--nystrom_heads',  type=int, default=1, help = '# of nystrom heads to use.')
    parser.add_argument('--nystrom_landmarks',  type=int, default=128, help = '# of nystrom landmarks to use.')
    parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.")
    parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')
    parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
    parser.add_argument('--hidden_dim',        type=int, default=128, help = '# features for hidden layers')
    parser.add_argument('--add_pe',        action='store_true', default=True, help='Enable postional encoding')
    parser.add_argument('--add_edge_attr',        action='store_true', default=True, help='Enable edge attribute')

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
    return args

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    args = get_custom_exp_code(args)
    args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
    print("Experiment Name:", args.exp_code)

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
                                        #    data_dir= os.path.join(args.data_root_dir, study_dir),
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
    args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
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
                'weighted_sample': args.weighted_sample,
                'gc': args.gc,
                'opt': args.opt}
    print('\nLoad Dataset')

    model_dict = {'n_classes': args.n_classes, 'num_layers': args.num_gcn_layers,
                        'nystrom_heads': args.nystrom_heads, 'resample': args.resample,
                        'nystrom_landmarks': args.nystrom_landmarks,'hidden_dim':args.hidden_dim,'add_pe':args.add_pe, 'edge_dim': 8}

    settings = ["_tumor_stroma_v2","_tumor_rest_v2","_stroma_rest_v2",""]
    data_dir_list = [f"{args.data_root_dir[:-1]}{setts}/" for setts in settings]

    cancerset_name = args.split_dir.split("_")[-1]
    if cancerset_name=="stad":
        result_dir_list = ["tcga_stad_tumor_stroma",
                        "tcga_stad_tumor_rest",
                        "tcga_stad_stroma_rest",
                        "tcga_stad_complete"]
    elif cancerset_name=="coadread":
        result_dir_list = ["tcga_coadread_tumor_stroma",
                   "tcga_coadread_tumor_rest",
                   "tcga_coadread_stroma_rest",
                   "tcga_coadread_complete"]
    elif cancerset_name=="brca":
        result_dir_list = ["tcga_brca_tumor_stroma",
                            "tcga_brca_tumor_rest",
                        "tcga_brca_stroma_rest",
                        "tcga_brca_complete"]
    elif cancerset_name=="ucec":
        result_dir_list = ["tcga_ucec_tumor_stroma",
                        "tcga_ucec_tumor_rest",
                        "tcga_ucec_stroma_rest",
                        "tcga_ucec_complete"]
    else:
        raise ValueError(f"Dataset {cancerset_name} is not known, please specify the correct dataset")
    c_index_folds = []
    for fold in range(5):
        ### Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, fold))
        print('Fold: {}, training: {}, validation: {}'.format(fold,len(train_dataset), len(val_dataset)))
        val_loader = get_split_loader(val_dataset,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
        model_list = []
        for i in range(len(data_dir_list)):
            model = BaseModel.create(subclass_name=args.model_name,**model_dict)
            state_dict = torch.load(os.path.join(args.results_dir,"5foldcv/GTN_nll_surv_a0.0_5foldcv_gc32/",
                                                result_dir_list[i], "s_{}_checkpoint.pt".format(fold)))
            for key in list(state_dict.keys()):
                state_dict[
                    key.replace("module.", "")
                ] = state_dict.pop(key)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            model_list.append(model)
        c_index = single_fold_computation(args, val_loader, model_list, data_dir_list)
        c_index_folds.append(c_index)
    print(c_index_folds)
    print("C_index: {} ± {}".format(np.array(c_index_folds).mean(),np.array(c_index_folds).std()))

