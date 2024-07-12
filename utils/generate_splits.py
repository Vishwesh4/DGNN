"""
Divide data into 5 fold splits stratified based on censorship values
"""
import os
import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

#Generate data similar to patchgcn format
def get_slideloc(x):
    try:
        idx = short_form.index(x)
        return available_slides[idx].name
    except:
        return None

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

project_code = "STAD"
# project_code = "COADREAD"

root_dir = Path(f"/aippmdata/public/TCGA-{project_code}")
available_slides = list((root_dir / "images").rglob("*.svs"))
short_form = [avail.stem.split(".")[0] for avail in available_slides]

clinical_data = pd.read_csv(root_dir / "clinical/clinical.tsv",sep="\t")
slide_data = pd.read_csv(root_dir / "biospecimen/slide.tsv",sep="\t")

clinical_data = clinical_data.merge(slide_data[["case_id","slide_submitter_id"]],how="left",left_on="case_id",right_on="case_id",suffixes=(None,None))
clinical_data = clinical_data[["case_id","case_submitter_id","slide_submitter_id","gender","age_at_index","vital_status","days_to_last_follow_up","project_id","days_to_death"]]
clinical_data.replace("'--",np.NaN,inplace=True)
clinical_data = clinical_data.drop_duplicates()
clinical_data["slide_submitter_id"] = clinical_data["slide_submitter_id"].apply(lambda x: get_slideloc(x))
clinical_data["is_female"] = (clinical_data["gender"]=="female")*1
clinical_data["survival_months"] = clinical_data["days_to_last_follow_up"].copy()
clinical_data.loc[clinical_data["vital_status"]=="Dead","survival_months"] = clinical_data.loc[clinical_data["vital_status"]=="Dead","days_to_death"].values
#to replace cases where there is no survival months reported but days to death is reported
index = clinical_data[clinical_data["survival_months"].isna()].index
clinical_data.loc[clinical_data.index.isin(list(index)),"survival_months"] = clinical_data.loc[clinical_data.index.isin(index),"days_to_last_follow_up"].values
clinical_data["survival_months"] = np.round(clinical_data["survival_months"].astype(float)/30.44,2)
clinical_data["project_id"] = clinical_data["project_id"].apply(lambda x: x.split("-")[-1])
clinical_data["censorship"] = (clinical_data["vital_status"]=="Alive")*1
clinical_data["site"] = clinical_data["case_submitter_id"].apply(lambda x: x.split("-")[1])
clinical_data.drop(["case_id","vital_status","days_to_last_follow_up","days_to_death","gender"],axis=1,inplace=True)
clinical_data.rename(columns={"case_submitter_id":"case_id","slide_submitter_id":"slide_id","age_at_index":"age","project_id":"oncotree_code"},inplace=True)
clinical_data.dropna(inplace=True)
clinical_data.sort_values(by="case_id",inplace=True)
clinical_data.reset_index(drop=True,inplace=True)

seed_torch(1)

slidecases_list = list(Path(f"/localdisk3/ramanav/TCGA_processed/TCGA_MIL_TILgraph/knn_no_sample_{project_code}/").glob("*.pt"))
slide_avoid = []
for paths in slidecases_list:
    data = torch.load(paths)
    try:
        dens = data.density
    except:
        slide_avoid.append(paths.name)

slidecases_list = [slide_cases for slide_cases in slidecases_list if slide_cases.name not in slide_avoid]
availableslides = ["-".join(slidecases.stem.split("-")[:3]) for slidecases in slidecases_list]
clinical_data = clinical_data.loc[clinical_data["case_id"].isin(availableslides)]

temp = clinical_data[["case_id","censorship"]].drop_duplicates(subset="case_id")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
fold_splits = list(kfold.split(temp["case_id"].to_list(), temp["censorship"].values))
clinical_data.to_csv(f"/home/vramanathan/Projects/TCGA_MIL/Patch-GCN/datasets_csv/tcga_{project_code.lower()}_all_clean_new.csv")
save_dir = Path(f"/home/vramanathan/Projects/TCGA_MIL/Patch-GCN/splits/5foldcv/tcga_{project_code.lower()}_new")
Path.mkdir(save_dir,parents=True,exist_ok=True)
for i in range(5):
    train_cases = temp.iloc[fold_splits[i][0]]["case_id"].to_list()
    val_cases = temp.iloc[fold_splits[i][1]]["case_id"].to_list()
    val_cases.extend([np.nan]*(len(train_cases)-len(val_cases)))
    data = pd.DataFrame({'train' : train_cases,'val': val_cases})
    data.to_csv(save_dir/f"splits_{i}.csv")