import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # (Ensure program CUDA order matches actual CUDA order)
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"  # (Use only GPU 2)
# os.environ["WANDB_DISABLED"] = "true"
import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from fslcd_datasets_aug2 import ISIC_few_shot_da,  CropDisease_few_shot_da, Chest_few_shot_da, EuroSAT_few_shot_da#, CropDisease_few_shot_da, Chest_few_shot_da, Pattern_few_shot_da
from utils import *
from run_utils import *
from VtT import run_lora
from fsl_dataset.dataloader import EpisodeSampler, RepeatSampler
from fsl_dataset.dataset import DatasetWithTextLabel


def main():

    # Load config file
    args = get_arguments()
    # if(args.dataset == 'ChestX'):
    #     args.epochs = int(args.epochs/ 2)
    #     args.lr = 4e-4
    #     args.mamba_lr = 1e-3
    # if(args.dataset == 'ISIC'):
    #     args.epochs = int(args.epochs/ 2)
    #     args.lr = 4e-4
    #     args.mamba_lr = 1e-3
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    #print(preprocess.transforms[4])
    clip_model = clip_model.half()
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
    few_shot_params = dict(n_way=args.way, n_support=args.shot, n_query=15)
    if args.dataset == 'ISIC':
        print ("Loading ISIC")
        datamgr = ISIC_few_shot_da.SetDataManager(args.image_size, n_eposide=args.episodes, **few_shot_params)
        test_loader = datamgr.get_data_loader(aug=True)
    elif args.dataset == 'EuroSAT':
        print ("Loading EuroSAT")
        datamgr = EuroSAT_few_shot_da.SetDataManager(args.image_size, n_eposide=args.episodes, **few_shot_params)
        test_loader = datamgr.get_data_loader(aug=True)
    elif args.dataset == 'CropDisease':
        print ("Loading CropDisease")
        datamgr = CropDisease_few_shot_da.SetDataManager(args.image_size, n_eposide=args.episodes, **few_shot_params)
        test_loader = datamgr.get_data_loader(aug=True)
    elif args.dataset == 'ChestX':
        print ("Loading ChestX")
        datamgr = Chest_few_shot_da.SetDataManager(args.image_size, n_eposide=args.episodes, **few_shot_params)
        test_loader = datamgr.get_data_loader(aug=True)
    # elif args.dataset == 'Pattern':
    #     print ("Loading Pattern")
    #     datamgr = Pattern_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
    #     novel_loader = datamgr.get_data_loader(aug=True)


    run_lora(args, clip_model, logit_scale, test_loader)

if __name__ == '__main__':
    main()