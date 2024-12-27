import os
import sys
import numpy as np
import torch
from torch_geometric.data import DataLoader, Batch
from prot_learn.models.drugclip.unimol.tasks.drugclip import DrugCLIP
from prot_learn.models.drugclip.unimol.losses.cross_entropy import IBSLoss
import warnings

import logging
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
from unicore.optim import UnicoreOptimizer
import argparse


warnings.filterwarnings('ignore')


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("unimol.inference")


def main(args):
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    '''
    Pre-train
    '''
    task = tasks.setup_task(args)
    
    # Build model
    model = task.build_model(args)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Set up optimizer
    optimizer = UnicoreOptimizer(args)

    
    loss_fn = IBSLoss(task)

    model = task.train_model(model, optimizer, loss_fn, args)
    exit(0)
    '''
    Test
    '''

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)

    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Test on PCBA dataset
    print("\nTesting on PCBA dataset:")
    dataset_name = args.test_task.lower()
    if(dataset_name == 'pcba'):
        auc_list = []
        ef_list = []
        bedroc_list = []

        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }

        targets = os.listdir("../DrugCLIP/data/lit_pcba/")
        
        for target in targets:
            mol_data_path = "../DrugCLIP/data/lit_pcba/" + target + "/mols.lmdb"
            mol_dataset = task.load_mols_dataset(mol_data_path, "atoms", "coordinates")
            num_data = len(mol_dataset)

            batch_size = 64
            print(num_data//batch_size)

            mol_dataloader = torch.utils.data.DataLoader(mol_dataset, batch_size=batch_size, collate_fn=mol_dataset.collater)

            pocket_data_path = "../DrugCLIP/data/lit_pcba/" + target + "/pockets.lmdb"
            pocket_dataset = task.load_pockets_dataset(pocket_data_path)
            pocket_dataloader = torch.utils.data.DataLoader(pocket_dataset, batch_size=batch_size, collate_fn=pocket_dataset.collater)

            
            auc, bedroc, ef, re = task.predict(
                task=dataset_name,
                mol_dataloader=mol_dataloader,
                pocket_dataloader=pocket_dataloader,
                model=model
            )

            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            # print("re", re)
            # print("ef", ef)
            for key in re:
                re_list[key].append(re[key])

            print(auc_list)
        print(ef_list)
        print("auc 25%", np.percentile(auc_list, 25))
        print("auc 50%", np.percentile(auc_list, 50))
        print("auc 75%", np.percentile(auc_list, 75))
        print("auc mean", np.mean(auc_list))
        print("bedroc 25%", np.percentile(bedroc_list, 25))
        print("bedroc 50%", np.percentile(bedroc_list, 50))
        print("bedroc 75%", np.percentile(bedroc_list, 75))
        print("bedroc mean", np.mean(bedroc_list))
        #print(np.median(auc_list))
        #print(np.median(ef_list))
        for key in ef_list:
            print("ef", key, "25%", np.percentile(ef_list[key], 25))
            print("ef",key, "50%", np.percentile(ef_list[key], 50))
            print("ef",key, "75%", np.percentile(ef_list[key], 75))
            print("ef",key, "mean", np.mean(ef_list[key]))
        for key in re_list:
            print("re",key, "25%", np.percentile(re_list[key], 25))
            print("re",key, "50%", np.percentile(re_list[key], 50))
            print("re",key, "75%", np.percentile(re_list[key], 75))
            print("re",key, "mean", np.mean(re_list[key]))
    elif(dataset_name == 'dude'):
        print("\nTesting on DUD-E dataset:")
        dataset_name = 'dude'
        targets = os.listdir("../DrugCLIP/data/DUD-E/raw/all/")
        
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list= []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }

        for target in targets:
            mol_data_path = "../DrugCLIP/data/DUD-E/raw/all/" + target + "/mols.lmdb"
            mol_dataset = task.load_mols_dataset(mol_data_path, "atoms", "coordinates")
            num_data = len(mol_dataset)

            batch_size = 64
            print(num_data//batch_size)

            mol_dataloader = torch.utils.data.DataLoader(mol_dataset, batch_size=batch_size, collate_fn=mol_dataset.collater)

            pocket_data_path = "../DrugCLIP/data/DUD-E/raw/all/" + target + "/mols.lmdb"
            pocket_dataset = task.load_pockets_dataset(pocket_data_path)
            pocket_dataloader = torch.utils.data.DataLoader(pocket_dataset, batch_size=batch_size, collate_fn=pocket_dataset.collater)

            auc, bedroc, ef, re, res_single, labels = task.predict(
                task=dataset_name,
                mol_dataloader=mol_dataloader,
                pocket_dataloader=pocket_dataloader,
                model=model
            )

            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            res_list.append(res_single)
            labels_list.append(labels)
        res = np.concatenate(res_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        print('DUDE Test Results:')
        print(f"AUC mean: {np.mean(auc_list):.4f}")
        print(f"BEDROC mean: {np.mean(bedroc_list):.4f}")
        for k, v in ef_list.items():
            print(f"EF {k} mean: {np.mean(v):.4f}")

        for k,v  in re_list.items:
            print(f"RE {k} mean: {np.mean(v):.4f}")


def add_args(parser):
    parser.add_argument('--test-task', type=str, choices=['DUDE', 'PCBA'], default='PCBA',
                       help='Test task type')
    parser.add_argument('--clip-norm', type=float, default=1.0,
                       help='Clip threshold for gradient norm')
    parser.add_argument('--update-freq', type=int, default=1,
                       help='Gradient accumulation')
    parser.add_argument('--save-interval', type=int, default=1,
                       help='Save interval')
    parser.add_argument('--max-epoch', type=int, default=200,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=2000,
                       help='Patience for early stopping')
    parser.add_argument('--keep-last-epochs', type=int, default=5,
                       help='Number of last epochs to keep')
    parser.add_argument('--tmp-save-dir', type=str, default='tmp_save_dir',
                       help='Temporary save directory')
    parser.add_argument('--maximize-best-checkpoint-metric', action='store_true',
                       help='Maximize the best checkpoint metric')
    
    parser.add_argument('--save-dir', type=str, default='save_dir',
                       help='save directory')
    parser.add_argument('--master_port', type=int, default=10055)
    parser.add_argument('--data', type=str, default='../DrugCLIP/data/train_no_test_af')
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--best-checkpoint-metric', type=str, default='valid_bedroc',
                       help='Metric to use for saving best checkpoint')
    # parser.add_argument('--finetune-mol-model', type=str, default=None,
    #                    help='Path to pretrained molecular model')
    # parser.add_argument('--finetune-pocket-model', type=str, default=None,
    #                    help='Path to pretrained pocket model')

def cli_main():
    # add args
    
    # parser = argparse.ArgumentParser()
    # add_args(parser)
    # args = parser.parse_args()
    # main(args)
    parser = options.get_validation_parser()
    add_args(parser)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)

if __name__ == '__main__':
    cli_main()