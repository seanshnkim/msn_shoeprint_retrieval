import logging
import sys
import yaml
from tqdm import tqdm
import os
import argparse
from time import strftime, localtime

import numpy as np

import torch
from torch.utils.data import DataLoader

import src.deit as deit
from src.dataset import DefaultDataset

start_time_stamp = strftime("%m-%d_%H%M", localtime())
cur_fname = os.path.basename(__file__).rstrip('.py')
log_save_dir = os.path.join('logs', f'{cur_fname}_{start_time_stamp}.log')
logging.basicConfig(filename=log_save_dir, level=logging.INFO)
logger = logging.getLogger()

def load_pretrained(
    r_path,
    encoder
):
    checkpoint = torch.load(r_path, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    # logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
    #             f'path: {r_path}')

    del checkpoint
    return encoder


def init_model(
    device,
    num_blocks,
    model_name='resnet50',
):
    # -- init model
    encoder = deit.__dict__[model_name]()
    emb_dim = 192 if 'tiny' in model_name else 384 if 'small' in model_name else 768 if 'base' in model_name else 1024 if 'large' in model_name else 1280
    emb_dim *= num_blocks
    encoder.fc = None
    encoder.norm = None

    encoder.to(device)
    # encoder = load_pretrained(
    #     r_path=r_enc_path,
    #     encoder=encoder)

    return encoder


def load_from_path(
    r_path,
    encoder
):
    encoder = load_pretrained(r_path, encoder)
    checkpoint = torch.load(r_path, map_location=None)

    best_acc = None
    if 'best_top1_acc' in checkpoint:
        best_acc = checkpoint['best_top1_acc']

    # epoch = checkpoint['epoch']
    epoch = None

    logger.info(f'read-path: {r_path}')
    
    del checkpoint
    return encoder, epoch, best_acc


def main(args):
    with open(args.fname, 'r') as file:
        cfg = yaml.safe_load(file)

    w_enc_path = args.ckpt_path
    device = args.devices
    num_blocks = 1
    
    encoder = init_model(
        device=device,
        num_blocks=num_blocks,
        model_name='deit_small'
    )    
    # model_name = 'ViT-Large'
    model_name = args.ckpt_path.split('.')[0]
    is_trained = True
    
    if is_trained:
        model_name += '-finetuned'
        
        encoder, start_epoch, best_acc = load_from_path(
            r_path=w_enc_path,
            encoder=encoder)
    else:
        model_name += '-base'

    # DO NOT CHANGE BATCH SIZE OF QUERY
    BATCH_SIZE_QUERY = 1
    BATCH_SIZE_REF = 1

    query_train_dataset = DefaultDataset(cfg, "query_train")
    query_test_dataset = DefaultDataset(cfg, "query_test")
    ref_dataset = DefaultDataset(cfg, "ref_test")
    query_train_dataloader = DataLoader(query_train_dataset, batch_size=BATCH_SIZE_QUERY, shuffle=True)
    query_test_dataloader = DataLoader(query_test_dataset, batch_size=BATCH_SIZE_QUERY, shuffle=True)
    ref_dataloader = DataLoader(ref_dataset, batch_size=BATCH_SIZE_REF, shuffle=True)

    # NUM_QUERY = 300
    # NUM_REF = 1175

    save_dir = f'np_features_{model_name}'
    os.makedirs(os.path.join(save_dir, "query"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "ref"), exist_ok=True)
    
    for query_img, query_idx in tqdm(query_train_dataloader):
        query_idx = query_idx.item()
        save_fname = os.path.join(save_dir, "query", f"{query_idx:05d}.npy")
        if os.path.exists(save_fname):
            continue
        
        query_img = query_img.to(device)
        outputs = encoder.forward_blocks(query_img, num_blocks)
        # detach from cuda:0 and convert to numpy
        outputs = outputs.detach().cpu().numpy()
        np.save(save_fname, outputs)
    
    for query_img, query_idx in tqdm(query_test_dataloader):
        query_idx = query_idx.item()
        save_fname = os.path.join(save_dir, "query", f"{query_idx:05d}.npy")
        if os.path.exists(save_fname):
            continue
        
        query_img = query_img.to(device)
        outputs = encoder.forward_blocks(query_img, num_blocks)
        outputs = outputs.detach().cpu().numpy()
        np.save(save_fname, outputs)
        
    for ref_img, ref_idx in tqdm(ref_dataloader):
        ref_idx = ref_idx.item()
        save_fname = os.path.join(save_dir, "ref", f"{ref_idx:05d}.npy")
        if os.path.exists(save_fname):
            continue
        
        ref_img = ref_img.to(device)
        outputs = encoder.forward_blocks(ref_img, num_blocks)
        outputs = outputs.detach().cpu().numpy()
        np.save(save_fname, outputs)


if __name__ == "__main__":
    '''
    args format example:
    
    "args": [
            "--fname",
            "configs/eval/test_custom.yaml",
            "--devices",
            "cuda:0",
            "--ckpt_path",
            "msn-experiment-1-ep25.pth.tar",
            ]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help="path to config file")
    parser.add_argument("--devices", type=str, help="device to use")
    parser.add_argument("--ckpt_path", type=str, help="path to checkpoint")
    args = parser.parse_args()
    
    main(args)