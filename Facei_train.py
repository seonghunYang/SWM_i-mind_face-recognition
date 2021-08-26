
import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from backbones import get_model
import losses
from dataset import MXFaceDataset, SyntheticDataset, ECFaceDataset ,DataLoaderX

from partial_fc import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


def main(args):
    cfg = get_config(args.config)
    world_size = 1
    rank = 0
    dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)
        
    local_rank = 0
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    if cfg.rec == "ECF":
        train_set = ECFaceDataset(root_dir=cfg.rec)
    elif cfg.rec == "synthetic":
        train_set = SyntheticDataset(local_rank=local_rank)
    else:
        train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    if cfg.rec == "ECF":
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoaderX(
            local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
            sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)

    if cfg.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            if rank == 0:
                logging.info("resume fail, backbone init successfully!")

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])

    backbone.train()
    margin_softmax = losses.get_loss(cfg.loss)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='facei ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    main(parser.parse_args())