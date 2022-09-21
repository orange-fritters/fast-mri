# https://github.com/LISTatSNU/FastMRI_challenge
"""
Train part for fastmri challenge
Modified the above repository
"""

import shutil
import time
from collections import defaultdict

import numpy as np
import torch

from utils.common.loss_function import SSIMLoss
from utils.common.utils import save_reconstructions, ssim_loss
from utils.data.load_data import create_data_loaders
from utils.models.VarNet_RCAN import VarNetRCAN


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    """
    Trains the model for one epoch.
    Args:
        args  (Any)       : arguments (max_key, target_key, input_key, batch_size)
        epoch (int)       : current epoch
        model (nn.Module) : model to train
        data_loader (DataLoader) : data loader for training data
        optimizer   (Optimizer)  : optimizer to use
        loss_type   (Loss)       : loss function to use

    Returns:
        total_loss (torch.Tensor): total train loss
        train_time (float)       : time taken for training
    """
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, grappa, target, maximum, _, _, clear = data

        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        clear = clear.float().cuda(non_blocking=True)
        grappa = grappa.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask, grappa)
        output = output * clear
        target = target * clear

        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            runtime = time.perf_counter() - start_iter
            left_sec = runtime / args.report_interval * (len_loader - iter)
            hour = left_sec // 3600
            minute = (left_sec - left_sec // 3600 * 3600) // 60
            print(
                f'Epoch=[{epoch + 1:2d}/{args.num_epochs:2d}] '
                f'Iter=[{iter:4d}/{len(data_loader):4d}] '
                f'Loss[Batch/Train]= {loss.item():.3f}/{total_loss / (iter + 1):3f} '
                f'Time= {int(runtime)}s '
                f'ETC={int(hour)}H {int(minute)}M '
            )
            start_iter = time.perf_counter()
    total_loss /= len_loader

    return total_loss, time.perf_counter() - start_epoch


def initialize(model, data_loader, loss_type):
    """
    initialize the model

    Args:
        model       (nn.Module)  : model to train
        data_loader (DataLoader) : data loader for training data
        loss_type   (Loss)       : loss function to use

    Returns:
        loss (torch.Tensor) : loss value

    """
    model.eval()

    for mask, kspace, grappa, target, maximum, _, _, clear in data_loader:
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        clear = clear.float().cuda(non_blocking=True)
        grappa = grappa.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask, grappa)
        output = output * clear
        target = target * clear

        loss = loss_type(output, target, maximum)
        break

    return loss


def validate(model, data_loader):
    """
    Validates the model on the validation set.
    Args:
        model (nn.Module) : model to validate
        data_loader (DataLoader) : data loader for validation data

    Returns:
        metric_loss (torch.Tensor): total validation loss
        num_subjects (int)        : number of subjects in the validation set
        reconstructions (dict)    : dictionary of reconstructions
        targets (dict)            : dictionary of targets
        None (None)               : None
        time.perf_counter() - start_first (float) : time taken for validation
    """

    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start_first = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, grappa, target, _, fnames, slices, clear = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)

            output = model(kspace, mask, grappa)

            target = target * clear
            clear = clear.float().cuda(non_blocking=True)
            output = output * clear

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )

    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )

    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)

    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start_first


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    """
    Saves the model to disk.
    Args:
        args     (Any) : Arguments
        exp_dir  (Path): Path to the save the model
        epoch    (int) : current epoch
        model     (nn.Module) : model to save
        optimizer (Optimizer) : optimizer to save
        best_val_loss (float) : the best validation loss
        is_new_best (bool)    : whether the current model is the best model
    """
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train(args):
    """
    Trains the model.
    Args:
        args (Any) : Arguments
    """

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    print(f'Current LR : {args.lr}')

    model = VarNetRCAN(args)
    for param in model.KNet.parameters():
        param.requires_grad = False
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

    start_epoch = 0
    best_val_loss = 1

    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, part=False)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, part=False)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch + 1:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        scheduler.step()
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(model, val_loader)

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch + 1:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
