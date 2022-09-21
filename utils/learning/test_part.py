# https://github.com/LISTatSNU/FastMRI_challenge
"""
Data Loader for fastmri challenge
Modified the above repository
"""

import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.models.VarNet_RCAN import VarNetRCAN
from utils.data.load_data import create_data_loaders


def test(model, data_loader):
    """
    Reconstruct the validation image
    Args:
        model       (nn.Module) : model to be used to reconstruct the image
        data_loader (DataLoader): data loader for the validation data

    Returns:
        reconstructions (np.array): reconstructed images
    """

    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for iters, (mask, kspace, grappa, _, _, fnames, slices, _ ) in enumerate(data_loader):
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            grappa = grappa.cuda(non_blocking=True)

            output = model.self_ensemble(kspace, mask, grappa)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

            if iters % 10 == 0:
                print(f'{iters}/{len(data_loader)}: {iters/len(data_loader):.1%}')

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    """
    Forward the validation data
    Args:
        args (Any) : arguments
    """

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    PATH = '../result/RCAN/checkpoints/best_model.pt'

    CKPT = torch.load(PATH, map_location='cpu')
    model = VarNetRCAN(args)
    model.load_state_dict(CKPT['model'])
    model.to(device=device)

    print("Starting...")

    forward_loader = create_data_loaders(data_path=args.data_path,
                                         args=args,
                                         isforward=True,
                                         part=False,)
    reconstructions, inputs = test(model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
