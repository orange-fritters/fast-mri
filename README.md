# https://github.com/facebookresearch/fastMRI

# https://github.com/PatrickTUM
"""
Data Loader for Planet-A Competition
Modified the above repository
"""

"""
Return the Data Loader

    Args:
        data_path (Path|str) : data path,
        transform=False (Bool) : Whether to transform the image,
        shuffle=False (Bool) : Whether to shuffle the batches in loader,
        val=False (Bool) : Whether the loader is used in validation phase,
        batch_size=16 : Batch Size of the loader
    Returns:
        data_loader (torch.utils.data.dataloader) : contains the split points of the image
"""

"""
Saves the reconstructions from a model into h5 files that is appropriate for submission
to the leaderboard.

Args:
    reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
        corresponding reconstructions (of shape num_slices x height x width).
    out_dir (pathlib.Path): Path to the output directory where the reconstructions
        should be saved.
    targets (np.array): target array
    inputs  (np.array): input array
"""