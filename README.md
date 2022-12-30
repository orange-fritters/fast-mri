## Snippets


# SNU FastMRI Challenge 

### 🏆🏆🏆 **3rd Place** 🏆🏆🏆

The final submitted pytorch code for the **2022 SNU Fastmri Challenge**

## Abstract

Fast MRI is an imaging technology that reduces MRI imaging time by acquiring less data than before.
In particular, various methods have been proposed by solving the aliasing problem caused by low data using
high-performance algorithms in the image reconstruction stage ([[2][3][4]](##Reference) are mainly used by image
reconstruction methods and compressed sensing using multi-phase channel receivers)

Deep Learning has recently been used in a variety of image processing fields and has shown remarkable performance.
Medical imaging is no exception. Recently, research has been actively conducted to reconstruct images of similar quality
to conventional MRI using only a fraction of the data based on Deep Learning [[5]](##Reference). As such, Deep Learning
shows infinite potential for FastMRI.

The goal of this challenge is to implement FastMRI technology using Deep Learning technology.
The MRI image data required for this Challenge will be distributed to Challenge participants,
and the participants aim to obtain high-quality MRI images by configuring their own deep neural network to learn
the data. [[1]](##Reference)

## Model Structure
<img src=./imgs/model_structure.png width="900" height="300" />

## Details

### Model
The final model is shown in the figure above.

First, I developed a strategy to extract images from kspace using
**E2Evarnet**, then concatenate it with grappa images to provide sources that were not enough with the kspace,
and applied Image Super Resolution's methodology in Image Domain to produce final results.
This was also valid in terms of reducing the burden on the network. In the image domain, I used a network called **RCAN**
as a baseline and modified it in my own way to learn and use it. 

The structure of the RCAN is approximate as follows:
Each overlapping Residual Attention Block has a skip connection, and there is also a skip connection between Residual
Groups where blocks are gathered. The Residual in Residual structure above was convenient for optimally adopting
Gradient Checkpointing, which will be described later.

### Memory Saving Technology
#### Parameter Freezing
The methodology I applied to solve the OOM problem of the model (out of memory) is as follows. First, the memory
occupied by the baseline E2E varnet was huge, so we had to **freeze** the varnet and learn the image domain network.

```python
# Parameter freezing Snippet
for param in model.varnet.parameters():
    param.requires_grad = False
```

#### Bottleneck Structure
Depth and efficiency could not be achieved without Residual learning to enhance the efficiency of the network.
In addition, I changed all the residual block of RCAN to **bottleneck structure** to reduce the number of parameters
Also, Res Scaling methodology was applied to stabilize the training process.
```python
# BottleNeack Structure Snippet
class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super(ResBlock, self).__init__()
        mid_feats = int(n_feats / 4)
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, mid_feats, kernel_size=1, padding=0, stride=1),
            nn.PReLU(),
            nn.Conv2d(mid_feats, mid_feats, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.Conv2d(mid_feats, n_feats, kernel_size=1, padding=0, stride=1))
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
```
#### Bucket Sampling
**Bucket sampling** is a method of making batch data sets of different sizes by size.
If the **size of your data varies**, I recommend you to try bucket sampling.
In the first attempt, the problem of putting multiple data in one batch due to different number of coils was forced to
be implemented by bucket sampling to increase the batch size.
However, I read about the futility of Batch Normalization in EDSR papers in the field of Super Resolution, fixed
the batch size to 1, and applied normalization only to the beginning and final phase.
This approach also saved the memory of the network.

#### ReLU inplacing
In addition, to save memory once more, the activation function ReLU was given the inplace option,
saving about 10% of memory.
There were also options such as PReLU(), but I used ReLU because I thought saving memory through inplacing was the key.
```python
# Saving memory by ReLU inplacing 
nn.ReLU(inplace=True)
```

### Gradient Checkpointing
<img src=./imgs/gradient_checkpointing.png width="750" height="350" />

In addition, I applied **Gradient Checkpointing** technique (very powerful).
The above figure shows the specific checkpoints of RCAN.
Gradient Checkpointing technology saves a lot of memory and solves **OOM** by creating multiple checkpoints 
on computational graphs that occur during backpropagation,
and the model size before gradient checkpointing was up to 1 million parameters before the technology was applied,
but this technology allowed us to build a larger image domain network with 30 million parameters.
Therefore, in RCAN, which was the baseline, several Residual Blocks were grouped into Residual Groups,
and checkpoints were placed between them.

When the model had only 1M or 1 million parameters,
it took about two hours per epoch,
and about six hours after applying the technology.
The size increased by 30 times, but I sacrificed only three times more training time.
Finally, 160 feature maps were formed in one Residual Attention Block,
and a total of 100 blocks were learned in 160 feature maps each.

**Tips on Gradient Checkpointing**
1. Devide your model into several groups using Python class (additional tip : RIR! Residual in residual)
2. The number of groups is related to numbers of checkpoints (#groups = #checkpoints - 1)
3. TMI) Some posts recommend applying square root of total layer number as checkpoint number...
4. Place checkpoints at the forward function of your group class

```python
# Grdient Checkpointing Snippets

from torch.utils import checkpoint

# Devided Group Class
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feats, reduction, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [RCAB(n_feats, reduction, res_scale=res_scale) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*modules_body)
        
    def forward(self, x):
        res = checkpoint.checkpoint(self.body, x) ### GRADIENT CHECKPOINTING
        res = res + x
        return res

# Overall Network Class
class RCAN(nn.Module):
def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        # ... skipped a lot of codes
        modules_body = [ResidualGroup() for _ in range(n_resgroups)]
```
### Additional Details

#### num_workers
num_workers option of Pytorch DataLoader is a very useful option for **speeding up training**.
num_workers is the number of processes that load data in parallel.
put num_workers option 4 times as your gpu number, and you can speed up training by 30%, 
Especially your data I/O time is huge.
```python
# num_workers Snippets
data_loader = DataLoader(
    dataset=data_storage,
    batch_size=args.batch_size,
    pin_memory=True,
    num_workers=4 ### NUM_WORKERS OPTION
)
```
#### Attention use in Computer Vision
Attention is a very powerful technology that has been used in many fields.
RCAN was chosen as the baseline because Attention layer lowered weights of features corresponding to low frequency 
out of 160 feature maps and increased its **attention** for high frequency feature maps, which play a crucial role in
SSIM. 

```python
# Channel Attention Layer Snippets
# REFERENCE!!! : https://github.com/sanghyun-son/EDSR-PyTorch 
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
```
#### Self-Ensemble
**Self-Ensemble** is a technique that increases the accuracy by averaging the results of one unique model by 
putting 180-degree rotation, flip, and 180-degree rotation images then reflip the output [[8]](#References)

If you want to use Self-Ensemble with more flip and rotation you can modify the below code a little bit.
If normal square sized image self-ensemble needed, of course you can modify the below code a little bit.

```python
# Self-Ensemble Snippets : self_ensemble method of final model class (use this at the test phase instead of forward or calling)
    def self_ensemble(self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa: torch.Tensor) -> torch.Tensor:
        output = self.KNet(masked_kspace, mask)

        flips = ['original', 'f']
        ensembles = [self._flip(output, grappa, flip=flip) for flip in flips]

        ensembled = []
        for i, (output, grappa) in enumerate(ensembles):
            input = torch.stack((output, grappa))
            img = self.INet(input)
            ensembled.append(self._unflip(img.squeeze(0), flips[i]))
        output = sum(ensembled) / 2
        return output

    @staticmethod
    def _flip(image, grappa, flip):
        if flip == 'original':
            return image, grappa
        elif flip == 'f':
            image_f = torch.flip(image, [2])
            grappa_f = torch.flip(grappa, [2])
            return image_f, grappa_f

    @staticmethod
    def _unflip(image, flip):
        if flip == 'original':
            return image
        elif flip == 'f':
            image_original = torch.flip(image, [2])
            return image_original
```

## Project Structure
```
├──  Code
│    ├── evaluate.py ... evaluation code
│    └── train.py ... train the model
|
├──  data
│    └── val
│       ├── image
│       └── kpsace
│    └── train
│       ├── image
│       └── kpsace
│
├──  utils
│    └── common
│       ├── fastmri ... fastmri library
│       ├── utils.py
│       └── loss_function.py
│    
│    └── data
│       ├── load_data.py
│       └── transform.py
│    
│    └── learning
│       ├── test_part.py
│       └── train_part.py
│
│    └── models
│       ├── common.py
│       ├── RCAN.py
│       ├── unet.py
│       ├── varnet.py
│       └── VarNet_RCAN.py
│
└──  README.md

```


## How to run

### Train
datasets available at https://fastmri.org/dataset/

Modifty the `train.py` file or use arp-parser and run it
At the #result folder, directory of model checkpoint with the name of the network is made
```bash
python train.py --net-name Test
                   --num-epochs 20  
                   --lr 1e-4 
                   --batch-size 1 
                   --n_feats 160
                   --n_resblocks 25
                   --n_resgroups 4
                   --reduction 16
                   --res_scale 0.125
```
### Validation
Modifty the `evaluate.py` file with same args as train or use arp-parser and run it
At the result folder of the network, reconstructions will be saved
```
python evaluate.py --net-name Test
```
### Test
modify the net_name you want to test inside the `test_SSIM.py` file and run it
```python
if __name__ == '__main__':
    NET_NAME = 'TEST'
```

## TODO List
- CutMix augmentation
- Rotation and Flip while the data loading
- Batching with BucketSampler (for the same size of the image)

## Reference
```
[1] SNU FastMRI Challenge. (2022, Sept 21). 2022 SNU FastMRI Challenge, Home, FastMRI? 
http://fastmri.snu.ac.kr/blog-home-1.html
[2] SENSE: Sensitivity encoding for fast MRI, Magnetic Resonance In Medicine, Volume42, Issue5 Pages 952-962
[3] Generalized autocalibrating partially parallel acquisitions (GRAPPA), Magnetic Resonance In Medicine, Volume47, Issue6, Pages 1202-1210
[4] Sparse MRI: The application of compressed sensing for rapid MR imaging, Magnetic Resonance In Medicine, Volume58, Issue6, Pages 1182-1195Deep Learning Grappa
[5] Deep-Learning Methods for Parallel Magnetic Resonance Imaging Reconstruction: A Survey of the Current Approaches, Trends, and Issues, IEEE Signal Processing Magazine, Volume 37, Issue 1, 2020 
[6] Sriram, Anuroop, et al. "End-to-end variational networks for accelerated MRI reconstruction." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.
[7] Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." Proceedings of the European conference on computer vision (ECCV). 2018.
[8] Timofte, Radu, Rasmus Rothe, and Luc Van Gool. "Seven ways to improve example-based single image super resolution." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[9] Liang, Jingyun, et al. "Swinir: Image restoration using swin transformer." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
[10] Zhang, Zhengxin, Qingjie Liu, and Yunhong Wang. "Road extraction by deep residual u-net." IEEE Geoscience and Remote Sensing Letters 15.5 (2018): 749-753
[11] Zhang, Yulun, et al. "Residual dense network for image super-resolution." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
[12] Muckley, Matthew J., et al. "Results of the 2020 fastmri challenge for machine learning mr image reconstruction." IEEE transactions on medical imaging 40.9 (2021): 2306-2317.
[13] Eo, Taejoon, et al. "KIKI‐net: cross‐domain convolutional neural networks for reconstructing undersampled magnetic resonance images." Magnetic resonance in medicine 80.5 (2018): 2188-2201.
[14] Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.
```

## Useful Links
```
[L1] https://pytorch.org/docs/stable/checkpoint.html
[L2] https://pytorch.org/docs/stable/data.html#memory-pinning
[L3] https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088
[L4] https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
```


