# Transformer-based Image Compression
Pytorch Implementation of "Transformer-based Image Compression"[[arXiv]](https://arxiv.org/abs/2111.06707), DCC 2022.

Our newly released work **"TinyLIC"** with more efficient performance can be found at the [homepage](https://njuvision.github.io/TinyLIC/). 

## Acknowledgement
The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/), we add our networks in compressai.models.tic and compressai.layers for usage.

## Installation
To get started locally and install the development version of our work, run the following commands (The [docker environment](https://registry.hub.docker.com/layers/pytorch/pytorch/1.8.1-cuda11.1-cudnn8-devel/images/sha256-024af183411f136373a83f9a0e5d1a02fb11acb1b52fdcf4d73601912d0f09b1?context=explore) is recommended):
```bash
git clone https://github.com/lumingzzz/TIC.git
cd TIC
pip install -U pip && pip install -e .
pip install timm
```

## Usage

### Train
We use the [Flicker2W](https://github.com/liujiaheng/CompressionData) dataset for training, and the [script](https://github.com/xyq7/InvCompress/tree/main/codes/scripts) for preprocessing.

Run the script for a simple training pipeline:
```bash
python examples/train.py -m tic -d /path/to/my/image/dataset/ --epochs 300 -lr 1e-4 --batch-size 8 --cuda --save
```

### Evaluation

An example to evaluate model:
```bash
python -m compressai.utils.eval_model checkpoint path/to/eval/data/ -a tic -p path/to/pretrained/model --cuda
```

## Notes
Some implementations are slightly different from the paper:
1. We remove the activation functions after the convolutions (e.g. the GDN and LReLU), which have no influence to the performance. 
2. The implementation of the Causal Attention Module (CAM) is slightly different from the paper by directly masking the
input of context model, it shows more feasible than the original one. 

## Citation
If you find this work useful for your research, please cite:

```
@INPROCEEDINGS{9810760,
               author={Lu, Ming and Guo, Peiyao and Shi, Huiqing and Cao, Chuntong and Ma, Zhan},
               booktitle={2022 Data Compression Conference (DCC)}, 
               title={Transformer-based Image Compression}, 
               year={2022},
               volume={},
               number={},
               pages={469-469},
               doi={10.1109/DCC52660.2022.00080}}
              }
```
## Contact
If you have any question, please contact me via luming@smail.nju.edu.cn.
