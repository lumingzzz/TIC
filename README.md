# Transformer-based Image Compression
Pytorch Implementation for "Transformer-based Image Compression"[[arXiv]](https://arxiv.org/abs/2111.06707), DCC 2022

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

### Evaluation
The pre-trained model will be released soon.

### Train
We use the [Flicker2W](https://github.com/liujiaheng/CompressionData) dataset for training, the preprocess script is in [link](https://github.com/xyq7/InvCompress/tree/main/codes/scripts).

run the script for a simple training pipeline:
```bash
python3 examples/train.py -m tic -d /path/to/my/image/dataset/ --epochs 300 -lr 1e-4 --batch-size 8 --cuda --save
```

## Citation
If you find this work useful for your research, please cite:

```
@article{lu2021transformer,
	title={Transformer-based Image Compression},
	author={Lu, Ming and Guo, Peiyao and Shi, Huiqing and Cao, Chuntong and Ma, Zhan},
	journal={arXiv preprint arXiv:2111.06707},
	year={2021}
}
```
## Contact
If you have any question, please contact me via luming@smail.nju.edu.cn.

## Related links
 * Kodak image dataset: http://r0k.us/graphics/kodak/
 * BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
 * HEVC HM reference software: https://hevc.hhi.fraunhofer.de
 * VVC VTM reference software: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
