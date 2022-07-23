# Should All Proposals be Treated Equally in Object Detection? (ECCV 2022)
A [pytorch](http://pytorch.org/) implementation of [DPP](https://arxiv.org/pdf/2207.03520.pdf).

## Requirements
- Linux or macOS with Python ≥ 3.7
- PyTorch = 1.12 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.

## Models
Model | AP | Head FLOPS |  download
--- |:---:|:---:|:---:
DPP-XL | 45.0 | 15G | [init](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_pro300_pretrained.pth) / [model](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_dpp_xl.pth)
DPP-L | 43.7 | 6.8G  |  [init](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_pro300_pretrained.pth) / [model](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_dpp_l.pth)
DPP-M | 42.2 | 3.2G  | [init](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_pro100_pretrained.pth) / [model](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_dpp_m.pth)
DPP-S | 40.4 | 2.1G  |  [init](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_pro50_pretrained.pth) / [model](http://www.svcl.ucsd.edu/projects/dpp/assets/dpp/model_final_dpp_s.pth)

## Steps
1. Install and build libs
```
git clone https://github.com/liyunsheng13/dpp.git
cd dpp
python setup.py build develop
```

2. Link coco dataset path to dpp/datasets/coco
```
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

3. Train DPP
```
python projects/DPP/train_net.py --num-gpus 8 \
    --config-file projects/DPP/configs/dpp.l.resnet.3x.yaml MODEL.WEIGHTS /path/to/init_model.pth
```

4. Evaluate DPP
```
python projects/DPP/train_net.py --num-gpus 8 \
    --config-file projects/DPP/configs/dpp.l.resnet.3x.yaml \
    --eval-only MODEL.WEIGHTS /path/to/model.pth
