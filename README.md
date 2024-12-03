## Install Environment

1. Install package for tranining
```Shell
conda create -n florence-vl python=3.11 -y
conda activate florence-vl
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

2. Install package for evaluation (We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluation.)
```
cd lmms-eval
pip install -e .
```


## Dataset Download

1. Pretrain Data:

   Detailed Caption from [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose).






