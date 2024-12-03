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

   Detailed Caption from [PixelProse](https://huggingface.co/datasets/tomg-group-umd/pixelprose) and [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V).

2. Instruction Data:

   TODO.

## Training Script
### Pretraining with llama 3.1-8B

Set up your basic slurm information in the  ```scripts/florence-vl/llama/llama3.sh```
Then you can run pretrain and finetune job:


In ```scripts/florence-vl/llama/pretrain_llama.sh```, you need to manully export the following variable:

```shell
export NNODES=number of nodes
export DATA_PATH=/your/path/for/pretrain/data/json/file
export IMG=/your/image/folder
export OUTPUT=/checkpoint/save/path
```




In ```scripts/florence-vl/llama/finetune_llama.sh```, you need to manully export the following variable:


```shell
export NNODES=number of nodes
export DATA_PATH=/your/path/for/instuction/data/json/file
export IMG=/your/image/folder


export CKPT_PATH=/pretrain/checkpoint
export VIT_PATH=/pretrain/vision/tower (usually included in the pretrain checkpoint)
export OUTPUT=/checkpoint/save/path
```



## Checkpoint 

1. Florence-VL 8B: [Pretrained Checkpoint](https://huggingface.co/jiuhai/florence-llama-pretrain) and [Instructed Checkpoint](https://huggingface.co/jiuhai/florence-llama-llava-sft).
2. Florence-VL 3B: [Pretrained Checkpoint](https://huggingface.co/jiuhai/florence-phi-pretrain) and [Instructed Checkpoint](https://huggingface.co/jiuhai/florence-llama-ms).






