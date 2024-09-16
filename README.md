# Florence-2-vlm

## Install

1. Install Package
```Shell
conda create -n florence python=3.10 -y
conda activate florence
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```




- Cambrian-7M: (https://huggingface.co/datasets/nyu-visionx/Cambrian-10M)
```Shell
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nyu-visionx/Cambrian-10M', repo_type='dataset')
```
cd {HF_HOME}/hub/datasets--nyu-visionx--Cambrian-10M/snapshots/a087b9234c59bc6c64e7e4a091a6a618cb887132
python extract.py
mv {HF_HOME}/hub/datasets--nyu-visionx--Cambrian-10M/snapshots/a087b9234c59bc6c64e7e4a091a6a618cb887132  {IMG}/Cambrian-7M




- Vision_Flan: (https://huggingface.co/datasets/Vision-Flan/vision-flan)
```Shell
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Vision-Flan/vision-flan', repo_type='dataset')
```
cd {HF_HOME}/hub/datasets--Vision-Flan--vision-flan/snapshots/e8c6f09736277ef63b33dea5e9bbe94392dba76c
bash run.sh
mv {HF_HOME}/hub/datasets--Vision-Flan--vision-flan/snapshots/e8c6f09736277ef63b33dea5e9bbe94392dba76c   {IMG}/vision_flan




- Docmatix: (https://huggingface.co/datasets/jiuhai/docmatix)
```Shell
from huggingface_hub import snapshot_download
snapshot_download(repo_id='jiuhai/docmatix', repo_type='dataset')
```
unzip the files




Download the json file including the prompt and response: 

```Shell
from huggingface_hub import snapshot_download
snapshot_download(repo_id='jiuhai/florence-data-sft', repo_type='dataset')
```


TODO: ShareGPT4V



For llama3-8b
```Shell
source /fsx_0/user/jiuhai/florence/bin/activate
export HF_HOME=/fsx_0/user/jiuhai
export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'

bash finetune_llama3.sh

```


For phi3.5
```Shell
source /fsx_0/user/jiuhai/florence/bin/activate
export HF_HOME=/fsx_0/user/jiuhai
export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'

bash finetune_phi3.sh

```


