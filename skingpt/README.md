# [Reproduce] SkinGPT-4

from root project directory
```
cd ./model_skingpt4
```

## Conda Environment

### (Option 1) Installation from our yml

```
module load cuda/11.8.0
conda env create -f skingpt4_env.yml (-n new_name_optional)
conda activate skingpt4 (or new_name_optional)
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec remove skingpt4(or new_name_optional)
python -m ipykernel install --user --name skingpt4(or new_name_optional) --display-name "skingpt4(or new_name_optional)"
```
in jupyter notebook, select kernal skingpt4(or new_name_optional).


### (Option 2) Installation from scratch based on original repo
- original repo: https://github.com/JoshuaChou2018/SkinGPT-4
- **weird steps** cannot be skipped for rivanna hpc. the original installation has many cuda problems (and dependence conflicts) on our gpu.

```
#source ~/miniconda3/etc/profile.d/conda.sh
#module avail
module load cuda/11.8.0
#which nvcc 

conda env create -f environment.yml -n skingpt4
conda activate skingpt4
conda install -c conda-forge mamba=1.4.7
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda uninstall pytorch torchvision torchaudio pytorch-cuda
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
(optional outside session)
```
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec list
jupyter kernelspec remove skingpt4
python -m ipykernel install --user --name skingpt4 --display-name "skingpt4"
```



## Download LLMs and pull their weights

### step 1: pretrained weights are under /weights folder
- modify line 10 at SkinGPT-4-llama2/eval_configs/skingpt4_eval_llama2_13bchat.yaml to be the path of current repo SkinGPT-4 weight.
- modify line 11 at SkinGPT-4-llama2/eval_configs/skingpt4_eval_vicuna.yaml to be the path of current repo SkinGPT-4 weight.

### step 2: download LLMs and pull their weights
create a hf_models folder under model_skingpt4
(run once) remember add model folders to .gitignore
```
echo "hf_models/" >> .gitignore
echo "Llama-2-13b-chat-hf/" >> .gitignore
echo "llama-13b/" >> .gitignore
echo "vicuna-13b-delta-v0/" >> .gitignore
```

```
conda activate skingpt4
cd ./hf_models
```

#### step 2.1: Llama2 Version

```shell
conda install -c conda-forge git-lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
cd Llama-2-13b-chat-hf
git lfs install
# pull the real files
git lfs pull
cd ..
```
Then modify line 16 at /skingpt4/configs/models/skingpt4_llama2_13bchat.yaml to be the path of Llama-2-13b-chat-hf.

#### step 2.2: Vicuna Version

```shell
# download Vicuna’s **delta** weight
conda activate skingpt4
conda install -c conda-forge git-lfs
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0
cd vicuna-13b-delta-v0
git lfs install
# pull the real files
git lfs pull
cd ..

# get llama-13b model
git clone https://huggingface.co/huggyllama/llama-13b
cd llama-13b
git lfs install
# pull the real files
git lfs pull
cd ..

pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
pip install transformers==4.28.0
python -m fastchat.model.apply_delta --base ./llama-13b --target ./vicuna --delta ./vicuna-13b-delta-v0
cd ..
```
Then modify line 16 at /skingpt4/configs/models/skingpt4_vicuna.yaml to be the path of vicuna.



## Launching Demo Locally

from root project dir 
```
cd ./model_skingpt4
```

##### Llama2 Version
```
conda activate skingpt4
pip uninstall -y numpy
pip install numpy==1.26.4
python demo.py --cfg-path eval_configs/skingpt4_eval_llama2_13bchat.yaml  --gpu-id 0
```

##### Vicuna Version
```
conda activate skingpt4
cd ./model_skingpt4
python demo.py --cfg-path eval_configs/skingpt4_eval_vicuna.yaml  --gpu-id 0
```

## (Final export to env.yml)
```
conda env export --name skingpt4 --no-builds | grep -v "prefix:" > skingpt4_env.yml
```
