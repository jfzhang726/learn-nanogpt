# learn-nanogpt
Learn Andrej Karpathy's nano-gpt tutorial

Andrej Karpathy published a 4 hour long video in July 2024, explaning reproducing a GPT model. It took me quite a while to finish the tutorial video. Reason one is that I literally typed the code line by line, i.e. pause the video every few minutes to type the code, so the progress is slow. Reason two, more importantly, is that there are too many details to pay attention to, and many additional documents are read to understand the details. 

I finally had a 125M GPT model trained in my hand. I want to record my experience of learning, coding, and training.  

This document:
- IS the note of my experience of learning, recording things that took me some time to figure out during learning and coding, but not elaborated or covered in the tutorial. 
- IS NOT a note or transacript or review of the nano-gpt tutorial. Andrej Karpathy is an amazing teacher, his tutorial is clear and instructive. There is no need for additional review.

This document is written after I finished the tutorial, and the order of the sections are rearranged to be more logical.

I will start with the workflow of using Lmabdalab GPU instance, and then a number of highlights of what I learned from the tutorial and the code. 

## 1. Workflow of using Lmabdalab GPU instance 


I care about money and want to keep the GPU instance running as short as possible. The workflow achieves it by:
- preprocessing training data on Google Colab CPU instance
- using Huggingface-Hub as storage of training data and model, to transfer data to/from GPU instance very fast, otherwise uploading/downloading data from my local PC would take hours of GPU instance time.
- using GitHub as storage of code, to transfer code to/from GPU instance conveniently.

Workflow: 
- Preprocess training data on Google Colab CPU instance, upload to Huggingface-Hub Dataset Repo
- develop training code on Cursor IDE on my local PC, push to GitHub
- train model on Lambdalabs GPU instance
    - download training code from GitHub to Lambdalabs GPU instance
    - start training
    - upload model (best checkpoint) to Huggingface-Hub Model Repo
    - optionally, download other checkpoints to Google Drive from Huggingface-Hub Model Repo

Prerequisites:
- Google Colab account: for tokenization of edu_fineweb10B-sample
- Google Drive account: for saving tokenized edu_fineweb10B-sample
- Huggingface account: for creating Huggingface-Hub dataset and model repos
- Huggingface-Hub access token: for uploading training data and models/checkpoints to Huggingface-Hub repos
- GitHub account: for creating code repo on GitHub
- GitHub access token: for push code to GitHub repos
- Lambdalabs account: for creating and using Lambdalabs GPU instance
- Lambdalabs ssh key: for accessing Lambdalabs GPU instance from my local PC

Note:
- Ssh key is used for Lambdalabs instance because it is mandatory to use lambdalabs GPU instance. Need to config ssh key on local PC and GPU instance before lambda labs instance can be used. 
- Access tokens instead of ssh key are used for accessing Huggingface-Hub and GitHub because I found it is easier to login to Huggingface-Hub and GitHub from GPU instance using access tokens than to config ssh on GPU instance and login. 



## 1.1 Preprocess training data on Google Colab CPU instance, upload to Huggingface-Hub Dataset Repo

Andrej Karpathy downloaded and tokenized fineweb-edu-10B-sample dataset on the fly during training on his 8x A100 GPU instance. The processing time is about 30 minutes. I want to save money, so I run the tokenization on my Colab CPU instance. It took ~3 hours to tokenize and save the 100 .npy files to my Google Drive. 
Extra work is done to figure out the (hopefully) the easiest way to make the files available to Lambdalabs GPU instance. After exeprimented Google Drive and Huggingface-Hub, I found Huggingface-Hub is the best choice. It is easy to download the files from Huggingface-Hub to Lambdalabs instance. 

Notebook xxxx explains the approaches that I tried, and why I chose the current approach. 

Here describes the current approach. 
### 1.1 Prepare and upload   
<B> Step 1:</B> Tokenize fineweb-edu-10B-sample on Google Colab CPU instance, and save to Google Drive
- Use Andrej Karpathy's fineweb.py script, with small modifications.
- fineweb-edu/sample/10B is a subset of fineweb, with 10B tokens
- Ran ~ 3 hours on Colab CPU instance to tokenize <br> 
    GPU is useless for tokenization because it is basically to iterate over all sentences. CPU instance is used.
- Generated .npy files are saved to Drive folder for two reasons. It is faster to save to/load from drive than to/from local PC hard drive. More importantly, Colab CPU instance's 107GB hard drive is barely enough for loading and processing fineweb-edu 10B dataset but not enough space for saving tokenized files. 
- 100 .npy files are generated, each file has 100M tokens, using uint16, about 170M on hard drive

<B> Step 2:</B> Upload 100 tokenized .npy files to Huggingface-Hub </B>
The easiest way to upload .npy files to Huggingface-Hub is to use huggingface_hub.HfApi. 
- Use HfApi to create a dataset repo. Remeber to pass argument repo_type="dataset" otherwise error may happen.
- HfApi.upload_file() is used to upload files. 

Code exmaple:
```python
import os
from huggingface_hub import HfApi, login

api = HfApi()
login(token="") # fill in token from Huggingface

local_dir = "/content/drive/MyDrive/Colab Notebooks/nanogpt/edu_fineweb10B/"
repo_id = "jfzhang/edu_fineweb10B_tokens_npy_files"

api.create_repo(repo_id=repo_id, repo_type="dataset")

fn_list = os.listdir(local_dir)
for filename in fn_list:
    if filename.endswith(".npy"):
        local_path = os.path.join(local_dir, filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset" # it is important to pass repo_type="dataset", otherwise error will happen
        )
``` 

<B> Step 3:</B> Remember to add the code of downloading the dataset in the training script

During training, the tokenized files will be downloaded from Huggingface-Hub Dataset repo to the Lambdalabs instance. 
The easiest way is to use snapshot_download() to download the files in training script.

Code example:
```python
from huggingface_hub import snapshot_download

repo_id = "jfzhang/edu_fineweb10B_tokens_npy_files"
local_dir = "./edu_fineweb10B/"
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
``` 


This approach has the following advantages:
- can use .npy files directly, no need to convert to other formats (datasets.Dataset is annoying in this case)
- is very fast to upload/download files (datasets.Dataset is too slow)
- is convenient and reliable to authenticate  (git is annoying in this case)




### 1.2 develop training code on Cursor IDE on my local PC, push to GitHub 
Cursor IDE is a very smart AI assistant. It can write code, explain code, and debug code. It is amazing. 
Configging GitHub on Cursor IDE is the same as in VSCode. 


### 1.3 train model on Lambdalabs GPU instance, and upload to Huggingface-Hub model repo
<B> Step 1:</B> Create a Huggingface-Hub model repo

I found the easiest way of using Huggingface-Hub model repo is to treat it as a git repo. 
I created the repo through Huggingface-Hub Web UI. 
I will use use git commands in terminal (which is ssh to Lambdalabs instance) to push/pull/clone the repo as a normal git repo. 

This approach has the following advantages:
- can create checkpoints in any format ( because no need to use model.save_pretrained() and model.push_to_hub())

<B> Step 2:</B> start a Lambdalabs GPU instance
Do it from Lambdalabs Web UI. 
It will take minutes to start a GPU instance. A public IP address of the instance will be shown in the web UI soon after you click the start button. But it takes a while for the instance to be ready for ssh. 

<B> Step 3:</B> SSH to GPU instance
Launch terminals from local PC. I use Windows Power Shell. 
The command to ssh to GPU instance is:
```bash
ssh -i <path-to-private-key> ubuntu@<public-ip-address>
```
path-to-private-key is the path to the private key file on local PC. 
public-ip-address is the public IP address of the GPU instance. 

After the ssh is successful, you will be in the home directory of the GPU instance. 


<B> Step 4:</B> clone the GitHub repo of training code to GPU instance

If the repo is public and you won't save files modified/generated in training to GitHub, you can clone it directly:
```bash
git clone https://github.com/<username>/<repo-name>.git
```
If the repo is private or you want to save changes to the repo, you need to use the access token:
```bash
git clone https://<username>:<access-token>@github.com/username/repo-name.git
```

<B> Step 5:</B> install dependencies
```bash
pip install -r requirements.txt
```

<B> Step 6:</B> start training
The training was on one node, 8 GPUs, and the command was:
```bash
torchrun --standalone --nproc_per_node=8 <training-script>
```
This training did not take long (less than 3 hours). It was fun for me to watch the training log line by line real-time. (Note that it was the first time I use DDP and the first time I train a model with 8 GPUs.) 


<B> Step 7:</B> when the training was running, I cloned the Huggingface-Hub model repo to GPU instance
CPU was not busy during train, so I launched another terminal to clone the Huggingface-Hub model repo to the GPU instance. The repo was empty at the time. It was going to be used after training finished ---- I was going to use git commands to upload the best/last checkpoint from GPU instance to Huggingface-Hub model repo. 
```bash
git clone https://<huggingface_username>:<huggingface_access_token>@huggingface.co/<huggingface_username>/<model-repo-name>

sudo apt-get install git-lfs

git lfs install

git config --global user.email "<your-email>"
git config --global user.name "<your-username>"

```
I used .pt as the file extension for the checkpoints, which is included in the .gitattributes file by default, so no need to explicitly add it.
If the file extension used for checkpoint is not included in the .gitattributes file, need to add it to the .gitattributes file.



<B> Step 8:</B> after training finished, I uploaded the last (and best) checkpoint to Huggingface-Hub model repo
Copy the selected checkpoints to the local folder of model repo.
```bash
cp <checkpoint-path> <model-repo-path>/<checkpoint-name>
```

Upload the checkpoints to the model repo.
```bash
cd <model-repo-path>
git add .
git commit -m "add checkpoint"
git push
```

<B> Step 9:</B> (optional) I downloaded other checkpoints from GPU instance to Google Drive
All the data on Lambdalabs instance will be deleted after the instance is terminated. If you want to keep the checkpoints other than the ones uploaded to Huggingface-Hub, you can download them to Google Drive like me.
1. Start a Google Colab CPU notebook. 
2. mount Google Drive
3. Copy the ssh private key file used by Lambdalabs instance to the Colab instance at a path like ~/.ssh/<private-key-filename> 
4. scp to download the checkpoints from GPU instance to Google Drive
```bash
!chmod 400 ~/.ssh/<private-key-filename>
!scp -o StrictHostKeyChecking=no -i ~/.ssh/<private-key-filename> ubuntu@<gpu-instance-public-ip-address>:~/<checkpoint-path-on-gpu-instance> /content/drive/MyDrive/<path-on-google-drive>
```

## 2. Notes on the code






# Gradient Accumulation vs DP vs DDP
- Gradient Accumulation: single process with single GPU, accumulate gradients in CPU memory then update parameters. 
- DP: DataParallel, single process with multiple GPUs, each GPU is a process
- DDP: DistributedDataParallel, multiple processes with multiple GPUs, each GPU is a process
https://blog.csdn.net/deephub/article/details/111715288



# model.require_backward_grad_sync
Looks like the operation is removed in karpathy's code. Will look into it later. 
```python
            if ddp: # karpathy removed this operation
                 model.require_backward_grad_sync = (micro_step + 1 == gradient_accumulation_steps)
```


# One bug caused ddp hang
The conidtion master_process stops other processes from exeucting code, however torch.distributed.all_reduce() will hang if some processes are not done with their job. 
```python
for step in range(max_steps):
    if step % 250 == 0 and master_process:  # master_process condition caused ddp hang
        model.eval()
        ...
        ...
        if ddp:
            torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG)
        ...
        ...
```



# misc
- V100 does not support bfloat16
- T4, L4, A10, A100 (40G) supports bfloat16
