# learn-nanogpt
Learn Andrej Karpathy's nano-gpt tutorial


## Preprocess fineweb-edu-10B-sample
- Use Andrej Karpathy's fineweb.py script, with small modifications.
- fineweb-edu/sample/10B is a subset of fineweb, with 10B tokens
- Ran ~ 3 hours on Colab CPU instance to tokenize <br> 
    GPU is useless for tokenization because it is basically to iterate over all sentences. CPU instance is used.
- Generated .npy files are saved to Drive folder for two reasons. It is faster to save to/load from drive than to/from local PC hard drive. More importantly, Colab CPU instance's 107GB hard drive is barely enough for loading and processing fineweb-edu 10B dataset but not enough space for saving tokenized files. 
- 100 .npy files are generated, each file has 100M tokens, using uint16, about 170M on hard drive
## Upload 100 tokenized .npy files to Huggingface-Hub 
- Create dataset repository in Huggingface-Hub web ui
- Use Colab instance to create datasets.DatasetDict, where each .npy file is a datasets.Dataset
- Use datasets.DatasetDict.push_to_hub() to push dataset dicts to dataset repository
- Colab CPU instance RAM is too small to load all 100 files, so do push for mulitple times. 
    - Evch push will overwrite the list of files in README.md with the file names in current push, which will make README.md meta data section contains only the files pushed in the last time. When dataset.load_dataset(), only the files in the last push are retrieved. 
    - Solution is simply to delete the file names in README.md. 

## Download tokenized files from Huggingface-Hub
Option 1: git clone dataset on Huggingface-hub

Huggingface-hub datasets are git repositories. Simple command git clone 
https://huggingface.co/datasets/jfzhang/edu_fineweb10B_tokens 
downloads 100 files. The files are parquet instead .npy because datasets library convert .npy files to parquet. 

After data set is cloned, simple code to convert .parquet files back to .npy files.
```python
        import pandas as pd
        import numpy as np
        import os
        for fn in os.listdir("/content/edu_fineweb10B_tokens/data/"):
        print(fn)
        df_shard = pd.read_parquet(os.path.join("/content/edu_fineweb10B_tokens/data/", fn))
        shard = np.array(df_shard["tokens"][0], dtype=np.uint16)
        np.save(os.path.join("./data/", fn), shard)
```

Option 2: use datasets.load_dataset()

    Use




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
