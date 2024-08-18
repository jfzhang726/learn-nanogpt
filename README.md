# learn-nanogpt
Learn Andrej Karpathy's nano-gpt tutorial


## Preprocess fineweb-edu 10B sample
- Use Andrej Karpathy's fineweb.py script, with small modifications.
- Cost ~ 3 hours on Colab CPU instance <br> 
    GPU is useless for tokenization because it is basically to iterate over all sentences. CPU instance is used.
- Generated .npy files are saved to Drive folder for two reasons. Firstly, it is faster to save to/load from drive than to/from local PC hard drive. Secondly, more importantly, Colab CPU instance's 107GB hard drive is barely enough for loading and processing fineweb-edu 10B dataset but not enough space for saving tokenized files. 

## 