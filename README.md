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