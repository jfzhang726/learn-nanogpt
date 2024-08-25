from dataclasses import dataclass
import inspect
import numpy as np
import torch 
import torch.nn as nn
from torch.nn import functional as F
import math

import logging
from datetime import datetime
import time
import pytz

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# Define the timezone
local_timezone = pytz.timezone('Australia/Sydney')

# Custom formatter to include time in the specified timezone and line number
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, local_timezone)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s

    def format(self, record):
        record.tzname = local_timezone.zone
        record.lineno = record.lineno
        return super().format(record)

# Define the logging format
log_format = "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s"

# Configure the logging
logging.basicConfig(level=logging.DEBUG,
                    format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S %Z',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('train_gpt2.log') 
                              ])

# Apply the custom formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(CustomFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S %Z'))

# Example usage
logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, in a batch
        # 3 is for q, k, v 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # mark the layer is on residual stream so that its weight will be initialized by (number of layers)**-0.5
        self.c_proj.IS_ON_RESIDUAL_STREAM = True
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Andrej: not really a 'bias', more of a mask, but following the OpenAI/HF naming though 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "the number of heads", hs is "the head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        
        # calculate q, k, v at the same time 
        # c_attn is Linear(n_embd, 3 * n_embd), so its weight A is [out_dim, in_dim] i.e. [3*n_embd, n_embd], 
        # and output y will be x@transpose(A).
        # Assume x has only one token, x is [1, n_embd], transpose(A) is [n_embd, 3*n_embd], 
        # so output y is [1, 3*n_embd], where each value y_i in y is calculated by x * transpose(A)[:,i].
        # Note that all of the 3*n_embd y_i are calculated independant of each other, so calculating all 3*n_embd y_i simutaneously (qkv)
        # is equivalent to calculating n_embed y_i (q,k or v) at a time for 3 times.
        # Extra dimensions in x i.e. batch size and sequence length are broadcasted to get the proper final shape, so they don't affect the calculation of token_embedding@transpose(A).    

        # The possible reason for PyTorch implements Linear that A is [out_dim, in_dim] and y=x@transpose(A), instead of A is [in_dim, out_dim] and y=x@A: 
        #    Linear transformation in math is usually defined as y = A@x where A is [out_dim, in_dim] and x is column vector, but in pytorch x is row vector, so PyTorch keeps A as the same as in math, and applies it to row vector x by y = x@transpose(A).
        qkv = self.c_attn(x) # [B, T, 3 * C], where q, k, and v vectors of the same token are in the same row

        # separate out q, k, v by cut each row into 3 vectors
        q, k, v = qkv.split(self.n_embd, dim=2)  # [B, T, C]
        # cut row (channels) into n_head heads, then make head as batch, so that the last 2 dimenstions are corresponding to a sequences of tokens. 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, nh, T, hs]
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        ##============================##
        ## Flash Attention can make these 4 lines very fast (7.6x). torch.compile() is not able to optimize them.
        ## Flash Attention uses more flops, but it is much faster by avoiding read and write the large TxT attention matrix to HBM. 
        ## 2018: <i>Online normalizer calculation for softmax </i> paper by NVIDIA
        ## 2022: <i>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</i> paper by Standford
        ## 2023: <i>FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning</i> paper by Standford 
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, nh, T, T]
        # # the name "bias" is confusing. It is not really a bias, more of a mask. 
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # [B, nh, T, T]
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # [B, nh, T, T] x [B, nh, T, hs] = [B, nh, T, hs]

        ## replace above 4 lines of code with one line below. PyTorch will invoke FlashAttention. 
        ## This change reduce time by 27%. 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        ##============================##

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, [B, T, C] [B, nh, T, hs]
        # output projection
        y = self.c_proj(y)
        return y
      




class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # two linear layers with gelu activation in between
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        """
        GELU: 
        <I>Gaussian Error Linear Units (GELUs) </I> https://arxiv.org/pdf/1606.08415


        https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

        approximate: in the early days extact GELU compupation implementated in Tensorflow was slow, 
                    and the authors developed tanh approximate to speed up the computation.
                    BERT and GPT2 used tanh approximate.
                    Approximate is not necessary now, and is a historical quirk. 


        Use GELU instead of RELU
        - Dead RELU problem: RELU tail is flat and exactly 0. Any activation falls in the region will get exact 0 gradient, and never get changed.
        - GELU: it is a smooth approximation of RELU, and it is not exactly 0 at the tail, so there is always going to be a change.


        Activations develop in more modern models, e.g LLAMA uses SwiGLU https://www.ai-contentlab.com/2023/03/swishglu-activation-function.html, 
        """
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # mark the layer is on residual stream so that its weight will be initialized by (number of layers)**-0.5
        self.c_proj.IS_ON_RESIDUAL_STREAM = True

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)  
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Original transformer paper: residual paths have layer normalization inside them.
        GPT implementation: clean residual pathways between inputs and supervision.

        Clean residual pathway is desirable. 
        Addition is the branch of flow of gradients during backpropagatioin. 
        In one granch, gradients flow straight from top (supervision) to the bottom (tokens) without changes. 
        In the other branch, gradients flow through blocks.  
        """
        
        # Transformer is the repeated applications of map reduce.
        # attention is the reduce: 1024 tokens communicate, exchange information (pooling, weighted sum, reduce).
        x = x + self.attn(self.ln_1(x))
        # mlp is the map: every single token thinks individually about the information it gathered.
        x = x + self.mlp(self.ln_2(x))

        # GPT-2 paper <i>Language Models are Unsupervised Multitask Learners</i> 2.3
        # "A modified initialization which accounts
        # for the accumulation on the residual path with model depth
        # is used. We scale the weights of residual layers at initialization 
        # by a factor of 1/√N where N is the number of residual layers"
        # Andrej explained motivation:
        #    if values of self.attn(self.ln_1(x)) or self.mlp(self.ln_2(x)) are of normal distribution
        #    with mean=0.0 and var=1.0, after adding the value to x for 100 times
        #    i.e. 
        #    x = torch.zeros(768)
        #    n = 100
        #    for i in range(n):
        #        x += torch.randn(768)
        #
        #   print(x.std())
        #   will show x.std() is around 10
        #   
        #   if change the code into:
        #    x = torch.zeros(768)
        #    n = 100
        #    for i in range(n):
        #        x += n**-0.5 * torch.randn(768)
        #
        #   print(x.std())
        #   will show x.std() is around 1
        #   
        #   The reason is that the sum of n normally distributed random variables have 
        #    mean = sum of means and var = sum of vars. In this case x will have mean=0 and var=n (std=n**0.5) after values of att/mlp are added to x for n times.
        #   If each random variable v = n**-0.5 * v, variance of each variable will be variance/n, 
        # and the sum of all variables' variances will be 1.
        # Therefore multiplying n**-0.5 to weights of attn/mlp can make x have var 1 after 
        # adding att/mlp for n times.
        # And n equals 2 * self.n_layer because each layer (block) has one attn and one mlp added to x.

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens, + 1 <|endoftext|>
    n_layer: int = 12 # number of layers 
    n_head: int = 12 # number of heads 
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing schema
        # In <i>Attention is all your need</i> 3.4:
        # "In our model, we share the same weight matrix betwwen the two embedding layers and 
        # the pre-softmax linear transformation, similar to [30]".
        # The [30] paper is <i>Using the Output Embedding to Improve Language Models</i> 
        # https://arxiv.org/abs/1608.05859
        # "We call U the input embedding, and V the
        # output embedding. In both matrices, we expect
        # rows that correspond to similar words to be similar: 
        # for the input embedding, we would like the
        # network to react similarly to synonyms, while in
        # the output embedding, we would like the scores
        # of words that are interchangeable to be similar"
        # 
        # GPT-2 model uses thise weight sharing schema.
        
        self.transformer.wte.weight = self.lm_head.weight 
        # now self.transformer.wte.weight points to self.lm_head.weight, 
        # and the original tensor of self.transformer.wte.weight is orphaned, 
        # and will be reclaimed by torch.
        # Andrej Karpathy coded in this way during tutorial. In production probably 
        # it can be coded in more decent way? 
        #
        # This weight sharing schema also saves parameters. The matrix has 768*50257=38,597,376
        # which is 39M/128M=30% of the total number of parameters in GPT-2. 

        # init all weights
        self.apply(self._init_weights)

    # Karpathy found GPT-2 initialization from code 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # GPT-2 initialize linear layer using normal distribution with std 0.02
            if hasattr(module, 'IS_ON_RESIDUAL_STREAM'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=(2*self.config.n_layer)**-0.5 * 0.02)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # GPT-2 initialize bias to 0, which is different from PyTorch's default (uniform) 
                torch.nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding):
            # GPT-2 initializes wte with std 0.02 and wpe with std 0.01. 
            # Andrdj Karpathy decided use 0.02 to both.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # GPT-2 layer norm initialization is the same as PyTorch default 
        # so no need special treatment.
        #  
        # 0.02 roughly matches what Xavier initialization  would have i.e. sqrt(1/embedding_size) 
        # as of GPT-2 embedding sizes 768, 1024, 1280, 1600: 0.036, 0.031, 0.028, 0.025.
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_


    def forward(self, idx, targets=None):
        # idx is input token ids, shape (B, T)
        # Tensor.shape and Tensor.size() return the same value. Tensor.shape was added to match numpy.
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the tokens and position embeddings
        # use gpu if device is gpu
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_embd), all rows of input share the same position embedding
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
        x = tok_emb + pos_emb # shape (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm
        x = self.transformer.ln_f(x)
        # forward the final linear layer (classifier)
        logits = self.lm_head(x) # shape (B, T, vocab_size),  
        # The logits indicat what is the token comes next in the sequence, i.e. token B, T+1
        # Take the first sequence in batch as example, logits [0, -1, :] indicates the token T+1 following the current sequence.
        # Note that one execution of forward() predicts one next token right of the input sequence.  

        loss = None 
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # batch is flattened into 2D (B*T, C)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss        

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pretrained GPT-2 model weights from huggingface
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        logger.info("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined based on model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # ignore these mask/buffer 

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are matched in both models
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.maksed_bias')] # ignore these autoregressive masks. But why they can be ignored?
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # Andrej: basically the openai checkpoints use a "Conv1d" module, but we only want to use a vanilla "Linear" module
        # this means that we have to transpose these weights when we import them 
        # Me: For example in sd_hf transformer.h.0.attn.c_attn.weight torch.Size([768, 2304]). Note that weights W in 
        # Linear is of shape (out_features, in_features), so the correct size of attn.c_attn.weight should be (3*768, 768) but not the one in sd_hf. So transpose is needed.  
        # But in pytorch weights W in Conv1d(in_channels, out_channels, kernel_size) are in [out_channels, in_channels, kernel_size], and in Linear(in_fatures, out_features) weights W are [out_features, in_features], so they are basically in the same order of dimensions. 
        # In the video, Andrej says "then then one additional kind of annoyance is that this comes from the 
        # tensorflow repo and I'm not sure how this is a little bit annoying but some of the weights are 
        # transposed from what pytorch would want and so manually I hardcoded the weights that should be 
        # transposed and then we transpose them"
        # It sounds like the transposing of weights is not caused by Conv1d v.s Linear but by some weird codes. 

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed): # any() is disjunctive operator
                # special treatment for the Conv1d weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape # reverse the shape of sd_hf[k] to check if match sd[k]
                
                # Wrapping in-place operation in torch.no_grad() ensures that the in-place operation does not 
                # interfere with the gradient computation graph.
                with torch.no_grad():
                    # Tensor.copy_(src, non_blocking=False) -> Tensor
                    # Copies the elements from src into self tensor and returns self.
                    # The shape of self should match the shape of src.


                    # Tensor.t() returns transposed tensor. Input is <= 2-D tensor, and transposes dimensions 0 and 1
                    sd[k].copy_(sd_hf[k].t())
                
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require gradients)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups, Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        # Create AdamW optimizer and use the fused version if we have CUDA
        # fused is a lot faster. It 
        # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in str(device)
        logger.info(f"use_fused={use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer



####=======================================####
# Dataset:
#  - Commoncrawl: low quality, extremely noisy
#  - red pajama dataset https://github.com/togethercomputer/RedPajama-Data
#  - slim pajama dataset: https://huggingface.co/datasets/cerebras/SlimPajama-627B cleaned and deduplicated subset of red pajama
#  - fineweb daatset: subset of common crawl, cleaned and high quality https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1 karpathy likes it, especially FineWeb-Edu 

# The 10B sample of FineWeb-Edu will be used here. Karpathy said he experimented and found 10B sample is enough 
# to make model close to GPT2.
# 


#data_root = "edu_fineweb10B"
# data_root = "/content/drive/MyDrive/Colab Notebooks/nanogpt/edu_fineweb10B/"
data_root = "edu_fineweb10B"

def load_tokens(filename):
    '''
    load a shard file which is a numpy file
    '''
    npt = np.load(filename) # np.uint16, as specified in fineweb.py
    ptt = torch.tensor(npt, dtype=torch.long) # convert to torch.long
    return ptt

import tiktoken
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # with open('data/input.txt', 'r') as f:
        #     text = f.read()
        # self.enc = tiktoken.get_encoding("gpt2")
        # tokens = self.enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # logger.info(f"Loaded {len(tokens)} tokens")
        # logger.info(f"1 epoch = {len(self.tokens) // (self.B * self.T * self.num_processes)} batches")

        # Unlike working with small file like tiny shakespear, in case of fineweb we tokenized dataset in advance and save token indices in a number of numpy files (see code edu_fineweb.py)
        
        assert split in {'train', 'val'}
        # get shard names

        shards = os.listdir(data_root)
        shards = [os.path.join(data_root, x) for x in shards if split in x]
        shards = sorted(shards)
        self.shards = shards
        assert len(self.shards) > 0, f"No {split} shards found"
        if master_process:
            logger.info(f"Found {len(self.shards)} {split} shards")
        self.reset()

    def reset(self):
        # state --- initialize at 1st shard of the split
        # self.current_position = 0 # if not ddp 
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # if ddp, initial postion as for one process


    def next_batch(self):
        B, T = self.B, self.T
        start = self.current_position
        end = start + B*T + 1
        buf = self.tokens[start: end]
        
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T*self.num_processes # advance 
        # avoid out of bound
        if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_shard += 1
            if self.current_shard >= len(self.shards): # out of bound
                self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank # reset starting position
            # now the resetting of position happens when reamining tokens are less than B*T*world_size. 
            # The batchs can be slightly different with multiple GPUs compared with one GPU, and we may observe 
            # that the losses in epoches might be slightly different from the running of one GPU.
            logger.info("Resetting data loader")
        return x, y



# check code is correct or not by loading gpt2 weights into code
if False: 

    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained('gpt2')
    logger.info("Model loaded")
    # set model to evaluation mode
    # Andrdj explanation: the code above has no dropout or batch norm things so the code should behave the same 
    # for training and inference, so he is not sure if it is needed to set to evaluation mode.
    model.eval() # set model to evaluation mode
    model.to(device)


    # prefix the input
    prefix = "Hello, I'm a language model,"
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # get indices of tokens
    # can check what tokens would be at https://tiktokenizer.vercel.app/
    # got [15496, 11, 314, 1101, 257, 3303, 2746, 11]
    tokens = enc.encode(prefix) 
    logger.info(f"Tokens: {tokens}")
    tokens = torch.tensor(tokens, dtype=torch.long) # shape (L)
    tokens = tokens.unsqueeze(0) # shape (1, L)
    tokens = tokens.repeat(num_return_sequences, 1) # shape (N, L)
    x = tokens.to(device)


    # Generate tokens following input. 
    # x is (B, T) where B=5, T=8
    # set the seed to 42
    torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad(): 
            logits = model(x) # (B, T, vocab_size)
            # take only the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs is (5, 50) and topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # (B, 50)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
            # gather the corresponding token indices
            x_next = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, x_next), dim=1)
    # print the generated tokens
    for seq in x:
        text = enc.decode(seq.tolist())
        logger.info(f"Generated text: {text}")




# check if code is correct
# 1. examine loss before training
#    cross entropy loss before training should be close to loss of random guess -math.log(1/vocab_size) 
# 2. examine loss after training
if False:
    device = "cpu"
    #get a data batch
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    with open('data/input.txt', 'r') as f:
        text = f.read()
    text = text[:1000]
    tokens = enc.encode(text)
    B, T = 4, 32 # batch size, block size
    buf = torch.tensor(tokens[:B*T+1], dtype=torch.long) 
    # need to assign to varialbe i.e. buf = buf.to（device)
    # It is different from model where model.to(device) without the need of assignment model = model.to(device)
    buf = buf.to(device) 
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)



    model = GPT(GPTConfig())
    model.to(device)

    logits, loss = model(x,y)

    print(f'logit {logits.shape}')
    print(f'loss {loss.item()}')
    # compare loss with uniform probability of all tokens
    # import math
    import math
    print(f"expected loss before training:  {-math.log(1/50257)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        optimizer.zero_grad() # zero gradients
        logits, loss = model(x, y)
        loss.backward() # accumulate gradients
        optimizer.step() # update weights
        print(f"step {i}")
        print(f"loss {loss.item()}") # tensor.item() convert tensor to scalar value and move to CPU


# =======================================
import torch
### command to run without ddp
###    python train_gpt2_ddp.py 

### ddp run with 1 node and 8 GPUs. --standalone for 1 node, --nproc_per_node for number of GPUs
###    torchrun --standalone --nproc_per_node=8 train_gpt2_ddp.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
import os

# set up DDP (distributed data parallel)
# torchrn command sets env variable RANK, LOCAL_RANK, and WORLD_SIZE
# these are concepts in parallel computing https://en.wikipedia.org/wiki/Message_Passing_Interface
# DDP terminology: https://hprc.tamu.edu/files/aces_nvidia_jupyterlab.pdf
# A process or a worker:  (interchangeable terms) refers to an instance of the Python program.
#              Each process controls a single, unique GPU. The number of available GPUs for distributed
#              learning dictates the number of processes that are launched by DDP.
# A node or a host: (another set of interchangeable terms) is a physical computer along with
#              all of its hardware components (e.g., CPUs, GPUs, RAM, etc.). Each node is assigned a
#              unique ID.
# WORLD_SIZE: total number of processes participating in training. Processes in the "world" can communicate 
#             with each other. In training, usually each GPU corresponds to one process. 
# RANK: (global rank) the unique ID given to a process, so that other processes know how to 
#             identify a partical process. The process with rank 0 is referred to as the main process. 
# LOCAL_RANK: the unique ID assigned to each process within a node.
# 
# For example, if there are two nodes having two GPUs each:
#       node:    0    |     1
#       RANK: 0,   1,    2,    3
# LOCAL_RANK: 0,   1, |  0,    1
# 
ddp = int(os.environ.get("RANK", -1)) != -1 # to check this is a ddp run 
if ddp:
    # make sure cuda is available
    assert torch.cuda.is_available(), "CUDA is not available"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"]) # GPU 0 will have RANK 0, GPU 1 will have RANK 1, etc
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # 
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}" # e.g. "cuda:0"
    torch.cuda.set_device(device)
    # if ddp_rank (global rank) == 0, then it is master process, and will do logging, checkpointing etc. 
    master_process = ddp_rank == 0 
    logger.info(f"ddp_rank={ddp_rank} ddp_local_rank={ddp_local_rank} ddp_world_size={ddp_world_size}")
else:
    # if not ddp, then it is single process 
    ddp_rank = 0   
    ddp_local_rank = 0 
    ddp_world_size = 1
    master_process = True # the only process is the main process
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # mac
        device = "mps"
    logger.info(f"Not using DDP, using device: {device}")


# all the processes (GPUs) run the same copy of code below, not aware of the existence of other copies

# set random seeds to ensure reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


total_batch_size = 524288 # 2**19, ~0.5M tokens as specified in GPT3(?) paper, and make the number "nice" (power of 2)
# on my colab instance with 1 GPU L4, set B = 16, will use 18.9/22.5 GB GPU memory, process about 35K tokens/sec,
# will use ~80 hours to finish 10B tokens;
# Andrej uses GPU A100-SXM4-80GB, with 80G memory, can set B=64, 8 GPU process about 1.5M tokens/sec, can finish # 10B tokens in 1.85 hrs
B = 16 # micro batch size (per GPU)
T = 1024 # sequence length (per GPU)
# 16 * 1024 * ddp_world_size tokens in one forward 
assert total_batch_size % (B*T*ddp_world_size) == 0, "total_batch_size should be divisible by B*T*ddp_world_size"
gradient_accumulation_steps = total_batch_size // (B*T*ddp_world_size) # how many batches to accumulate before doing backward pass. Each process now works on batch size of total_batch_size/ddp_world_size.  
if master_process:
    logger.info(f"total desired batch_size = {total_batch_size}")
    logger.info(f"calculated gradient_accumulation_steps = {gradient_accumulation_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train') 
# train_loader = DataLoaderLite(B=16, T=1024) # GPT-2 max length 1024
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# speed up training by using tf32 (Tensor Float 32)). In Andrej Karpathy's A100 GPU, it is 3x faster. 
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# override GPT-2 vocab size 50257 by 50304 because 50304 is "nicer" ---- contains powers of 2 i.e. 50304=128*393
# powers of 2 can make GPU work faster.
# The reason is that the GPU kernels chunk input and do the nice part first, then do the remaining part in the 2nd phase.
# 
# this change speeds up by roughly 4%. 
# If the change was applied to earlier versions of PyTorch (2.3.1 and earlier), it might get 30% improvement.
model = GPT(GPTConfig(vocab_size=50304)) #model = GPT(GPTConfig())

model.to(device)
#  Karpathy showed that the single line of torch.compile achieves 2.3x speedup.
#  He recommended to use torch.compile by default.  
use_compile = False # torch.compile breaks evaluation code
if use_compile:
    model = torch.compile(model) # it is not working in my vs code. 
if ddp:
    # wrap the model in the DistributedDataParallel container
    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    # Karpathy thinks device_id must be local rank although documentation is not clear
    # DDP container does:
    # - in forward pass, it does nothing 
    # - in backward pass, DDP container does an average across all the ranks of their gradients, and then 
    #    deposite the average on every single rank
    model = DDP(model, device_ids=[ddp_local_rank]) 
# store raw model 
raw_model = model.module if ddp else model
# learning rate scheduler
# GPT-3 paper <i>Language Models are Few-Shot Learners</i> Appendix B Details of Model Training
# " we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260
# billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup 
# over the first 375 million tokens. "

# max_lr 6e-4 as per GPT-3 paper <i>Language Models are Few-Shot Learners</i> Table 2.1.
# However, some people experimented 3 times larger max_lr to make it learn faster. Could try it.
max_lr = 6e-4  

min_lr = max_lr / 10.0

# warmup_steps = 10
# max_steps = 50
# 
# set warmup_steps and max_steps for edu_fineweb10B data.
# Model will be trained 1 epoch. 
# edu_fineweb10B has about 10e9 tokens, and total_batch_size is 2**19 ~0.5M
# max_steps = 10e9 / 2**19 = 19073.486328125 ~= 19073
# GPT3 paper said LR warmup over the first 375 million tokens, 
# so warmup_steps = 375e6/ 2**19 = 715.2557373046875 ~= 715
# Karpathy said warm up 715 step is mild and it could be more aggressive 
# for example warmup for 100 steps. 

warmup_steps = 715
max_steps = 19073
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use consine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0) # coeff starts at 1 and ends at 0.5
    return min_lr + (max_lr - min_lr) * coeff
    



# hyperparameters from GPT-3 paper <i>Language Models are Few-Shot Learners</i> Appendix B Details of Model Training
# beta_1 = 0.9
# beta_2 = 0.95
# eps = 1e-8
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
# config optimizer on raw_model instead of ddp model
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# create log dir to store checkpoints and logs

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # start with empty file
    pass

import tiktoken
enc = tiktoken.get_encoding("gpt2")
from hellaswag import iterate_examples, render_example 

# helper function for HellaSwag eval 
def get_most_likely_row(tokens, mask, logits):
    # evaluate autoregressive loss at all positions
    # ... in numpy indicates "as many as needed dimensions", so [..., :-1, :] is equivalent to [:,:,..., :-1, :]
    # 
    # torch.Tensor.contiguous(): some tensor operations including Tensor.transpose(), 
    # Tensor.expand(), Tensor.narrow() and Tensor.view() don't change the content of the tensor, but change the meta data to specify new layout. (On the contrary, operations like Tensor.reshape() combines view() and contiguous() operations and create a new tensor. )
    # 
    # Tensor.contiguous() is useful because some torch operations exepect a contiguous tensor, otherwise the indexing might be wrong. PyTorch will either make the memory contiguous or raise an error if the input tensor is not contiguous. When there is an error saying cotiguous tensor is needed, Tensor.contiguous() 
    # needs to be called explicitly. 
    # On the other hand, many operations don't need congituous tensor so it is not good to call contiguous() all the time. 
    # https://discuss.pytorch.org/t/does-contiguous-tensor-affect-training-result/143921
    # https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
    shift_logits = logits[..., :-1, :].contiguous() # remove last token
    shift_tokens = tokens[..., 1:].contiguous() # remove first token
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # flatten logits, 2D
    flat_shift_tokens = shift_tokens.view(-1) # flatten tokens, 1D
    # reduction='none' so returns loss of each token instead of a summarized value.   
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # 1D. 
    shift_losses = shift_losses.view(tokens.size(0), -1) # restore 2D
    # get the average loss just for the completion region (where mask == 1) in each row
    shift_mask = mask[..., 1:].contiguous() # must shift mask, so start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask # 0 for tokens not in completion region
    total_per_row = masked_shift_losses.sum(dim=1) # sum losses for each row
    num_tokens_per_row = shift_mask.sum(dim=1) # count number of tokens in each row
    avg_loss_per_row = total_per_row / num_tokens_per_row # average loss for each row
    # now we have the loss for each of the 4 completions
    # the one with the lowest loss is the most likely
    pred_norm = avg_loss_per_row.argmin().item() # get the index of the lowest loss
    return pred_norm


def train():
    for step in range(max_steps):
        t0 = time.time() # start time
        last_step = (step == max_steps - 1)
        # evaluate on validation set every 100 steps
        # Training data is roughly infinite so training loss and val loss should be about the same.
        if step % 100 == 0 and master_process:
            model.eval()
            val_loader.reset() # reset to start of data
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    # device_type must be "cuda" instead of "cuda:0", "cuda:1" ...
                    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach() 
            if ddp:
                torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG)
            if master_process:
                logger.info(f"step {step} val loss {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # save checkpoint
                    check_point_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    check_point = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'optimizer': optimizer.state_dict(),
                        # random seed
                    }
                    torch.save(check_point, check_point_path)

        # evaluate hellaswag
        if (step % 250 == 0 or last_step) and not use_compile:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples('val')):
                # each process only do a part of examples
                if i % ddp_world_size != ddp_rank:
                    continue
                # render exmaple into tokens and lable
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get logits 
                with torch.no_grad():
                    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += (pred_norm == label)
                # reduce the states across all processes
            if ddp:
                num_total = torch.tensor(num_total).to(device=device, dtype=torch.long)
                num_correct_norm = torch.tensor(num_correct_norm).to(device=device, dtype=torch.long)
                torch.distributed.all_reduce(num_total, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_correct_norm, op=torch.distributed.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                logger.info(f"step {step} hellaswag acc: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")
        # generate sample every 100 steps.
        # Karpathy says torch.compile throws error in code below. 
        # Either disable torch.compiler to generate samples which makes training a bit slower,
        # or disable sample generation to use torch.compiler.
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile) :
            num_return_sequences = 4
            max_length = 32

            model.eval() # set model to evaluation mode

            # prefix the input
            prefix = "Hello, I'm a language model,"
            # get indices of tokens
            # can check what tokens would be at https://tiktokenizer.vercel.app/
            # got [15496, 11, 314, 1101, 257, 3303, 2746, 11]
            tokens = enc.encode(prefix) 
            logger.info(f"Tokens: {tokens}")
            tokens = torch.tensor(tokens, dtype=torch.long) # shape (L)
            tokens = tokens.unsqueeze(0) # shape (1, L)
            tokens = tokens.repeat(num_return_sequences, 1) # shape (N, L)
            xgen = tokens.to(device)

            # Generate tokens following input. 
            # set different random state to different processes
            sample_random_generator = torch.Generator(device=device)
            sample_random_generator.manual_seed(42 + ddp_rank)
            
            while xgen.size(1) < max_length:
                # forward pass to get logits
                with torch.no_grad(): 
                    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take only the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # apply softmax to get probabilities
                    probs = F.softmax(logits, dim=-1) # (B, vocab_size)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs is (5, 50) and topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # (B, 50)
                    # select a token from the top-k probabilities
                    ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_random_generator) # (B, 1)
                    # gather the corresponding token indices
                    x_next = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, x_next), dim=1)
            # print the generated tokens
            # for seq in xgen:
            #     text = enc.decode(seq.tolist())
            #     logger.info(f"Generated text: {text}")        
            for i in range(num_return_sequences):
                seq = xgen[i, :max_length]
                text = enc.decode(seq.tolist())
                logger.info(f"rank {ddp_rank} Generated sample {i}: {text}")


        # training
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(gradient_accumulation_steps):
            # get a data batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # device_type must be "cuda" instead of "cuda:0", "cuda:1" ...
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # turn script execution into interactive mode
            #import code; code.interact(local=locals())


            
            # scale the loss to account for gradient accumulation.
            # because gradients add on each successive backward().
            # addition of gradients corresponds to a SUM in the loss function (cross_entropy here), but 
            # we want MEAN in the loss function (cross_entropy). So we divide by gradient_accumulation_steps.
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach() # detach the tensor from computation graph. record accumulated loss for logging
            
            # loss.backward() does x.grad += dloss/dx i.e. accumulate gradients. 
            # If not for gradient accumulation, we would zero gradient before calling loss.backward().
            # https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward
            # Here we want to accumulate gradients.
            # Now we have DDP. we don't want ddp to do communication (average) in every step except after the 
            # last of accumulation steps. 
            if ddp:
                model.require_backward_grad_sync = (micro_step + 1 == gradient_accumulation_steps)
            loss.backward()
        # Below is for printing loss_accum only. don't confuse it with calculating average of gradients. 
        # Remember we want to print in main process only, but main process has its own loss_accum. The line below 
        # calculate average of loss_accum across all ranks, and deposite it on every rank, so main process is able to
        # print the average of loss_accum of all ranks. 
        if ddp:
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
        # GPT-3 paper: clip global norm of gradients at 1.0
        # the norm is sqrt(g1**2 + g2**2 +...) i.e. the length of the vector.
        # the line below cap norm no more than 1.0.
        # Sometimes e.g. in case of bad batch, the loss is high then gradient is high, and model is shocked.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step) # learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize() # wait for all kernels to finish
        t1 = time.time() # end time
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * gradient_accumulation_steps * ddp_world_size # all ranks
        tokens_per_sec = tokens_processed / dt     

        if master_process:
            logger.info(f"step {step:4d} | lr: {lr:.4e}| loss {loss.item():6f} | accumulated loss: {loss_accum.item():6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    # otherwise there are warning messages
    if ddp:
        destroy_process_group()
    # import sys; sys.exit(0)






