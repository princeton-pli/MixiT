#!/usr/bin/env python
# coding: utf-8

# %%
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset, Dataset
import logging
import random
import numpy as np
import pickle 
import os
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
from tqdm.auto import tqdm
from transformers.trainer_callback import TrainerCallback
import modeling_gpt2
import modeling_llama
import utils
from modeling_gpt2 import GPT2LMHeadModel, MixingGPTConfig
from attention_maps_callback import AttentionPlotCallback



logger = logging.getLogger(__name__)
CustomTrainer = Trainer # should be overwritten by task file

# __import__('builtins').breakpoint = lambda:None

arch = 'transformer'
run_name = ""
max_steps = 250
seed = 43
n_embd = 16
n_layer = 2
n_head = 4
n_positions = 40
device = 'cuda'
weight_decay = 0.001
learning_rate = 1e-2 if arch=='transformer' else 5e-3
warmup_steps = 300 # 500
save_steps = 600
eval_steps = 50
logging_steps = 600
save_total_limit = 3
evaluation_strategy = "steps"
lr_scheduler_type = "cosine"
random_str = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8)])
task = 'inductionheadstream_30'
calc_pca = False #True #False
tie_emb = False
weight_frozen = 1
vocab_size = 128
train_size = 40000
eval_size = 4000
dropout = 0.
eval_accumulation_steps = 1
shaped_attention = 'shaped' # vanilla | shaped | mixing
per_device_train_batch_size=64
per_device_eval_batch_size=64
depth_alpha = 0.
base_n_layer = 2.
freeze_attention = False
freeze_mlp = False
tau0 = 1.
plot_attention_maps = False

# # %% config from command line
exec(open('configurator.py').read()) # overrides from command line or config file
print(task,flush=True)

# guarantee a deterministic data generation process
random.seed(12345)

task_name = task.split('_')[0]
exec(open('task_'+task_name+'.py').read())

exec(open('configurator.py').read()) # re-override

# %% generate data
taskA_test = taskA_gen()
# assert everything is within vocab size
for d in taskA_test: assert all([0<=x<vocab_size for x in d])
# shuffle
random.shuffle(taskA_test)
assert len(taskA_test) > 0 #and len(taskB_test) > 0
print(f'Task A test: {len(taskA_test)}')
taskA_test_set = set(tuple(x) for x in taskA_test)

# %% generate run description
run_description = run_name + task+ ' lr'+str(learning_rate) + ' attn'+str(shaped_attention) + ' w'+str(n_embd)+' f'+str(weight_frozen)+' l'+str(n_layer)+' '+arch
# create folder task if not exists
if not os.path.exists(task):
    os.makedirs(task)
save_path = task+'/'+(random_str+' '+task+' '+' '+run_description).replace('(','_').replace(')','').replace(' ','_').replace(',','_')
print('Run',run_description)
print('Path',save_path)

# %% seed everything
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_model_config(args, training_args, vocab_size):
    """
    config_class = (
        modeling_gpt2.MixingGPTConfig
        if args.model_name_or_path == "gpt2"
        else AutoConfig
    )
    """
    attn_implementation = (
        # "eager" if True # training_args.shaped_attention in ["mixing", "shaped"]
        "eager" if training_args.shaped_attention in ["mixing", "shaped"]
        else "sdpa" # Double check default!
    )
    kwargs = {}
    if training_args.shaped_attention in ["mixing", "shaped"]:
        # Initialize weights from N(0, 1). Width-dependent scaling
        # is implemented in both llama and gpt2 module initializers.
        # Only need to be 1 in case of shaped or mixing attention.        
        kwargs["initializer_range"] = 0.02 # 1.
            
    import functools
    if args.model_name_or_path == "gpt2":
        config_getter = functools.partial(
            modeling_gpt2.MixingGPTConfig
        )
    elif args.model_name_or_path == "llama":
        config_getter = functools.partial(
            modeling_llama.MixingLlamaConfig
        )
    else:
        config_getter = functools.partial(
            AutoConfig.from_pretrained,
            model_name_or_path = args.model_name_or_path,
        )
    sample = gen_single()
    n_positions = len(sample)
    config = config_getter(        
        cache_dir=args.cache_dir,
        # revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        intermediate_size=args.n_embd*4, 
        num_attention_heads=args.n_head,
        shaped_attention=training_args.shaped_attention,
        max_position_embeddings=n_positions, #=args.max_seq_len,
        # attention_width = args.n_embd // args.n_head,
        skip_scaling = np.sqrt(0.5),
        attn_implementation = attn_implementation,
        activation_cminus = args.activation_cminus,
        # Initialize weights from N(0, 1). Width-dependent scaling
        # is implemented in both llama and gpt2 module initializers.
        #initializer_range = args.initializer_range,

        max_seq_len = n_positions, #args.max_seq_len,
        # eval_accumulation_steps=4,
        vocab_size=vocab_size,
        learning_rate=training_args.learning_rate,
        do_rope=training_args.do_rope,
        base_attn_mix=args.base_attn_mix,
        base_hidden_size=args.base_hidden_size,
        base_n_layer=args.base_n_layer,
        base_num_attention_heads=args.base_num_attention_heads,
        tau0=args.tau0,
        depth_alpha=args.depth_alpha,
        **kwargs,
    )
    
    # config.shaped_attention = training_args.shaped_attention
    config.attention_bias = True
    config.mlp_bias = True

    logger.warning(f"MODEL CONFIG: {config}")
    return config


parser = HfArgumentParser((utils.ScriptArguments, utils.TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# %% define the model
if arch == 'transformer':
    attn_implementation = (
        # "eager" if True # training_args.shaped_attention in ["mixing", "shaped"]
        "eager" if shaped_attention in ["mixing", "shaped"]
        else "sdpa" 
    )
    initializer_range = 0.02 # 1. if shaping in ["mixing", "shaped"] else 0.02
    sample = gen_single()
    n_positions = len(sample)
    """
    config = MixingGPTConfig(
        vocab_size=vocab_size,
        n_embd=1024,
        n_layer=n_layer,
        n_head=n_head,
        tie_word_embeddings=tie_emb,
        n_positions=n_positions,
        resid_pdrop = dropout,
        embd_pdrop = dropout,
        attn_pdrop = dropout,
        shaped_attention = shaping,
        skip_scaling = np.sqrt(0.5),
        attn_implementation = attn_implementation,
        activation_cminus = -1.,
        # Initialize weights from N(0, 1). Width-dependent scaling
        # is implemented in both llama and gpt2 module initializers.
        initializer_range = initializer_range,
    )
    """
    config = get_model_config(args, training_args, vocab_size)
    if args.model_name_or_path == "gpt2":
        
        model = modeling_gpt2.GPT2LMHeadModel(config).to(device)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"Model size: {model_size/1000**2:.1f}M parameters")
    elif args.model_name_or_path == "llama":
        import modeling_llama
        model = modeling_llama.LlamaForCausalLM(config).to(device)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"Model size: {model_size/1000**2:.1f}M parameters")
    else:    
        model = AutoModelForCausalLM.from_config(
            config=config,
            cache_dir=args.cache_dir,
            # revision=args.model_revision,        
        )
    model = model.to(device)
    if weight_frozen == 101:
        # initialize a sinusoidal position embedding
        n_pos = n_positions
        pos_emb = torch.zeros(n_pos, n_embd)
        position = torch.arange(0, n_pos).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-np.log(10000.0) / n_embd))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        # pos_emb = pos_emb.unsqueeze(0)
        model.transformer.wpe.weight.data = pos_emb
        print(pos_emb)
else:
    # Define the RNN model
    class VanillaRNN(torch.nn.Module):
        def __init__(self, vocab_size, n_embd, n_layer):
            super(VanillaRNN, self).__init__()
            self.encoder = torch.nn.Embedding(vocab_size, n_embd)
            self.rnn = {'rnn':torch.nn.RNN,'lstm':torch.nn.LSTM}[arch](n_embd, n_embd, n_layer, batch_first=True, dropout=dropout)
            self.decoder = torch.nn.Linear(n_embd, vocab_size)

        def forward(self, input_ids, labels=None, hidden=None):
            encoded = self.encoder(input_ids)
            output, hidden = self.rnn(encoded, hidden)
            decoded = self.decoder(output)
            return {'logits':decoded}#, 'hidden':hidden}

    # Instantiate the model
    model = VanillaRNN(vocab_size, n_embd, n_layer).to(device)
model_size = sum(t.numel() for t in model.parameters())
print(f'Arch: {arch}')
print(f"Model size: {model_size/1000**2:.1f}M parameters")


# %% build dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        super().__init__()
        self.input_ids = tokenized_data.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "labels": self.input_ids[idx]}


'''
Note that if it's a torch.utils.data.IterableDataset with some randomization
and you are training in a distributed fashion, your iterable dataset should
either use a internal attribute generator that is a torch.Generator for the
randomization that must be identical on all processes (and the Trainer will
manually set the seed of this generator at each epoch) or have a set_epoch()
method that internally sets the seed of the RNGs used.

Since we're currently not parallelizing we should be fine for now.
'''
    
class CustomIterDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()

    # def __len__(self):
    #     return train_size
    
    def __iter__(self):
        return iter(self.generate())

    def generate(self):
        while True:
            x = gen_single()
            if tuple(x) in taskA_test_set:
                continue
            yield {"input_ids": x,
                   "labels": x}
# Load your custom tokenized dataset (assuming each example is a list of token IDs)
# Padding is currently not implemented
datasets = {'trainA': None, 'testA': taskA_test}
datasets = {key: CustomDataset(tokenized_data=torch.tensor(val, dtype=torch.long))
            if val is not None and len(val) else CustomIterDataset() for key, val in datasets.items()}

print(datasets.keys())

for p,q in datasets.items():
    print(f'{p}: {q}')

print('Task A test:',len(datasets['testA']))

# %% some training utils

compute_metrics # it should be defined in the task file

print("batch size:", per_device_train_batch_size)
# %% training arguments
training_args_dict = dict(
    output_dir="./"+save_path,
    overwrite_output_dir=True,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    save_steps=save_steps,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    save_total_limit=save_total_limit,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    lr_scheduler_type=lr_scheduler_type,
    learning_rate=learning_rate,
    run_name=run_description,
)

# %% train
num_frozen = 0
num_trainable = 0

for n,p in model.named_parameters():
    if weight_frozen==2 and not any(x in n for x in ['lm_head', 'decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen==3 and not any(x in n for x in ['embed_tokens', 'wte','wpe','encoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen in [100,101] and not any(x in n for x in ['embed_tokens', 'wte','lm_head','encoder','decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen==1 and not any(x in n for x in ['embed_tokens', 'wte','lm_head','wpe','encoder','decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    else:
        print('trainable:',n,p.shape)
    if p.requires_grad:
        num_trainable += p.numel()
    else:
        num_frozen += p.numel()

print(f'# parameters: {num_trainable} trainable, {num_frozen} frozen')

"""
training_args = TrainingArguments(
    max_steps=train_steps*train_size//per_device_train_batch_size,
    **training_args_dict
)
"""
# training_args.max_steps=train_steps*train_size//args.per_device_train_batch_size,
for arg, val in training_args_dict.items():
    setattr(training_args, arg, val)
    
# The train_steps in previous repo is actually num_train_epochs.
training_args.max_steps=training_args.max_steps*train_size//per_device_train_batch_size

row_idxs = [0,1,2]   # e.g. the very first sentence
samples = []
for row_idx in row_idxs:
    raw_sample = datasets["testA"][row_idx]      # ↳ returns {"input_ids": ids, "labels": ids}
    # make it a *batch* (batch‑dim = 1) and keep only the keys the model expects
    sample = {k: v.unsqueeze(0) for k, v in raw_sample.items() if k in ("input_ids", "attention_mask", "labels")}
    samples.append(sample)

print(samples)

callbacks = []
if training_args.plot_attention_maps: 
    plot_cb = AttentionPlotCallback(
        example_texts=samples,
        output_dir=save_path + '/' + "plots", 
        show=False
    )
    callbacks.append(plot_cb)
    

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=(
        datasets['trainA']
    ),
    eval_dataset={'train_'+a:b for a,b in datasets.items() if a.startswith('test')},
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

if training_args.freeze_attention:
    for n,p in model.named_parameters():
        if "q_proj" in n or "k_proj" in n:
            print("freezing: ", n)
            p.requires_grad = False 
            
if training_args.freeze_mlp:
    for n,p in model.named_parameters():
        if "mlp" in n:
            print("freezing: ", n)
            p.requires_grad = False 
        
trainer.train()
logger.warning(f"MODEL CONFIG: {config}")
logger.warning(f"TRAINING args: {training_args} {args}")
metrics = trainer.evaluate()
print(metrics)
os.makedirs(task, exist_ok=True)
#with open(f'{task}/{shaped_attention}-{n_layer}-{n_head}.pkl', 'wb') as f:
#    pickle.dump(trainer.state.log_history, f)
