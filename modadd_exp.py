#!/usr/bin/env python
# coding: utf-8

# %%
import torch
# from transformers import GPT2Config #, GPT2LMHeadModel
from transformers.trainer_callback import TrainerCallback
import wandb
from modeling_gpt2 import GPT2LMHeadModel, MixingGPTConfig
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset, Dataset
import random
import numpy as np
import wandb
import os

import utils
import logging
import modeling_gpt2
import modeling_llama
import utils
from modeling_gpt2 import GPT2LMHeadModel, MixingGPTConfig
logger = logging.getLogger(__name__)
    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # print(inputs)
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # print(labels.shape,logits.shape)
        labels = labels[:,-1]
        logits = logits[:,-2]
        # print(labels.shape,logits.shape)
        # print(logits.view(-1, logits.shape[-1]).shape)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    
# %% build dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "labels": self.input_ids[idx],
                "attention_mask": torch.full_like(self.input_ids[idx],1)}



from sklearn.decomposition import PCA
class L2NormCallback(TrainerCallback):
    def __init__(self, phase):
        self.phase = phase
        self.pcas = []
        self.all_metrics = []
        # self.step0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        # self.step0 = state.global_step
        pass

    def on_step_end(self, args, state, control, **kwargs):
        # if state.global_step == self.step0:
        #     return
        model = kwargs['model']
        """
        l2_norm = sum([p.norm()**2 for p in model.parameters()]).sqrt()
        wandb.log({'l2_norm_all': l2_norm.item()})
        wandb.log({self.phase+'_l2_norm_all': l2_norm.item()})
        l2_norm_noln = sum([p.norm()**2 for n,p in model.named_parameters() if '.ln' not in n]).sqrt()
        wandb.log({'l2_norm': l2_norm_noln.item()})
        wandb.log({self.phase+'_l2_norm': l2_norm_noln.item()})
        """
        if calc_pca:
            # calculate PCA of input embeddings
            embds = model.transformer.wte.weight
            pca = PCA(n_components=20)
            embd_PCA = pca.fit_transform(embds.detach().cpu().numpy())
            wandb.log({'PCA_explained_variance_ratio': pca.explained_variance_ratio_.tolist()})
            self.pcas.append(embd_PCA)
    
    def save_pca(self):
        import pickle
        pca_log = np.array(self.pcas)
        with open("./"+save_path+'/pca_'+self.phase+'.pickle', 'wb') as f:
            pickle.dump(pca_log, f)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero:

            self.all_metrics.append(metrics)


def get_train_eval_dataset(**kwargs):
    # P = int(task_split[1])
    fracA = 0.95
    P = int(kwargs["prime"])
    vocab_size = P
    def taskA_gen():
        return [[a,b,(a+b)%P] for a in range(P) for b in range(P)]
    full_train_size = int(P*P*fracA)

    # %% generate data
    taskA_data = taskA_gen()
    # assert everything is within vocab size
    for d in taskA_data: assert all([0<=x<vocab_size for x in d])
    # shuffle
    random.shuffle(taskA_data)
    # split into train and test for both tasks, first {full_train_size} used for training
    taskA_train = taskA_data[:full_train_size]
    taskA_test = taskA_data[full_train_size:]
    assert len(taskA_test) > 0 #and len(taskB_test) > 0
    print(f'Task A train: {len(taskA_train)} (of {len(taskA_train)}), test: {len(taskA_test)}')
    taskA_train_len = len(taskA_train)
    taskA_test_len = len(taskA_test)
    return taskA_train, taskA_test



fracA = 0.95
train_steps = 100 # 5000
seed = 0 #random.randint(0,10**9)
n_embd = 256
n_layer = 2 #30 # 2
n_head = 4
n_positions = 5
device = 'cuda'
weight_decay = 0.001
learning_rate = 1e-3
warmup_steps = 50 #500
save_steps = 600
eval_steps = 400 # 50
logging_steps = 600
save_total_limit = 3
evaluation_strategy = "steps"
lr_scheduler_type = "cosine"
random_str = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8)])
task = 'modadds_199'
# task = 'modadds_599'

calc_pca = False #True #False
tie_emb = False
weight_frozen = 1

# Parameters to change for the mixing transformer:
shaped_attention = "mixing" # "shaped"
# The negative slope for leaky relu will be 1 + activation_cminus / \sqrt(width) 
activation_cminus = -1

# # %% config from command line
exec(open('configurator.py').read()) # overrides from command line or config file
print(task,flush=True)


# guarantee a deterministic data generation process
random.seed(12345)

# %% specify the task
full_train_size = 0
if task.startswith('modadds_'):
    task_split = task.split('_')
    if len(task_split) == 2:
        P = int(task_split[1])
        vocab_size = P
        def taskA_gen():
            return [[a,b,(a+b)%P] for a in range(P) for b in range(P)]
        full_train_size = int(P*P*fracA)
        print('haha')

assert full_train_size > 0
# everything fits!
per_device_train_batch_size = 4000
per_device_eval_batch_size = 4000

# %% generate data
taskA_data = taskA_gen()
# assert everything is within vocab size
for d in taskA_data: assert all([0<=x<vocab_size for x in d])
# shuffle
random.shuffle(taskA_data)
# split into train and test for both tasks, first {full_train_size} used for training
taskA_train = taskA_data[:full_train_size]
taskA_test = taskA_data[full_train_size:]
assert len(taskA_test) > 0 #and len(taskB_test) > 0
print(f'Task A train: {len(taskA_train)} (of {len(taskA_train)}), test: {len(taskA_test)}')
taskA_train_len = len(taskA_train)
taskA_test_len = len(taskA_test)

# %% generate run description
run_description = task+' w'+str(n_embd)+' f'+str(weight_frozen)
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

# %% init wandb
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(config)
wandb.init(project='rand_transformer_vwfn_modadd', name=run_description, config=config)
wandb.run.log_code(".")

# %% define the model
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
    # sample = gen_single()
    n_positions = len(taskA_train[0]) # len(sample)

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
        tau0=1.,
        **kwargs,
    )
    
    # config.shaped_attention = training_args.shaped_attention
    config.attention_bias = True
    config.mlp_bias = True

    logger.warning(f"MODEL CONFIG: {config}")
    return config

parser = HfArgumentParser((utils.ScriptArguments, utils.TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()
"""
config = MixingGPTConfig( # GPT2Config(
    vocab_size=vocab_size,
    n_embd=n_embd,
    intermediate_size=n_embd*6,
    n_layer=n_layer,
    n_head=n_head,
    tie_word_embeddings=tie_emb,
    n_positions=n_positions,
    resid_pdrop = 0.0,
    embd_pdrop = 0.0,
    attn_pdrop = 0.0,    
)
"""
    
config = get_model_config(args, training_args, vocab_size)
# breakpoint()
"""
config._attn_implementation = "eager"
config._attn_implementation_autoset = False
# Mixing transformer
config.shaped_attention = shaped_attention
config.activation_cminus = activation_cminus
"""
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
    raise ValueError("Model not supported.")

if shaped_attention in ["mixing", "shaped"]:
    # figure out seq len
    config.attention_width = n_embd // n_head
    seq_len = 3 # each seq is [a, b, (a+b)%p]
    # config.mixing_noise = torch.normal(torch.zeros(seq_len, seq_len), std=1./(seq_len * width))
    config.max_seq_len = 3
    config.skip_scaling = np.sqrt(0.5)
    # Initialize weights from N(0, 1)
    config.initializer_range = 0.02

# model = GPT2LMHeadModel(config).to(device)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")


# Load your custom tokenized dataset (assuming each example is a list of token IDs)
# Padding is currently not implemented
datasets = {'trainA': taskA_train, 'testA': taskA_test}
datasets = {key: CustomDataset(tokenized_data=torch.tensor(val, dtype=torch.long)) for key, val in datasets.items() if len(val)}

print(datasets.keys())

for p,q in datasets.items():
    print(f'{p}: {len(q)}')

# %% some training utils

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)[...,-2]
    corr = predictions == labels[...,-1]
    return {'accuracy': np.mean(corr)}


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
for arg, val in training_args_dict.items():
    setattr(training_args, arg, val)

# The train_steps in previous repo is actually num_train_epochs.
training_args.num_train_epochs = training_args.max_steps #.max_steps=training_args.max_steps*train_size//per_device_train_batch_size

# %% train
wandb.log({'phase': 'train'})
num_frozen = 0
num_trainable = 0

for n,p in model.named_parameters():
    if weight_frozen==2 and not any(x in n for x in ['lm_head', 'decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen==3 and not any(x in n for x in ['wte','wpe','encoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen in [100,101] and not any(x in n for x in ['wte','lm_head','encoder','decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen==1 and not any(x in n for x in ['wte','lm_head','wpe','encoder','decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    else:
        print('trainable:',n,p.shape)
    if p.requires_grad:
        num_trainable += p.numel()
    else:
        num_frozen += p.numel()

print(f'# parameters: {num_trainable} trainable, {num_frozen} frozen')
wandb.log({'num_trainable': num_trainable})
wandb.log({'num_frozen': num_frozen})

"""
training_args = TrainingArguments(
    num_train_epochs=train_steps,
    **training_args_dict
)
"""

train_logger = L2NormCallback('train')

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=(
        datasets['trainA']
    ),
    eval_dataset={'train_'+a:b for a,b in datasets.items()},
    compute_metrics=compute_metrics,
    callbacks=[train_logger],
)

trainer.train()
logger.warning(f"{training_args} {args}")
trainer.evaluate()

# save metrics
import pickle, datetime
cur_time = datetime.datetime.now()
cur_time = datetime.datetime.isoformat(cur_time)

file_path = f"results/{task}_{config.shaped_attention}{cur_time}.res"
#with open(file_path, "wb") as fn:
#    pickle.dump(train_logger.all_metrics, fn, protocol=pickle.HIGHEST_PROTOCOL)

