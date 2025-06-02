"""Main trainer class."""
import collections
import os
import logging
import sys

import numpy as np
# import streaming
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)

import modeling_gpt2
import modeling_llama
import modadd_exp_modular
import utils
import data_utils


logger = logging.getLogger(__name__)

TASK_TO_MODULE = {
    "modadd": modadd_exp_modular
}

TASK_TO_DATA_ARGS = {
    "modadd": {"prime": 199}
}
TASK_TO_TRAINER = {
    "modadd": modadd_exp_modular.CustomTrainer
}
TASK_TO_TRAINER = collections.defaultdict(lambda : transformers.Trainer, TASK_TO_TRAINER)

torch_perplexity = False # Currently computing perplexity directly from eval_loss
if torch_perplexity:
    # Requires torcheval package.
    from torcheval.metrics.text import Perplexity
    import torch.distributed as dist
    PERPLEXITY_METRIC = Perplexity()


def register_models(model_name):
    if model_name == "gpt2":
        AutoConfig.register("mixing_gpt", modeling_gpt2.MixingGPTConfig)
        AutoModel.register(modeling_gpt2.MixingGPTConfig, modeling_gpt2.GPT2LMHeadModel)
    else:
        # TODO: incorporate other models like llama
        pass
        

def compute_metrics(eval_pred, epsilon=1e-5):
    """
    This assumes that the pred have been pre-processed to only include
    the probability for the ground truth label (for memory saving).
    """
    # TODO: some implementations just use perplexity = math.exp(metrics["eval_loss"])?!
    pred, _ = eval_pred
    pred = np.clip(pred, min=epsilon)
    perp = np.mean(np.exp(-np.mean(np.log(pred.squeeze()), axis=-1)))
    # perp = np.exp(-np.mean(np.log(pred)))
    return {"perp": perp}

def compute_metrics_torch_perp(eval_pred):
    """This version currently causes OOM due to HF memory leak."""
    pred, labels = eval_pred
    # consider race conditions?!
    PERPLEXITY_METRIC.update(torch.tensor(pred), torch.tensor(labels))
    perp = PERPLEXITY_METRIC.compute()
    PERPLEXITY_METRIC.reset()
    return {"perplexity": perp}

def preprocess_logits_for_metrics(logits, labels):
    # Logits have shape e.g. [batch, seq_len, vocab_sz]
    labels = labels.unsqueeze(-1)
    # breakpoint()
    labels_orig = torch.tensor(labels)
    # Padding tokens have label -100
    labels = torch.clip(labels, min=0)
    logits = torch.softmax(logits, -1)
    logits = torch.gather(logits, -1, labels)
    
    logits = torch.masked_select(logits, labels_orig != -100)
    return logits

def tokenize_dataset(dataset, tokenizer, args):

    #    """pre-tokenize the dataset before training; only collate during training"""
    # TODO: update this to args.
    dataset_text_field = "text"
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding="max_length", # False,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        labels = torch.tensor(outputs["input_ids"])
        # breakpoint()
        # Ignore loss on pad tokens.
        labels[outputs["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs = {
            "input_ids": outputs["input_ids"],
            "labels": labels
        }

        return model_inputs
    
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        # num_proc=1, # training_args.dataset_num_proc,
    )

    return dataset


def get_train_dataloader_for_streaming(self):
    """
    Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    The plainest data loader is enough.
    """
    # Put import here again. Remove after environment issues are resolved.
    import streaming
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers, 
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    # Streaming is iterable so no need to set sampler etc.

    # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    self.train_dataloader = streaming.StreamingDataLoader(train_dataset, **dataloader_params)

    def _get_batch_size(cls, batch):
        # Because we changed how data loader works
        # the batch size count is not accurate, which affects data loading
        # return 1
        return self.args.per_device_train_batch_size


    self.train_dataloader._get_batch_size = _get_batch_size.__get__(self.train_dataloader, streaming.StreamingDataLoader)
    # breakpoint()

    assert self.train_dataset.replication is None, "Currently the dataset resuming on replication is not tested!"

    return self.train_dataloader


def get_eval_dataloader_for_streaming(self, eval_dataset):
    """
    Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    The plainest data loader is enough.
    """
    if eval_dataset is None and self.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")
    eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    data_collator = self.data_collator
    data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

    dataloader_params = {
        "batch_size": self.args.eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    # Streaming is iterable so no need to set sampler etc.

    # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    return streaming.StreamingDataLoader(eval_dataset, **dataloader_params) 

def get_learning_rate_scaling(args, shaped_attention):
    if shaped_attention in ['shaped', 'mixing'] and args.depth_alpha == 0: #TODO: this assumes that we do not train only readin and readout
        depth_scaling = args.base_n_layer / args.n_layer
        width_scaling = args.base_hidden_size / args.n_embd # TODO check if we need to add sqrt or not
        return depth_scaling * width_scaling     
    return 1.

def get_model_config(args, training_args, tokenizer):
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
    
    if training_args.scale_lr:
        training_args.learning_rate *= get_learning_rate_scaling(args, training_args.shaped_attention)
        
    # n_embd: int,
    # n_head: int,
    # n_layer: int,
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
    
    config = config_getter(        
        cache_dir=args.cache_dir,
        # revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        intermediate_size=args.n_embd*4, 
        num_attention_heads=args.n_head,
        shaped_attention=training_args.shaped_attention,
        max_position_embeddings=args.max_seq_len,
        # attention_width = args.n_embd // args.n_head,
        skip_scaling = args.skip_scaling,
        attn_implementation = attn_implementation,
        activation_cminus = args.activation_cminus,
        # Initialize weights from N(0, 1). Width-dependent scaling
        # is implemented in both llama and gpt2 module initializers.
        #initializer_range = args.initializer_range,

        max_seq_len = args.max_seq_len,
        # eval_accumulation_steps=4,
        vocab_size=len(tokenizer),
        learning_rate=training_args.learning_rate,
        do_rope=training_args.do_rope,
        base_attn_mix=args.base_attn_mix,
        base_hidden_size=args.base_hidden_size,
        base_num_hidden_layers=args.base_n_layer,
        base_num_attention_heads=args.base_num_attention_heads,
        tau0=1.,
        depth_alpha=args.depth_alpha,
        **kwargs,
    )
    
    # config.shaped_attention = training_args.shaped_attention
    config.max_seq_len = args.max_seq_len
    config.attention_bias = True
    config.mlp_bias = True

    logger.warning(f"MODEL CONFIG: {config}")
    return config


class EvalCallback(transformers.trainer_callback.TrainerCallback):
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # metrics = self.trainer.evaluate()
        # breakpoint()
        _add_model_perplexity(metrics)
        logger.warning(metrics)
        # print(metrics)

        
def setup(training_args):
    # initialize the process group
    if dist.get_rank() % torch.cuda.device_count() == 0:
        dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

    
def main():
    """Main function for running the trainer."""
    parser = HfArgumentParser((utils.ScriptArguments, utils.TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available else "cpu"
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    do_distributed = os.environ.get("WORLD_SIZE") is not None
    
    if do_distributed:
        # setup(training_args)
        torch.cuda.set_device(training_args.local_rank)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} world_size: {training_args.world_size} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if training_args.local_rank <= 0:
        logger.warning(f"Training/evaluation parameters {training_args}")
        logger.warning(f"Additional arguments {args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)
    # Cache mapped datasets.
    # datasets.set_caching_enabled()
    """
    # Tokenizer only used for natural language tasks.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    """

    is_streaming = args.task in ["dclm"]
    if is_streaming:
        # Conditional imports for now.
        import streaming
        import streaming_data
        # Multiplier for batch size, Roughly 8k / 1k
        training_args.streaming_effective_batch_size_multiplier = 8
        training_args.per_device_train_batch_size = max(1, training_args.per_device_train_batch_size // training_args.streaming_effective_batch_size_multiplier)
        training_args.per_device_eval_batch_size = max(1, training_args.per_device_eval_batch_size // training_args.streaming_effective_batch_size_multiplier)

    
    # To use llama tokenizer, you need to make sure you have access and that you are logged in. 
    # Request access in hugging face by selecting the models at this page: https://huggingface.co/meta-llama. 
    # Authenticate through command line: huggingface-cli login and copy/paste the token from your hugging face account: https://huggingface.co/settings/tokens    
    llama_tokenizer = "meta-llama/Llama-3.2-1B" if args.llama3 else "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path if args.model_name_or_path != "llama" else llama_tokenizer,
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length",
        # use_fast=args.use_fast_tokenizer,
    )
    # breakpoint()
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", truncation=True,
        max_length=training_args.response_length, trust_remote_code=model_args.trust_remote_code
    ) 
    """
    register_models(args.model_name_or_path)
    
    config = get_model_config(args, training_args, tokenizer)

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

    if do_distributed:
        device_id = dist.get_rank() % torch.cuda.device_count()
        model = model.to(device_id)
        # Enable find_unused_parameters to allow freezing specified parameters.
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    # if args.half_precision_training:
    #    model = model.to(half_dtype)

    # task_class = TASK_TO_MODULE[task]
    # data_args = TASK_TO_DATA_ARGS[task]

    # train_dataset, eval_dataset = task_class.get_train_eval_dataset(
    #    **data_args
    #)
    train_dataset, eval_dataset = data_utils.get_dataset(
        args.task,
        args=args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    # eval_dataset = eval_dataset.select(range(10))

    if not is_streaming:
        train_dataset = tokenize_dataset(train_dataset, tokenizer, args)
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, args)

    trainer_class = TASK_TO_TRAINER[args.task]
    training_args.eval_strategy = "steps"
    training_args.lr_scheduler_type = "linear" # "cosine"
    training_args.remove_unused_columns = False
    # training_args.optim_args = '{"min_lr_ratio": 0.1}'
    
    # This shouldn't be necessary, but causes DDP eval to not use labels is unspecified.
    training_args.label_names = ["labels"]
    # data_collator = data_utils.DataCollator(tokenizer)
    # effective_batch_size_multiplier = effective_batch_size_multiplier
    if is_streaming:
        data_collator = streaming_data.DataCollator(
            args,
            training_args=training_args,
            # think_token_id=think_token_id if args.think_token else None,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # effective_batch_size_multiplier=streaming_effective_batch_size_multiplier
            # mask_token_id=mask_token_id if args.do_mlm else None,
        )
        kwargs = {"data_collator": data_collator}
    else:
        kwargs = {}

    eval_logger = EvalCallback()

    # optim = torch.optim.AdamW
    # optimizer_cls_and_kwargs = (optim, {"min_lr_ratio": 0.1})
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # if training_args.do_train else None,
        eval_dataset=eval_dataset, # if training_args.do_eval else None,
        tokenizer=tokenizer,
        # optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[eval_logger],
        **kwargs
    )
    trainer_addon(trainer, args, training_args)
    
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
    metrics = trainer.evaluate()
    _add_model_perplexity(metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    # trainer.save_state()
    
    if do_distributed:
        cleanup()

    
def trainer_addon(trainer, args, training_args):
    if args.task in ["dclm"]:
        trainer.get_train_dataloader = get_train_dataloader_for_streaming.__get__(
            trainer, transformers.Trainer
        )
        trainer.get_eval_dataloader = get_eval_dataloader_for_streaming.__get__(
            trainer, transformers.Trainer
        )


def _add_model_perplexity(metrics):
    try:        
        perplexity = np.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["eval_perplexity"] = perplexity

if __name__ == "__main__":
    main()
