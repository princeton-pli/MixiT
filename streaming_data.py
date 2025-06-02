from transformers.utils import logging
import torch
from torch.utils.data import Dataset
import numpy as np
from streaming import LocalDataset, StreamingDataset, Stream
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
from collections.abc import Mapping
from streaming.base.world import World
from itertools import islice
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Iterator

logger = logging.get_logger(__name__)

class MDSDataset(StreamingDataset):

    def __init__(self, block_size, tokenizer=None, one_to_many_ratio=None, return_indices=False, uint16_data=False, sort_by_length_mega_batch=1, **kwargs):
        # kwargs["num_canonical_nodes"] = 1
        super().__init__(**kwargs)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.return_indices = return_indices
        self.uint16_data = uint16_data

        # Let's say we have a 32k length dataset but we only want to use 4k training length,
        # we can break it into multiple datasets
        if one_to_many_ratio is not None:
            logger.warn(f"Use one_to_many_ratio: {one_to_many_ratio}")
        self.one_to_many_ratio = one_to_many_ratio
        self.sort_by_length_mega_batch = sort_by_length_mega_batch

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if "indices" in item:
            return self.__getitem_indices__(item)
        else:
            return self.__getitem_first_seq__(item)
        
    def __getitem_indices__(self, item):

        # Normal pre-training dataset: "tokens"
        # The dataset should wrap each document with <s></s>. The input does not necessarily start with <s>.
        # When truncating, can start from anywhere
        # if self.uint16_data:
        #    tokens = np.frombuffer(item["tokens"], np.uint16).astype(np.int64)
        #else:
        tokens_orig = item["input_ids"].astype(np.int64)
        
        if "mask" in item:
            # if self.uint16_data:
            #    mask = np.frombuffer(item["mask"], np.ubyte).astype(np.bool_)
            # else:
            mask = item["mask"].astype(np.bool_) 
        
        # Truncate
        """
        if self.one_to_many_ratio is not None:
            start_id = offset * self.block_size
            tokens = tokens[start_id:start_id+self.block_size]
            if "mask" in item:
                mask = mask[start_id:start_id+self.block_size]
        elif len(tokens) > self.block_size:
        """
        #  start_id = 0 # np.random.randint(0, len(tokens) - self.block_size + 1)
        #if "indices" in item:
        for (a, b) in item["indices"]:            
            end_id = min(b, a + self.block_size)
            end_token_index = end_id - a
            tokens = tokens_orig[a : a+self.block_size]
            if len(tokens) < self.block_size:
                tokens = np.concatenate([tokens, np.zeros(self.block_size-len(tokens), dtype=np.int64)])
            indices = np.array([[0, len(tokens)]], dtype=np.int64)
            
            if "mask" in item:
                labels = tokens + 0
                labels[~mask] = -100 # Do not train on the mask=0 part (user input)
                if mask.sum() == 0:
                    # If the mask is all 0 we might get nan loss
                    labels[-10:] = tokens[-10:] + 0
            else:
                labels = tokens

            tokens[end_token_index:] = self.tokenizer.pad_token_id
            labels[end_token_index:] = -100
            if self.return_indices:
                yield {"input_ids": tokens, "labels": labels, "indices": indices}
            else:
                yield {"input_ids": tokens, "labels": labels}
        
        
    def __getitem_first_seq__(self, item):

        # Normal pre-training dataset: "tokens"
        # The dataset should wrap each document with <s></s>. The input does not necessarily start with <s>.
        # When truncating, can start from anywhere
        # if self.uint16_data:
        #    tokens = np.frombuffer(item["tokens"], np.uint16).astype(np.int64)
        #else:
        tokens = item["input_ids"].astype(np.int64)
        
        if "mask" in item:
            # if self.uint16_data:
            #    mask = np.frombuffer(item["mask"], np.ubyte).astype(np.bool_)
            # else:
            mask = item["mask"].astype(np.bool_) 
        
        # Truncate
        """
        if self.one_to_many_ratio is not None:
            start_id = offset * self.block_size
            tokens = tokens[start_id:start_id+self.block_size]
            if "mask" in item:
                mask = mask[start_id:start_id+self.block_size]
        elif len(tokens) > self.block_size:
        """
        start_id = 0 # np.random.randint(0, len(tokens) - self.block_size + 1)
        if "indices" in item:
            # indices is a 2D array.
            end_token_index = min(start_id+self.block_size, indices[0][1])
        else:
            end_token_index = start_id+self.block_size
            
        tokens = tokens[start_id:start_id+self.block_size]
        
        if "mask" in item:
            mask = mask[start_id:start_id+self.block_size]
        # else:
        #    start_id = 0
        breakpoint()
        """
        # Indices (for varlen attn)
        if "indices" in item:
            indices = item["indices"] 
            if start_id > 0 or indices[-1][1] > len(tokens):
                # Need to remove the indices that are before start_id or after start_id+block_size
                end_id = start_id + len(tokens)
                new_indices = []
                for (a, b) in indices:
                    if start_id >= b:
                        continue
                    if start_id > a:
                        a = start_id
                    if a >= end_id:
                        break
                    b = min(b, end_id)

                    new_indices.append((a, b))
        
                new_indices = np.array(new_indices, dtype=np.int64)
                new_indices = new_indices - start_id
                indices = new_indices
        else:
            indices = np.array([[0, len(tokens)]], dtype=np.int64)
        """
        indices = np.array([[0, len(tokens)]], dtype=np.int64)
        if "mask" in item:
            labels = tokens + 0
            labels[~mask] = -100 # Do not train on the mask=0 part (user input)
            if mask.sum() == 0:
                # If the mask is all 0 we might get nan loss
                labels[-10:] = tokens[-10:] + 0
        else:
            labels = tokens

        tokens[end_token_index:] = self.tokenizer.pad_token_id
        labels[end_token_index:] = -100

        if self.return_indices:
            return {"input_ids": tokens, "labels": labels, "indices": indices}
        else:
            return {"input_ids": tokens, "labels": labels}
        
    def __getitem__0(self, idx):
        if self.one_to_many_ratio is not None: 
            epoch = self.next_epoch - 1 
            offset = epoch % self.one_to_many_ratio

        item = super().__getitem__(idx)

        # Normal pre-training dataset: "tokens"
        # The dataset should wrap each document with <s></s>. The input does not necessarily start with <s>.
        # When truncating, can start from anywhere
        if self.uint16_data:
            tokens = np.frombuffer(item["tokens"], np.uint16).astype(np.int64)
        else:
            tokens = item["input_ids"].astype(np.int64)
        
        if "mask" in item:
            if self.uint16_data:
                mask = np.frombuffer(item["mask"], np.ubyte).astype(np.bool_)
            else:
                mask = item["mask"].astype(np.bool_) 
        
        # Truncate
        if self.one_to_many_ratio is not None:
            start_id = offset * self.block_size
            tokens = tokens[start_id:start_id+self.block_size]
            if "mask" in item:
                mask = mask[start_id:start_id+self.block_size]
        elif len(tokens) > self.block_size:
            start_id = 0 # np.random.randint(0, len(tokens) - self.block_size + 1)
            tokens = tokens[start_id:start_id+self.block_size]
            if "mask" in item:
                mask = mask[start_id:start_id+self.block_size]
        else:
            start_id = 0
        
        # Indices (for varlen attn)
        if "indices" in item:
            indices = item["indices"] 
            if start_id > 0 or indices[-1][1] > len(tokens):
                # Need to remove the indices that are before start_id or after start_id+block_size
                end_id = start_id + len(tokens)
                new_indices = []
                for (a, b) in indices:
                    if start_id >= b:
                        continue
                    if start_id > a:
                        a = start_id
                    if a >= end_id:
                        break
                    b = min(b, end_id)

                    new_indices.append((a, b))
        
                new_indices = np.array(new_indices, dtype=np.int64)
                new_indices = new_indices - start_id
                indices = new_indices
        else:
            indices = np.array([[0, len(tokens)]], dtype=np.int64)
        
        if "mask" in item:
            labels = tokens + 0
            labels[~mask] = -100 # Do not train on the mask=0 part (user input)
            if mask.sum() == 0:
                # If the mask is all 0 we might get nan loss
                labels[-10:] = tokens[-10:] + 0
        else:
            labels = tokens

        if self.return_indices:
            return {"input_ids": tokens, "labels": labels, "indices": indices}
        else:
            return {"input_ids": tokens, "labels": labels}


    def __iter__(self) -> Iterator[Dict[str, Any]]:
        iterator = super().__iter__()
        if self.sort_by_length_mega_batch <= 1:
            yield from iterator
        else:
            # Not tested yet!!
            raise NotImplementedError
            while True:
                block = list(islice(iterator, self.sort_by_length_mega_batch))  # Get a block of items
                if not block:
                    return  # Stop when iterator is exhausted

                yield from sorted(block, key=(lambda item: sum((b - a)**2 for b, a in item["indices"])), reverse=True)  # Yield sorted block items


def get_multiple_domain_dataset(
    root_dir,
    shuffle,
    domains_and_proportions,
    remote=False,
    block_size=None,
    tokenizer=None,
    one_to_many_ratio=None,
    return_indices=False,
    uint16_data=False,
    sort_by_length_mega_batch=1,
    **kwargs,
):
    if isinstance(domains_and_proportions, str):
        domains_and_proportions = eval(domains_and_proportions.replace("\n", ""))

    flatten = {}
    def dfs(d, ratio):
        for k, v in d.items():
            if isinstance(v, dict):
                dfs(v, ratio*float(k))
            else:
                flatten[k] = ratio * v
    dfs(domains_and_proportions, 1.0)
    domains_and_proportions = flatten

    logger.warning("Loading multiple domain dataset via MosaicML streaming.")
    logger.warning("***** Streaming dataset *****")
    logger.warning(f"Root dir: {root_dir}")
    logger.warning(f"Shuffle: {shuffle}")
    logger.warning(f"Domains: {domains_and_proportions}")
    logger.warning(f"Remote: {remote}")
    logger.warning(f"Block size: {block_size}")

    if remote:
        streams = [
            Stream(remote=root_dir+domain, proportion=domains_and_proportions[domain])
            for domain in domains_and_proportions
        ]
    else:
        streams = [
            Stream(local=os.path.join(root_dir, domain), proportion=domains_and_proportions[domain])
            for domain in domains_and_proportions
        ]

    dataset = MDSDataset(
        block_size=block_size,
        streams=streams,
        shuffle=shuffle,
        tokenizer=tokenizer,
        one_to_many_ratio=one_to_many_ratio,
        return_indices=return_indices,
        uint16_data=uint16_data,
        sort_by_length_mega_batch=sort_by_length_mega_batch,
        **kwargs
    )

    return dataset



def torch_mask_tokens(inputs, mask_rate, bos_token_id, eos_token_id, mask_token_id):

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mask_rate)
    special_tokens_mask = labels.eq(bos_token_id) | labels.eq(eos_token_id)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    inputs[masked_indices] = mask_token_id

    return inputs, labels

class DataCollator:

    def __init__(self, data_args, training_args=None, think_token_id=None, bos_token_id=None, eos_token_id=None, mask_token_id=None):
        self.args = data_args
        self.training_args = training_args
        self.think_token_id = think_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id

    def __call__(self, features) -> Dict[str, Any]:

        is_generator = False
        import typing
        if isinstance(features[0], typing.Generator):
            is_generator = True
            # first = next(features[0])
        else:
            if not isinstance(features[0], Mapping):

                features = [vars(f) for f in features]
            first = features[0]
        batch = {}

        
        if is_generator:            
            # Handling of all other possible keys.
            # Again, we will use the first element to figure out which key/values are not None for this model.
            input_ids = []
            labels = []
            effective_batch_size = len(features) * self.training_args.streaming_effective_batch_size_multiplier
            for sequences in features:
                # Actual Keys are 'input_ids', 'labels'
                # Each sequence is a variable length generator.
                for seq in sequences:
                    input_ids.append(seq['input_ids'])
                    labels.append(seq['labels'])

                if len(labels) > effective_batch_size:
                    break
            if isinstance(input_ids[0], np.ndarray):
                batch['input_ids'] = torch.tensor(np.stack(input_ids[:effective_batch_size]))
                batch['labels'] = torch.tensor(np.stack(labels[:effective_batch_size]))
            else:
                batch['input_ids'] = torch.stack(input_ids[:effective_batch_size])
                batch['labels'] = torch.stack(labels[:effective_batch_size])
            """
            for k, v in first.items():
                
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
            """
        elif "indices" in first:
            # Varlen attn
            assert "attention_mask" not in first
            input_ids = []
            labels = []
            seq_lengths = []

            for item in features:
                for a, b in item["indices"]:
                    if b - a <= 1:
                        continue

                    input_ids.append(torch.tensor(item["input_ids"][a:b], dtype=torch.long))
                    labels.append(torch.tensor(item["labels"][a:b], dtype=torch.long))
                    seq_lengths.append(b-a)

            input_ids = torch.concat(input_ids, dim=0)
            labels = torch.concat(labels, dim=0)
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
            batch = {
                "input_ids": input_ids,
                "attention_mask": None,
                "labels": labels,
                "seq_lengths": seq_lengths
            }
        else:
            # Padding
            # Handling of all other possible keys.
            # Again, we will use the first element to figure out which key/values are not None for this model.
            for k, v in first.items():
                # Actual Keys are 'input_ids', 'labels'
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
        
        if getattr(self.args, "mask_rate", None) is not None:
            batch["input_ids"], batch["labels"] = torch_mask_tokens(
                batch["input_ids"], 
                self.args.mask_rate, 
                self.bos_token_id, 
                self.eos_token_id, 
                self.mask_token_id
            )

        return batch
