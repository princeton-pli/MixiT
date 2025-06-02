"""Scripts for retrieving various LM datasets."""
# from datasets import load_dataset, Dataset
import datasets
import torch


def get_num_classes(dataset):
    """Gets number of label classes."""
    if dataset in ["yelp", "yelp_tiny"]:
        return 5
    elif dataset in ["yelp_polarity"]:
        return 2
    else:
        raise ValueError(f"get_num_classes is not supported for {dataset}.")

def get_yelp_polarity():
    """yelp reviews dataset."""
    # wikitext-103-v1	
    # wikitext = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-v1",
    #                                 split='train,test[:16]')
    train_data = datasets.load_dataset("fancyzhx/yelp_polarity", 
                                       split='train')
    test_data = datasets.load_dataset("fancyzhx/yelp_polarity",
                                       split='test')
    """
    train_size = 2**18 # 2**15
    test_size = 2**10
    train_data = train_data.select(range(train_size))
    test_data = test_data.select(range(test_size))
    """
    # train_data = wikitext["train"]
    # test_data = wikitext["test"]
    # breakpoint()
    return train_data, test_data

def get_yelp():
    """yelp reviews dataset."""
    # wikitext-103-v1	
    # wikitext = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-v1",
    #                                 split='train,test[:16]')
    train_data = datasets.load_dataset("Yelp/yelp_review_full", 
                                       split='train')
    test_data = datasets.load_dataset("Yelp/yelp_review_full",
                                       split='test')
    """
    train_size = 2**18 # 2**15
    test_size = 2**10
    train_data = train_data.select(range(train_size))
    test_data = test_data.select(range(test_size))
    """
    # train_data = wikitext["train"]
    # test_data = wikitext["test"]
    # breakpoint()
    return train_data, test_data

def get_yelp_tiny():
    """yelp reviews dataset."""
    # wikitext-103-v1	
    # wikitext = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-v1",
    #                                 split='train,test[:16]')
    train_data = datasets.load_dataset("Yelp/yelp_review_full", 
                                       split='train')
    test_data = datasets.load_dataset("Yelp/yelp_review_full",
                                       split='test')

    train_size = 2**10 # 2**15
    test_size = 2**6
    train_data = train_data.select(range(train_size))
    test_data = test_data.select(range(test_size))

    # train_data = wikitext["train"]
    # test_data = wikitext["test"]
    # breakpoint()
    return train_data, test_data


def get_wikitext_2():
    """wikitext 2."""
    # wikitext-103-v1	
    # wikitext = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-v1",
    #                                 split='train,test[:16]')
    train_data = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-v1",
                                       split='train')
    test_data = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-v1",
                                       split='test')
    
    # train_data = wikitext["train"]
    # test_data = wikitext["test"]
    # breakpoint()
    return train_data, test_data


def get_fineweb_edu():
    train_data = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)

    train_size = 2**20 # 2**15
    test_size = 2**12
    data1 = train_data.select(range(train_size))
    data2 = train_data.select(range(train_size, train_size + test_size))
    # TODO: make train / test split
    # test_data = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="test", streaming=True)
    # test_data = train_data
    # return train_data, test_data
    return data1, data2


def get_wikitext():
    train_data = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                       split='train')
    test_data = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                       split='test')
    
    return train_data, test_data

def get_dclm(args, training_args, tokenizer):
    # Conditional imports before environment issues are resolved.
    import streaming_data
    args.domains_and_proportions_train = args.domains_and_proportions_train.replace('=', ':')
    args.domains_and_proportions_val = args.domains_and_proportions_val.replace('=', ':')
    # train_batch_size = max(1, training_args.per_device_train_batch_size // training_args.streaming_effective_batch_size_multiplier)
    # eval_batch_size = max(1, training_args.per_device_eval_batch_size // training_args.streaming_effective_batch_size_multiplier)
    train_batch_size = training_args.per_device_train_batch_size
    eval_batch_size = training_args.per_device_eval_batch_size

    train_dataset = streaming_data.get_multiple_domain_dataset(
        root_dir=args.streaming_train_root, 
        domains_and_proportions=args.domains_and_proportions_train, 
        shuffle=True, 
        # remote=args.streaming_remote, # defaults to False
        block_size=args.max_seq_len, 
        tokenizer=tokenizer, 
        # one_to_many_ratio=args.one_to_many_ratio_train, 
        batch_size=train_batch_size,
        # uint16_data=args.uint16_data,
        # return_indices=args.return_indices, # Defaults to False
        # sort_by_length_mega_batch=args.sort_by_length_steps *
        #    args.gradient_accumulation_steps *
        #    args.per_device_train_batch_size // max(args.dataloader_num_workers, 1)
    )
    
    eval_dataset = streaming_data.get_multiple_domain_dataset(
        root_dir=args.streaming_val_root, 
        domains_and_proportions=args.domains_and_proportions_val, 
        shuffle=False, 
        # remote=args.streaming_remote, # defaults to False
        block_size=args.max_seq_len, 
        tokenizer=tokenizer, 
        # one_to_many_ratio=args.one_to_many_ratio_train, 
        batch_size=eval_batch_size,
    )
    return train_dataset, eval_dataset
    

def get_dataset(task: str, args, training_args,
                tokenizer=None):
    if task == "wikitext":
        return get_wikitext()
    elif task == "wikitext2":
        return get_wikitext_2()
    elif task == "dclm":
        return get_dclm(args, training_args, tokenizer)
    elif task == "fineweb":
        return get_fineweb_edu()
    elif task == "yelp":
        return get_yelp()
    elif task == "yelp_tiny":
        return get_yelp_tiny()
    elif task == "yelp_polarity":
        return get_yelp_polarity()
    else:
        raise ValueError(f"Task {task} not supported.")

class DataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    @torch.no_grad()
    def __call__(self, features):
        input_ids = []
        labels = []
        
        input_ids = self.tokenizer(features)
        labels = input_ids.astype(torch.long)
        return input_ids, labels
