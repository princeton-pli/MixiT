# Attention Retrieves, MLP Memorizes: Disentangling Trainable Components in the Transformer

What roles do different components of the Transformer play?

Our work disentangles the different trainable components with respect to different tasks, and finds that:
* A Transformer model with *random query and key projectors*, frozen at initialization, can perform competitively on language modeling tasks. 
* MLPs play a crucial role in memorization, whereas attention is crucial for forming specialized circuits such as induction heads, enabling tasks such as retrieval. MLPs and attention collaborate on memorization tasks.
* Even a model with a *completely random, input-independent attention scores matrix*, can solve certain algorithmic and reasoning tasks.

Our work takes a principled approch towards randomly initializing the frozen weights.
    
> **[Paper](https://arxiv.org/pdf/2506.01115)**

![Architecture overview](models_diagram.svg)
    
## 🚀 Setup

1. Clone this repository
```
git clone https://github.com/princeton-pli/MixiT
```
2. `cd` into the `MixiT` directory
3. Run `pip install -r requirements.txt` to install the necessary dependencies

## 🛠️ Usage
    
The code is designed to be modular and scalable, allowing for fast and efficient distributed training.

Note that the code for the algorithmic tasks is based on previous work on the [random transformer](https://github.com/fjzzq2002/random_transformers/tree/main), but with more systematized runner scripts.


To launch experiments in various settings, either algorithmic or language modeling experiments, we can use commands such as the following:
    
Language modeling:
```
python trainer.py --task wikitext --output_dir results --cache_dir /tmp/hf_cache --max_seq_len 256 --max_steps=40000 --shaped_attention=mixing --eval_steps 400 --logging_steps=400 --n_layer=12 --n_embd=512 --n_head=8 --learning_rate=5e-4 --per_device_train_batch_size=512 --model_name_or_path=llama --llama3=True
```

Yelp sentiment classification:
```
python trainer_classifier.py --task yelp_polarity --output_dir results --cache_dir /tmp/hf_cache --max_seq_len 256 --max_steps=3000 --shaped_attention=mixing --eval_steps 1000 --logging_steps=1000 --n_embd=1024 --n_head=16 --learning_rate=5e-4 --model_name_or_path=llama --llama3=True
```

Algorithmic tasks:
```
python depth.py --task=additionstream_10 --max_steps=500 --eval_steps=100 --logging_steps=100 --weight_frozen=1 --n_layer=2 --shaped_attention=mixing --n_embd=512 --n_head=4
```
    
We also provide scripts for launching experiments on slurm. For instance:

Language modeling:    
```
shaped_attention=mixing llama3=True model_name_or_path=llama n_layer=12 max_steps=80000 max_seq_len=256 n_embd=512 n_head=8 per_device_train_batch_size=128 job_hours=24 n_gpu=4 learning_rate=5e-4 task=wikitext scripts/run_train.sh
```

Yelp sentiment classification:
```
shaped_attention=vanilla master_port=29501 llama3=True model_name_or_path=llama n_layer=4 max_steps=10000 max_seq_len=256 n_embd=512 n_head=8 per_device_train_batch_size=128 job_hours=12 n_gpu=2 learning_rate=8e-4 postfix=alpha task=yelp_polarity activation_cminus=-1 weight_decay=0.07 scripts/run_train_classifier.sh
```

Algorithmic tasks:
```
shaped_attention=mixing weight_frozen=0 llama3=True model_name_or_path=llama n_layer=2 eval_steps=1000 logging_steps=1000 max_steps=300 max_seq_len=256 n_embd=512 n_head=8 per_device_train_batch_size=256 learning_rate=5e-4 task=dyckstream_40 scripts/run_depth_array.sh
```
    
More command line arguments can be found in [utils.py](utils.py).

### 👋 Citation
```
@article{mixit2025,
  title={Attention Retrieves, MLP Memorizes: Disentangling Trainable Components in the Transformer},
  author={Dong, Yihe and Noci, Lorenzo and Khodak, Mikhail and Li, Mufan},
  journal={arXiv preprint arXiv:2506.01115},
  year={2025}
}
```
