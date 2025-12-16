import json
import math
import os
import random
from functools import partial

import fire
import numpy as np
import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

from data import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset


class TokenExtender:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(self.index_path, "r") as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        new_tokens = set()
        for token_list in self.indices.values():
            for token in token_list:
                new_tokens.add(token)
        self.new_tokens = sorted(new_tokens)
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
):
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
        "merlin": "items",
    }
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    if micro_batch_size <= 0:
        raise ValueError("--micro_batch_size must be > 0")
    gradient_accumulation_steps = max(1, math.ceil(batch_size / micro_batch_size))
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        gradient_accumulation_steps = max(1, gradient_accumulation_steps)

    if torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        use_bf16 = bool(is_bf16_supported)
        use_fp16 = not use_bf16
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        use_bf16 = False
        use_fp16 = False
        torch_dtype = torch.float32

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    original_vocab_size = len(tokenizer)
    new_tokens = []
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(index_path=sid_index_path)
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
    elif sid_index_path:
        print(f"Warning: sid_index_path={sid_index_path} does not exist; semantic tokens will not be added.")

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if new_tokens:
            embedding_layer = model.get_input_embeddings()
            embedding_layer.weight.requires_grad = True

            def mask_grad(grad):
                # grad shape: [vocab_size, hidden_dim]
                grad[:original_vocab_size].zero_()
                return grad
            
            embedding_layer.weight.register_hook(mask_grad)

            print(
                f"Unfrozen {len(new_tokens)} new token embeddings "
                f"(indices {original_vocab_size} to {len(tokenizer) - 1})"
            )

        else:
            raise RuntimeError(
                "freeze_LLM=True but no new tokens were added. "
                "Provide a valid --sid_index_path (or add tokens manually), otherwise nothing is trainable."
            )

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    train_datasets = []
    train_datasets.append(
        SidSFTDataset(
            train_file=train_file,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category,
        )
    )

    # Optional auxiliary tasks (require item metadata + index mapping)
    if item_meta_path and os.path.exists(item_meta_path) and sid_index_path and os.path.exists(sid_index_path):
        train_datasets.append(
            SidItemFeatDataset(
                item_file=item_meta_path,
                index_file=sid_index_path,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category,
            )
        )
        train_datasets.append(
            FusionSeqRecDataset(
                train_file=train_file,
                item_file=item_meta_path,
                index_file=sid_index_path,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category,
            )
        )
    else:
        if not (item_meta_path and os.path.exists(item_meta_path)):
            print("Info: item_meta_path missing; skipping title/description auxiliary tasks.")
        if not (sid_index_path and os.path.exists(sid_index_path)):
            print("Info: sid_index_path missing; skipping auxiliary tasks that need index.json.")

    train_data = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    # Avoid materializing the entire dataset into a HuggingFace Dataset (can be very slow and memory heavy).
    # Let Trainer work with torch Dataset/ConcatDataset directly.
    steps_per_epoch = max(
        1,
        math.ceil(len(train_data) / (micro_batch_size * world_size * gradient_accumulation_steps)),
    )
    eval_step_ratio = 0.05
    eval_steps = max(1, int(eval_step_ratio * steps_per_epoch))
    print(f"steps_per_epoch={steps_per_epoch} eval_steps={eval_steps} world_size={world_size}")

    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
