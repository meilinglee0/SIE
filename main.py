import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from dataloader.dataloader import MentalDataModule 
from model.sie import SIEClassifier  
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model_type = SIEClassifier  

def main(args):
    checkpoint_callback = ModelCheckpoint(save_top_k=1, save_weights_only=True, monitor='val_macro_f1', 
        mode='max', dirpath='./checkpoints', filename='checkpoint-{epoch:02d}-{val_macro_f1:.2f}')
    model = model_type(**vars(args))
    if "bertweet" in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    use_symp = True
    data_module = MentalDataModule(
        bs=args.bs,
        input_dir=args.input_dir,
        tokenizer=tokenizer,
        max_len=args.max_len,
        disease=args.disease,
        setting=args.setting,
        use_symp=use_symp,
        bal_sample=args.bsamp,
        control_ratio=args.control_ratio
    )
    early_stop_callback = EarlyStopping(
        monitor='val_macro_f1',
        patience=args.patience,
        mode="max"
    )
    trainer = pl.Trainer(
        accelerator="gpu",  
        devices=1,  
        callbacks=[early_stop_callback, checkpoint_callback], 
        val_check_interval=1.0, 
        max_epochs=100, 
        min_epochs=1, 
        accumulate_grad_batches=args.accumulation, 
        gradient_clip_val=args.gradient_clip_val, 
        deterministic=True, 
        log_every_n_steps=100
    )

    if args.find_lr:
       
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)

      
        new_lr = lr_finder.suggestion()
      
    else:
        trainer.fit(model, data_module)
        results = trainer.test(model=model, datamodule=data_module, ckpt_path='best')
        print(results)
        result = results[0]
        result['ckpt_path'] = checkpoint_callback.best_model_path
        result['hyper_params'] = str(args)
        
        os.makedirs('results', exist_ok=True)
        
        try:
            disease_name = args.disease if args.disease else 'unknown'
            result_file = f'results/{disease_name}_res_{time.time()}.json'
            
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print(f"Error saving results: {e}")

    return checkpoint_callback.best_model_path

if __name__ == "__main__":
    seed_everything(2000, workers=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert-base-uncased",
                        help="Hugging Face model name")
    parser.add_argument("--hier", type=int, default=1)
    parser.add_argument("--disease", type=str, default='None')
    parser.add_argument("--setting", type=str, default="multilabel", 
                       help="Classification setting: binary or multilabel")
    parser.add_argument("--bsamp", type=bool, default=False)
    parser.add_argument("--control_ratio", type=float, default=0.7)
    temp_args, _ = parser.parse_known_args()
    model_type = SIEClassifier
    parser = model_type.add_model_specific_args(parser)
    parser.add_argument("--bs", type=int, default=8, help="Batch size")
    parser.add_argument("--input_dir", type=str, default="./processed/symp_sum_top")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--find_lr", action="store_true")
    
    # 添加新的参数
    parser.add_argument("--contrastive_weight", type=float, default=0.1, 
                        help="Weight for contrastive learning loss")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature parameter for contrastive learning")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="Number of classes including control")
    
    args = parser.parse_args()
    best_model_path = main(args)
    print('best_model:', best_model_path)