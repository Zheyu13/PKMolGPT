import pandas as pd 
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="Name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='Debug mode')
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='Condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='Use LSTM for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="Name of the dataset to train on", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="Properties to be used for conditioning", required=False)
    parser.add_argument('--num_props', type=int, default=0,
                        help="Number of properties to use for condition; if 0, will use len(props)", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="Number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="Number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="Embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="Total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="Batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=6e-4, help="Learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="Number of LSTM layers", required=False)

    args = parser.parse_args()

    set_seed(42)
    wandb.init(project="lig_gpt", name=args.run_name)

    data = pd.read_csv(args.data_name + '.csv')
    print("Original columns:", data.columns)

    # Only drop rows where 'smiles' column is empty/NaN
    data = data.dropna(subset=['smiles']).reset_index(drop=True)
    print("Columns after dropna (smiles only):", data.columns)

    data.columns = data.columns.str.lower()

    if 'moses' in args.data_name:
        train_data = data[data['split'] == 'train'].reset_index(drop=True)  # Moses uses 'split'
    else:
        train_data = data[data['source'] == 'train'].reset_index(drop=True)

    if 'moses' in args.data_name:
        val_data = data[data['split'] == 'test'].reset_index(drop=True)
    else:
        val_data = data[data['source'] == 'val'].reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    if args.num_props == 0:
        num_props = len(args.props)
    else:
        num_props = args.num_props

    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()

    scaffold = train_data['scaffold_smiles']
    vscaffold = val_data['scaffold_smiles']
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print("Max SMILES length:", max_len)

    # Handle NaN values in scaffold columns
    scaffold_values = [str(i) if pd.notna(i) else "" for i in (list(scaffold.values) + list(vscaffold.values))]
    lens = [len(regex.findall(i.strip())) for i in scaffold_values]
    scaffold_maxlen = max(lens)
    print("Scaffold max len:", scaffold_maxlen)

    smiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]
    # Handle NaN values in scaffold processing
    scaffold = [str(i) if pd.notna(i) else "" for i in scaffold]
    vscaffold = [str(i) if pd.notna(i) else "" for i in vscaffold]
    scaffold = [i + str('<')*(scaffold_maxlen - len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<')*(scaffold_maxlen - len(regex.findall(i.strip()))) for i in vscaffold]

    # Use a fixed vocabulary that matches generation.
    whole_string = [
        '#', '%10', '%11', '%12', '%13', '%14', '%15', '%16', '%17', '%18', '%19', '(', ')', '-', '.', '/', 
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 
        'B', 'Br', 'C', 'Cl', 'I', 'N', 'O', 'P', 'S', 
        '[13C@H]', '[13CH2]', '[2H]', '[Al+3]', '[As]', '[B-]', '[B@-]', '[B@@-]', '[B]', '[Br-]', 
        '[C+]', '[C-]', '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH+]', '[CH-]', '[CH2]', '[CH]', '[C]', 
        '[Ca+2]', '[Ca]', '[Cl-]', '[Co-2]', '[Co]', '[Fe+2]', '[Fe+6]', '[Fe-2]', '[Ga+3]', '[Ho]', '[I-]', 
        '[K+]', '[Mg+2]', '[Mg-2]', '[N+]', '[N-]', '[N@+]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', 
        '[Na+]', '[Na]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[P@@]', '[P@]', '[PH]', '[S+]', '[S@+]', 
        '[SH+]', '[SH]', '[Se]', '[SiH2]', '[SiH]', '[Si]', '[Tc]', '[Zn+2]', '[Zn]', '[c-]', 
        '[n+]', '[n-]', '[nH+]', '[nH]', '[n]', '[o+]', '[s+]', '\\', 'c', 'n', 'o', 's'
    ]


    stoi = {token: idx for idx, token in enumerate(whole_string)}
    itos = {idx: token for token, idx in stoi.items()}
    vocab_save_path = f"data/{args.data_name}_stoi.json"
    os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
    with open(vocab_save_path, "w") as f:
        json.dump(stoi, f)
    print("Final vocabulary (itos):", itos)
    print("Final vocabulary size:", len(itos))

    fixed_block_size = 300
    # Enforce fixed block size by slicing the padded strings.
    smiles = [s[:fixed_block_size] for s in smiles]
    vsmiles = [s[:fixed_block_size] for s in vsmiles]
    if args.scaffold:
        scaffold = [s[:fixed_block_size] for s in scaffold]
        vscaffold = [s[:fixed_block_size] for s in vscaffold]

    train_dataset = SmileDataset(
        args=args,
        data=smiles,
        block_size=fixed_block_size,
        stoi=stoi,
        prop=prop,
        aug_prob=0,
        scaffold=scaffold,
        scaffold_max_len=scaffold_maxlen
    )
    valid_dataset = SmileDataset(
        args=args,
        data=vsmiles,
        block_size=fixed_block_size,
        stoi=stoi,
        prop=vprop,
        aug_prob=0,
        scaffold=vscaffold,
        scaffold_max_len=scaffold_maxlen
    )
    
    mconf = GPTConfig(
        train_dataset.vocab_size, 
        fixed_block_size, 
        num_props=num_props,
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd,
        scaffold=args.scaffold, 
        scaffold_maxlen=scaffold_maxlen,
        lstm=args.lstm, 
        lstm_layers=args.lstm_layers
    )
    model = GPT(mconf)

    tconf = TrainerConfig(
        max_epochs=args.max_epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        lr_decay=True, 
        warmup_tokens=0.1 * len(train_data) * max_len, 
        final_tokens=args.max_epochs * len(train_data) * max_len,
        num_workers=0,  # Use 0 workers to avoid DataLoader issues.
        ckpt_path=f'../cond_gpt/weights/{args.run_name}.pt', 
        block_size=train_dataset.max_len, 
        generate=False
    )

    print("Block size:", train_dataset.max_len)

    trainer = Trainer(model, train_dataset, valid_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train(wandb)

    # Fix: Check if trainer.train() returned a summary DataFrame.
    if df is not None:
        df.to_csv(f'{args.run_name}.csv', index=False)
        print("Training summary saved.")
    else:
        print("Trainer.train() returned None. Training complete, but no summary DataFrame was produced.")
