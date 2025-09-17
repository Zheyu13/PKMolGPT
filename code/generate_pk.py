from utils import check_novelty, sample, canonic_smiles, intdiv1
from dataset import SmileDataset
from rdkit.Chem import QED, Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import torch

# choose GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from moses.utils import get_mol
import re
import json
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# Example usage:
# python generate.py --model_weight guacamol_sas_logp.pt --props sas logp --data_name guacamol_2k --csv_name gua_sas_logp_temp1 --gen_size 10000 --batch_size 512

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, help="Path of model weights", required=True)
    parser.add_argument('--scaffold', action='store_true', default=False, help="Condition on scaffold")
    parser.add_argument('--lstm', action='store_true', default=False, help="Use LSTM for transforming scaffold")
    parser.add_argument('--csv_name', type=str, help="Name to save generated molecules in CSV", required=True)
    parser.add_argument('--data_name', type=str, default='moses2', help="Name of dataset (used for vocabulary file)", required=False)
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size", required=False)
    parser.add_argument('--gen_size', type=int, default=10000, help="Total molecules to generate", required=False)
    # These arguments are placeholders; actual values are inferred from checkpoint.
    parser.add_argument('--vocab_size', type=int, default=113, help="Vocabulary size (will be inferred)", required=False)
    parser.add_argument('--block_size', type=int, default=300, help="Block size (will be inferred)", required=False)
    parser.add_argument('--props', nargs="+", default=[], help="Properties for conditioning", required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="Number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="Number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="Embedding dimension", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0, help="Number of LSTM layers", required=False)
    args = parser.parse_args()

    # Starting context. (Feel free to modify.)
    context = "C"

    # Load the dataset CSV (used here to optionally compute properties, novelty, etc.)
    data = pd.read_csv(args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    if 'moses' in args.data_name:
        smiles = data[data['split'] != 'test_scaffolds']['smiles']
        scaf = data[data['split'] != 'test_scaffolds']['scaffold_smiles']
    else:
        smiles = data[data['source'] != 'test']['smiles']
        scaf = data[data['source'] != 'test']['scaffold_smiles']

    # Define and compile regex pattern.
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    print("Regex pattern:", pattern)

    # Load vocabulary mapping from file saved during training.
    vocab_file = f"data/{args.data_name}_stoi.json"
    if os.path.exists(vocab_file):
        with open(vocab_file, "r") as f:
            stoi = json.load(f)
        print("Loaded vocabulary from file.")
    else:
        # Fallback: Build vocabulary from CSV (warning: might not match training!)
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}\n"
                                f"Please make sure to run training first to generate it.")
    itos = {i: ch for ch, i in stoi.items()}
    print("Vocabulary (itos):", itos)
    print("Vocabulary size (from file):", len(itos))
    num_props = len(args.props)

    # Load the model checkpoint and infer hyperparameters.
    checkpoint = torch.load(args.model_weight, map_location=torch.device('cpu'))
    inferred_block_size = checkpoint["pos_emb"].shape[1]
    inferred_vocab_size = checkpoint["tok_emb.weight"].shape[0]
    print("Inferred block size:", inferred_block_size)
    print("Inferred vocabulary size from checkpoint:", inferred_vocab_size)
    if len(itos) != inferred_vocab_size:
        print(f"WARNING: Loaded vocabulary size ({len(itos)}) does not match checkpoint ({inferred_vocab_size}).")
        print("Ensure you use the same vocabulary file as during training.")
        
    if args.scaffold:
        scaf_raw = [
            'O=C1c2ccccc2C(=O)c2ccccc21',
            'O=C1OCC=C1CCCCCCCCCCCCC[C@H]1CCCO1',
            'O=c1cccco1',
            'O=c1ccoc2ccccc12',
            'c1ccc(O[C@H]2CCCCO2)cc1'
        ]
        if args.lstm:  
            TRAIN_SCAF_TOKENS = args.lstm_layers  
            scaf_condition = scaf_raw  
        else:  
            TRAIN_SCAF_TOKENS = 100
            scaf_condition = [
                s + "<" * (TRAIN_SCAF_TOKENS - len(regex.findall(s)))
                for s in scaf_raw
            ]
            
        scaffold_max_len = TRAIN_SCAF_TOKENS
            
        print(f"Detected scaffold_max_len = {scaffold_max_len}")
        print("Using scaffold conditions:")
        for sc in scaf_condition:
            print(sc)
    else:
        scaffold_max_len = 0
        scaf_condition = None

    # Create model configuration.
    mconf = GPTConfig(
        inferred_vocab_size,
        inferred_block_size,
        num_props=num_props,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        scaffold=args.scaffold,
        scaffold_maxlen=scaffold_max_len,
        lstm=args.lstm,
        lstm_layers=args.lstm_layers
    )
    model = GPT(mconf)

    # Remove attention mask keys from checkpoint before loading.
    ckpt = checkpoint.copy()
    for key in list(ckpt.keys()):
        if "attn.mask" in key:
            del ckpt[key]
    model.load_state_dict(ckpt, strict=False)
    #model.to("cuda")
    # pick the right device, falling back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded.")

    # For generation, we use the fixed block size that was used during training.
    fixed_block_size = 300
    # Compute number of iterations.
    gen_iter = math.ceil(args.gen_size / args.batch_size)
    print("Generation iterations:", gen_iter)

    # Set up property conditions.
    if "guacamol" in args.data_name:
        prop2value = {
            "qed": [0.3, 0.5, 0.7],
            "sas": [2.0, 3.0, 4.0],
            "logp": [2.0, 4.0, 6.0],
            "tpsa": [40.0, 80.0, 120.0],
            "tpsa_logp": [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0],
                          [40.0, 4.0], [80.0, 4.0], [120.0, 4.0],
                          [40.0, 6.0], [80.0, 6.0], [120.0, 6.0]],
            "sas_logp": [[2.0, 2.0], [2.0, 4.0], [2.0, 6.0],
                         [3.0, 2.0], [3.0, 4.0], [3.0, 6.0],
                         [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]],
            "tpsa_sas": [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0],
                         [40.0, 3.0], [80.0, 3.0], [120.0, 3.0],
                         [40.0, 4.0], [80.0, 4.0], [120.0, 4.0]],
            "tpsa_logp_sas": [[40.0, 2.0, 2.0], [40.0, 2.0, 4.0],
                              [40.0, 6.0, 4.0], [40.0, 6.0, 2.0],
                              [80.0, 6.0, 4.0], [80.0, 2.0, 4.0],
                              [80.0, 2.0, 2.0], [80.0, 6.0, 2.0]],
            "qed_sas_logp_tpsa": [[0.7, 2.0, 4.0, 40.0], [0.5, 3.0, 4.0, 80.0],
                                  [0.5, 2.0, 4.0, 80.0], [0.7, 2.0, 4.0, 80.0]]
        }
    else:
        prop2value = {
            'logp': [0.0, 4.0, 8.0], 'qed': [0.1, 0.4, 0.8], 'sas': [2.0, 4.0, 7.0], 'tpsa': [70.0, 60.0, 140.0],
            'sas_logp': [[4.0, 4.0], [4.0, 0.0], [2.0, 4.0], [7.0, 4.0], [7.0, 0.0], [4.0, 8.0], [7.0, 8.0], [2.0, 8.0], [2.0, 0.0]],
            'tpsa_logp': [[70.0, 4.0], [60.0, 4.0], [140.0, 4.0], [70.0, 0.0], [60.0, 0.0], [140.0, 0.0], [70.0, 8.0], [60.0, 8.0], [140.0, 8.0]],
            'tpsa_sas': [[70.0, 4.0], [70.0, 2.0], [60.0, 4.0], [60.0, 2.0], [140.0, 4.0], [140.0, 7.0], [70.0, 7.0], [140.0, 2.0], [60.0, 7.0]],
            'tpsa_logp_sas': [[70.0, 4.0, 2.0], [70.0, 4.0, 4.0], [60.0, 4.0, 2.0], [60.0, 4.0, 4.0], [70.0, 0.0, 4.0], [140.0, 4.0, 4.0], [60.0, 0.0, 4.0], [140.0, 4.0, 7.0]],
            'qed_sas_logp_tpsa': [[0.4, 4.0, 4.0, 70.0], [0.8, 2.0, 4.0, 60.0], [0.4, 7.0, 4.0, 140.0], [0.8, 4.0, 4.0, 70.0]]
        }

    prop_condition = None
    if len(args.props) > 0:
        key = "_".join(args.props)
        if key in prop2value:
            prop_condition = prop2value[key]
            print("Using property condition:", prop_condition)
        else:
            print("No matching property condition found for key:", key)

    # === Generation main logic ===
    all_mols = []

    # 1 Scaffold + Property
    group_counter = 1
    if args.scaffold and (prop_condition is not None):
        for sca in scaf_condition:
            scaffold_tokens = re.findall(pattern, sca)
            scaffold_indices = [stoi[s] for s in scaffold_tokens if s in stoi]
            scaffold_tensor = torch.tensor(scaffold_indices, dtype=torch.long)[None, :].to(device)
            scaffold_tensor = scaffold_tensor.repeat(args.batch_size, 1)

            # Create independent groups for each scaffold
            scaffold_molecules = []
            scaffold_stats = {
                'scaffold': sca.replace('<', ''),
                'total_molecules': 0,
                'valid_molecules': 0,
                'unique_molecules': 0,
                'novel_molecules': 0
            }

            for c in prop_condition:
                molecules = []
                print(f"Generating for scaffold: {sca[:20]}... and property: {c}")
                for iter_idx in tqdm(range(gen_iter), desc=f"S={sca[:10]}... | P={c}"):
                    x_tokens = re.findall(pattern, context)
                    x_indices = [stoi[s] for s in x_tokens if s in stoi]
                    x = torch.tensor(x_indices, dtype=torch.long)[None, :].repeat(args.batch_size, 1).to(device)
                    if x.shape[1] > fixed_block_size:
                        x = x[:, :fixed_block_size]

                    if len(args.props) == 1:
                        p = torch.tensor([[c]]).repeat(args.batch_size, 1).to(device)
                    else:
                        p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to(device)

                    y = sample(model, x, fixed_block_size, temperature=1.0, sample=True, top_k=50,
                               prop=p, scaffold=scaffold_tensor)
                    for gen_seq in y:
                        gen_seq = gen_seq[:fixed_block_size]
                        try:
                            gen_smiles = "".join([itos[int(i)] for i in gen_seq]).replace("<", "")
                        except KeyError:
                            continue
                        mol = get_mol(gen_smiles)
                        if mol is not None:
                            molecules.append(mol)

                print("Valid molecules % = {}".format(len(molecules)))
                print(f'Condition: {c}')
                print(f'Scaffold: {sca}')
                print('Valid ratio: ', np.round(len(molecules)/(args.batch_size*gen_iter), 3))
                
                # Calculate uniqueness and novelty for this property condition
                mol_dict = []
                for i in molecules:
                    mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i, canonical=True, isomericSmiles=True)})
                
                results = pd.DataFrame(mol_dict)
                canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                unique_smiles = list(set(canon_smiles))
                if 'moses' in args.data_name:
                    novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))
                else:
                    novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))
                
                print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio/100, 3))
                print('-' * 50)

                # Add molecules to scaffold group
                for m in molecules:
                    scaffold_molecules.append({
                        "molecule": m,
                        "smiles": Chem.MolToSmiles(m, isomericSmiles=True),
                        "scaffold": sca.replace('<', ''),
                        "property": str(c),
                        "mode": "scaffold+property",
                        "group": group_counter,
                        "intdiv1": intdiv1_val,
                        "qed": QED.qed(m),
                        "sas": sascorer.calculateScore(m),
                        "logp": Crippen.MolLogP(m),
                        "tpsa": CalcTPSA(m)
                    })

            # Calculate statistics for the entire scaffold
            scaffold_stats['total_molecules'] = len(scaffold_molecules)
            if scaffold_stats['total_molecules'] > 0:
                scaffold_df = pd.DataFrame(scaffold_molecules)
                canon_smiles_scaffold = [canonic_smiles(s) for s in scaffold_df['smiles']]
                unique_smiles_scaffold = list(set(canon_smiles_scaffold))
                scaffold_stats['valid_molecules'] = len(scaffold_molecules)
                scaffold_stats['unique_molecules'] = len(unique_smiles_scaffold)
                
                if 'moses' in args.data_name:
                    novel_ratio_scaffold = check_novelty(unique_smiles_scaffold, set(data[data['split']=='train']['smiles']))
                else:
                    novel_ratio_scaffold = check_novelty(unique_smiles_scaffold, set(data[data['source']=='train']['smiles']))
                scaffold_stats['novel_molecules'] = novel_ratio_scaffold

                print(f"\n=== Scaffold Group Statistics: {sca[:20]}... ===")
                print(f"Total molecules: {scaffold_stats['total_molecules']}")
                print(f"Unique molecules: {scaffold_stats['unique_molecules']}")
                print(f"Novelty ratio: {np.round(scaffold_stats['novel_molecules']/100, 3)}")
                
                # Calculate IntDiv1 for the entire scaffold group
                smiles_list_scaffold = [m["smiles"] for m in scaffold_molecules]
                scaffold_stats['intdiv1_val'] = intdiv1(smiles_list_scaffold, radius=2, nBits=2048, unique_valid_only=True, sample_max=20000, seed=42)
                print(f"IntDiv1: {np.round(scaffold_stats['intdiv1_val'], 6)}")
                print("=" * 60)

            # Add all molecules from scaffold group to total results
            all_mols.extend(scaffold_molecules)
            group_counter += 1

    # 2 Scaffold only
    elif args.scaffold and (prop_condition is None):
        for sca in scaf_condition:
            scaffold_tokens = re.findall(pattern, sca)
            scaffold_indices = [stoi[s] for s in scaffold_tokens if s in stoi]
            scaffold_tensor = torch.tensor(scaffold_indices, dtype=torch.long)[None, :].to(device)
            scaffold_tensor = scaffold_tensor.repeat(args.batch_size, 1)

            molecules = []
            print(f"Generating for scaffold only: {sca[:20]}...")
            for iter_idx in tqdm(range(gen_iter), desc=f"Scaffold-{sca[:15]}..."):
                x_tokens = re.findall(pattern, context)
                x_indices = [stoi[s] for s in x_tokens if s in stoi]
                x = torch.tensor(x_indices, dtype=torch.long)[None, :].repeat(args.batch_size, 1).to(device)
                if x.shape[1] > fixed_block_size:
                    x = x[:, :fixed_block_size]

                y = sample(model, x, fixed_block_size, temperature=1.0, sample=True, top_k=50,
                           prop=None, scaffold=scaffold_tensor)
                for gen_seq in y:
                    gen_seq = gen_seq[:fixed_block_size]
                    try:
                        gen_smiles = "".join([itos[int(i)] for i in gen_seq]).replace("<", "")
                    except KeyError:
                        continue
                    mol = get_mol(gen_smiles)
                    if mol is not None:
                        molecules.append(mol)

            print("Valid molecules % = {}".format(len(molecules)))
            print(f'Scaffold: {sca}')
            print('Valid ratio: ', np.round(len(molecules)/(args.batch_size*gen_iter), 3))
            
            # Calculate uniqueness and novelty for this scaffold
            mol_dict = []
            for i in molecules:
                mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i, canonical=True, isomericSmiles=True)})
            
            results = pd.DataFrame(mol_dict)
            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))
            if 'moses' in args.data_name:
                novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))
            else:
                novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))
            
            print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
            print('Novelty ratio: ', np.round(novel_ratio/100, 3))
            
            # Calculate IntDiv1 for this scaffold+property combination
            smiles_list = [Chem.MolToSmiles(m, isomericSmiles=True) for m in molecules]
            intdiv1_val = intdiv1(smiles_list, radius=2, nBits=2048, unique_valid_only=True, sample_max=20000, seed=42)
            print('IntDiv1: ', np.round(intdiv1_val, 6))
            
            # Calculate SD and MAD for this condition
            if len(args.props) > 0:
                # Create a temporary dataframe with calculated properties
                temp_mol_dict = []
                for m in molecules:
                    temp_mol_dict.append({
                        'molecule': m,
                        'qed': QED.qed(m),
                        'sas': sascorer.calculateScore(m),
                        'logp': Crippen.MolLogP(m),
                        'tpsa': CalcTPSA(m)
                    })
                temp_df = pd.DataFrame(temp_mol_dict)
                
                # Calculate differences from target
                if len(args.props) == 1:
                    target_val = c
                    prop_name = args.props[0]
                    if prop_name in temp_df.columns:
                        diff = (temp_df[prop_name] - target_val).abs()
                        mad_val = diff.mean()
                        sd_val = diff.std(ddof=0)
                        print(f'{prop_name}_MAD: {mad_val:.4f}')
                        print(f'{prop_name}_SD: {sd_val:.4f}')
                else:
                    # Multiple properties
                    for i, prop_name in enumerate(args.props):
                        if prop_name in temp_df.columns:
                            target_val = c[i] if isinstance(c, (list, tuple)) else c
                            diff = (temp_df[prop_name] - target_val).abs()
                            mad_val = diff.mean()
                            sd_val = diff.std(ddof=0)
                            print(f'{prop_name}_MAD: {mad_val:.4f}')
                            print(f'{prop_name}_SD: {sd_val:.4f}')
            
            print('-' * 50)

            # Create independent groups for each scaffold
            scaffold_molecules = []
            for m in molecules:
                                    scaffold_molecules.append({
                        "molecule": m,
                        "smiles": Chem.MolToSmiles(m, isomericSmiles=True),
                        "scaffold": sca.replace('<', ''),
                        "mode": "scaffold-only",
                        "group": group_counter,
                        "intdiv1": intdiv1_val,
                        "qed": QED.qed(m),
                        "sas": sascorer.calculateScore(m),
                        "logp": Crippen.MolLogP(m),
                        "tpsa": CalcTPSA(m)
                    })

            # Calculate statistics for the entire scaffold
            if len(scaffold_molecules) > 0:
                scaffold_df = pd.DataFrame(scaffold_molecules)
                canon_smiles_scaffold = [canonic_smiles(s) for s in scaffold_df['smiles']]
                unique_smiles_scaffold = list(set(canon_smiles_scaffold))
                
                if 'moses' in args.data_name:
                    novel_ratio_scaffold = check_novelty(unique_smiles_scaffold, set(data[data['split']=='train']['smiles']))
                else:
                    novel_ratio_scaffold = check_novelty(unique_smiles_scaffold, set(data[data['source']=='train']['smiles']))

                print(f"\n=== Scaffold Group Statistics: {sca[:20]}... ===")
                print(f"Total molecules: {len(scaffold_molecules)}")
                print(f"Unique molecules: {len(unique_smiles_scaffold)}")
                print(f"Novelty ratio: {np.round(novel_ratio_scaffold/100, 3)}")
                print("=" * 60)

            # Add all molecules from scaffold group to total results
            all_mols.extend(scaffold_molecules)
            group_counter += 1

    # 3 Property only
    elif (not args.scaffold) and (prop_condition is not None):
        for c in prop_condition:
            molecules = []
            print(f"Generating for property only: {c}")
            for iter_idx in tqdm(range(gen_iter), desc=f"Property-{c}"):
                x_tokens = re.findall(pattern, context)
                x_indices = [stoi[s] for s in x_tokens if s in stoi]
                x = torch.tensor(x_indices, dtype=torch.long)[None, :].repeat(args.batch_size, 1).to(device)
                if x.shape[1] > fixed_block_size:
                    x = x[:, :fixed_block_size]

                if len(args.props) == 1:
                    p = torch.tensor([[c]]).repeat(args.batch_size, 1).to(device)
                else:
                    p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to(device)

                y = sample(model, x, fixed_block_size, temperature=1.0, sample=True, top_k=50,
                           prop=p, scaffold=None)
                for gen_seq in y:
                    gen_seq = gen_seq[:fixed_block_size]
                    try:
                        gen_smiles = "".join([itos[int(i)] for i in gen_seq]).replace("<", "")
                    except KeyError:
                        continue
                    mol = get_mol(gen_smiles)
                    if mol is not None:
                        molecules.append(mol)

            print("Valid molecules % = {}".format(len(molecules)))
            print(f'Condition: {c}')
            print('Valid ratio: ', np.round(len(molecules)/(args.batch_size*gen_iter), 3))
            
            # Calculate uniqueness and novelty for this property condition
            mol_dict = []
            for i in molecules:
                mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i, canonical=True, isomericSmiles=True)})
            
            results = pd.DataFrame(mol_dict)
            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))
            if 'moses' in args.data_name:
                novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))
            else:
                novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))
            
            print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
            print('Novelty ratio: ', np.round(novel_ratio/100, 3))
            
            # Calculate IntDiv1 for this scaffold
            smiles_list = [Chem.MolToSmiles(m, isomericSmiles=True) for m in molecules]
            intdiv1_val = intdiv1(smiles_list, radius=2, nBits=2048, unique_valid_only=True, sample_max=20000, seed=42)
            print('IntDiv1: ', np.round(intdiv1_val, 6))
            
            # Calculate SD and MAD for this condition
            if len(args.props) > 0:
                # Create a temporary dataframe with calculated properties
                temp_mol_dict = []
                for m in molecules:
                    temp_mol_dict.append({
                        'molecule': m,
                        'qed': QED.qed(m),
                        'sas': sascorer.calculateScore(m),
                        'logp': Crippen.MolLogP(m),
                        'tpsa': CalcTPSA(m)
                    })
                temp_df = pd.DataFrame(temp_mol_dict)
                
                # Calculate differences from target
                if len(args.props) == 1:
                    target_val = c
                    prop_name = args.props[0]
                    if prop_name in temp_df.columns:
                        diff = (temp_df[prop_name] - target_val).abs()
                        mad_val = diff.mean()
                        sd_val = diff.std(ddof=0)
                        print(f'{prop_name}_MAD: {mad_val:.4f}')
                        print(f'{prop_name}_SD: {sd_val:.4f}')
                else:
                    # Multiple properties
                    for i, prop_name in enumerate(args.props):
                        if prop_name in temp_df.columns:
                            target_val = c[i] if isinstance(c, (list, tuple)) else c
                            diff = (temp_df[prop_name] - target_val).abs()
                            mad_val = diff.mean()
                            sd_val = diff.std(ddof=0)
                            print(f'{prop_name}_MAD: {mad_val:.4f}')
                            print(f'{prop_name}_SD: {sd_val:.4f}')
            
            print('-' * 50)

            # Create independent groups for each property condition
            property_molecules = []
            for m in molecules:
                property_molecules.append({
                    "molecule": m,
                    "smiles": Chem.MolToSmiles(m, isomericSmiles=True),
                    "property": str(c),
                    "mode": "property-only",
                    "group": group_counter,
                    "intdiv1": intdiv1_val,
                    "qed": QED.qed(m),
                    "sas": sascorer.calculateScore(m),
                    "logp": Crippen.MolLogP(m),
                    "tpsa": CalcTPSA(m)
                })

            # Add all molecules from property group to total results
            all_mols.extend(property_molecules)
            group_counter += 1

    # 4 Unconditional
    else:
        print("Generating without any condition (unconditional)...")
        molecules = []
        for iter_idx in tqdm(range(gen_iter), desc="Unconditional"):
            x_tokens = re.findall(pattern, context)
            x_indices = [stoi[s] for s in x_tokens if s in stoi]
            x = torch.tensor(x_indices, dtype=torch.long)[None, :].repeat(args.batch_size, 1).to(device)
            if x.shape[1] > fixed_block_size:
                x = x[:, :fixed_block_size]

            y = sample(model, x, fixed_block_size, temperature=1.0, sample=True, top_k=50,
                       prop=None, scaffold=None)
            for gen_seq in y:
                gen_seq = gen_seq[:fixed_block_size]
                try:
                    gen_smiles = "".join([itos[int(i)] for i in gen_seq]).replace("<", "")
                except KeyError:
                    continue
                mol = get_mol(gen_smiles)
                if mol is not None:
                    molecules.append(mol)

        print("Valid molecules % = {}".format(len(molecules)))
        print('Valid ratio: ', np.round(len(molecules)/(args.batch_size*gen_iter), 3))
        
        # Calculate uniqueness and novelty for unconditional generation
        mol_dict = []
        for i in molecules:
            mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i, canonical=True, isomericSmiles=True)})
        
        results = pd.DataFrame(mol_dict)
        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        if 'moses' in args.data_name:
            novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))
        else:
            novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))
        
        print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio/100, 3))
        
        # Calculate IntDiv1 for unconditional generation
        smiles_list = [Chem.MolToSmiles(m, isomericSmiles=True) for m in molecules]
        intdiv1_val = intdiv1(smiles_list, radius=2, nBits=2048, unique_valid_only=True, sample_max=20000, seed=42)
        print('IntDiv1: ', np.round(intdiv1_val, 6))
        
        print('-' * 50)

        # Create independent groups for unconditional generation
        unconditional_molecules = []
        for m in molecules:
            unconditional_molecules.append({
                "molecule": m,
                "smiles": Chem.MolToSmiles(m, isomericSmiles=True),
                "mode": "unconditional",
                "group": group_counter,
                "intdiv1": intdiv1_val,
                "qed": QED.qed(m),
                "sas": sascorer.calculateScore(m),
                "logp": Crippen.MolLogP(m),
                "tpsa": CalcTPSA(m)
            })

        # Add all molecules from unconditional group to total results
        all_mols.extend(unconditional_molecules)

if all_mols:
    results = pd.DataFrame(all_mols)

    canon_smiles = [canonic_smiles(s) for s in results["smiles"]]
    unique_smiles = list(set(canon_smiles))
    if "moses" in args.data_name:
        novel_ratio = check_novelty(unique_smiles,
                                    set(data[data["split"] == "train"]["smiles"]))
    else:
        novel_ratio = check_novelty(unique_smiles,
                                    set(data[data["source"] == "train"]["smiles"]))

    # Add statistics to each molecule record
    validity_ratio = np.round(len(results) / (args.batch_size * gen_iter), 3)
    unique_ratio = np.round(len(unique_smiles) / len(results), 3)
    novelty_ratio = np.round(novel_ratio / 100, 3)
    
    results['validity_ratio'] = validity_ratio
    results['unique_ratio'] = unique_ratio
    results['novelty_ratio'] = novelty_ratio

    # Calculate IntDiv1 for the entire dataset
    # smiles_list_total = [m["smiles"] for m in all_mols]
    # intdiv1_val_total = intdiv1(smiles_list_total, radius=2, nBits=2048, unique_valid_only=True, sample_max=20000, seed=42)
    
    # Calculate SD and MAD for property conditions
    if prop_condition is not None and len(args.props) > 0:
        print("\n" + "="*60)
        print("PROPERTY CONDITION ANALYSIS")
        print("="*60)
        
        # Create condition grid for analysis
        props_to_calc = args.props
        cond_grid = prop_condition
        cond_df = pd.DataFrame(cond_grid, columns=props_to_calc, index=range(1, len(cond_grid)+1))
        
        # Add group information to results based on property conditions
        results['group'] = 0
        for idx, condition in enumerate(cond_grid, 1):
            if len(args.props) == 1:
                # Single property condition
                mask = results['property'] == str(condition)
            else:
                # Multiple property conditions
                mask = results['property'] == str(condition)
            results.loc[mask, 'group'] = idx
        
        # Add canonical SMILES for analysis
        results['canon'] = [canonic_smiles(s) for s in results['smiles']]
        
        # Get training set for novelty calculation
        if 'moses' in args.data_name:
            train_set = set(data[data['split']=='train']['smiles'])
        else:
            train_set = set(data[data['source']=='train']['smiles'])
        
        # Calculate and print analysis for each group
        for g in cond_df.index:
            gdf = results[results['group'] == g]
            if gdf.empty:
                continue

            uniq = set(gdf['canon'])
            uniqueness = len(uniq) / len(gdf)
            novelty = len({s for s in uniq if s not in train_set}) / len(uniq)

            # Calculate IntDiv1 for this group
            smiles_list = gdf['smiles'].tolist()
            intdiv1_val = intdiv1(smiles_list, radius=2, nBits=2048, unique_valid_only=True, sample_max=20000, seed=42)

            print(f"\nGroup {g}:")
            print(f"  Size: {len(gdf)}")
            print(f"  Uniqueness: {round(uniqueness, 4)}")
            print(f"  Novelty: {round(novelty, 4)}")
            print(f"  IntDiv1: {round(intdiv1_val, 6)}")

            targets = cond_df.loc[g]
            for p in props_to_calc:
                if p in gdf.columns:
                    diff = (gdf[p] - targets[p]).abs()
                    mad_val = round(diff.mean(), 4)
                    sd_val = round(diff.std(ddof=0), 4)
                    print(f"  {p}_MAD: {mad_val}")
                    print(f"  {p}_SD: {sd_val}")
                    
                    # Add MAD and SD to the main results for this group
                    mask = results['group'] == g
                    results.loc[mask, f'{p}_MAD'] = mad_val
                    results.loc[mask, f'{p}_SD'] = sd_val
            
            # Add IntDiv1 to the main results for this group
            mask = results['group'] == g
            results.loc[mask, 'intdiv1'] = intdiv1_val

    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print("Valid ratio :", validity_ratio)
    print("Unique ratio:", unique_ratio)
    print("Novelty ratio:", novelty_ratio)
    # print("IntDiv1:", np.round(intdiv1_val_total, 6))
    print('=' * 60)
    results.to_csv(args.csv_name + ".csv", index=False)
    print(f"Saved {len(results)} molecules to {args.csv_name}.csv")
else:
    print("No molecules generated.")

