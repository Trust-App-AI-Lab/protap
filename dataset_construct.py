import os
import json
import pickle
from datasets import Dataset, load_from_disk

from data.dataset import EgnnDataset, ProtacDataset
from data.tokenizers import ProteinTokenizer, ProtacTokenizer
from data.drug_graph import sdf_to_graphs

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

def generate_pretrain_dataset():
    raw_data = '/mnt/data/protein_data/swiss_540k_family_index_list.json'

    with open(raw_data) as json_file:
        data = json.load(json_file)
        
    seq = []
    coords = []
    family = []
    for i in range(len(data)):
        seq.append(data[i]['seq'])
        coords.append(data[i]['coords'])
        family.append(data[i]['Pfam'])
        
    dataset = {
        'seq' : seq,
        'coords' : coords,
        'family' : family
    }

    dataset = Dataset.from_dict(dataset)
    dataset = dataset.save_to_disk('protein_family_0')
    dataset = Dataset.load_from_disk('protein_family_0')
    
    print(dataset['seq'][0])
    # dataset = load_dataset('swiss-50k-hf', streaming=True, num_proc=8)

    # Tokenize the amino acid sequence.
    tokenizer = ProteinTokenizer(max_seq_length=768, dataset=dataset, padding_to_longest=False)
    dataset = EgnnDataset(tokenizer=tokenizer, generate=True)

    dataset = load_from_disk('protein_family_1')
    input_ids, coords, masks = [], [], []
    families = []
    for protein in tqdm(dataset):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
        families.append(torch.tensor(protein['family']))

    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    families = torch.stack(families)
    dataset = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks,
        "family" : families
    }

    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type='torch', columns=['input_ids', 'coords', 'masks', 'family'])
    dataset = dataset.save_to_disk("protein_family_2")
    
def generate_pli_dataset():
    raw_data = '/mnt/data/protein_data/pli_data/davis_drug_pdb_data.txt'
    data = pd.read_csv(raw_data, sep='\t')
    
    dataset = Dataset.from_pandas(data)
    # print(dataset.column_names)
    dataset = dataset.remove_columns('Unnamed: 0')
    print(dataset[0:2])
    
    with open('/mnt/data/protein_data/pli_data/pli_structure.json', 'r') as json_file:
        proteins = json.load(json_file)
    
    coord = []
    seq = []
    drug = []
    y = []
    for x in tqdm(dataset):
       protein = x['protein_pdb'] # Query the protein coords and seq with the PDB id.
       protein_data = proteins[protein]
       seq.append(protein_data['seq'])
       coords = protein_data['coord']
       coord.append(coords)
       drug.append(x['drug']) # Store the drug id for joint training.
       y.append(x['y']) # Store the target value.
       
    dataset = {
        'seq' : seq,
        'coords' : coord,
        'drug' : drug,
        'y' : y
    }
    dataset = Dataset.from_dict(dataset)
    # dataset = dataset.add_column(column=seq, name='seq')
    # dataset = dataset.add_column(column=coord, name='coord')
    print(dataset[0]) # {'seq', 'coords', 'drug'}
    print(len(dataset))
    
    # Pocket length: [67, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85].
    
    # Tokenize the amino acid sequence.
    print("Tokenizing the data...")
    # Padding to the max length: 85.
    tokenizer = ProteinTokenizer(max_seq_length=85, dataset=dataset, padding_to_longest=False)
    dataset = EgnnDataset(tokenizer=tokenizer, generate=True, include_drug=True)
    
    dataset = load_from_disk('protein_drug_1')
    print(dataset[0])

    input_ids, coords, masks, y = [], [], [], []
    drugs = []
    for protein in tqdm(dataset):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
        drugs.append(torch.tensor(protein['drug']))
        y.append(torch.tensor(protein['y']))

    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    drugs = torch.stack(drugs)
    y = torch.stack(y)
    dataset = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks,
        "drugs" : drugs,
        "y" : y
    }

    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type='torch', columns=['input_ids', 'coords', 'masks', 'drugs', 'y'])
    dataset = dataset.save_to_disk("protein_drug_2")
    
    return dataset

def generate_protac_dataset():
    raw_data = './data/protac_data/protac_clean_structure_label.txt'
    data = pd.read_csv(raw_data, sep='\t')
    
    dataset = Dataset.from_pandas(data)
    # print(dataset.column_names)
    dataset = dataset.remove_columns('Unnamed: 0')
    print(dataset[0:2])
    
    # load the protein structure.
    with open('./data/protac_data/protac_poi_e3ligase_structure.json', 'r') as json_file:
        proteins = json.load(json_file)
    
    poi_seq, poi_coord = [], []
    e3_ligase_seq, e3_ligase_coord = [], []
    # warhead, linker, e3_ligand
    warhead, linker, e3_ligand = [], [], []
    label = []
    max_length = 0
    # Iterate the dataset.
    for x in tqdm(dataset):
       # poi, e3_ligase
        poi_id, e3_ligase_id = x['poi'], x['e3_ligase_structure'] # Obtein the ids.
        poi, e3_ligase = proteins[poi_id], proteins[e3_ligase_id] # Obtain the structures.
        poi_seq.append(poi['seq'])
        poi_coord.append(poi['coords'])
        e3_ligase_seq.append(e3_ligase['seq'])
        e3_ligase_coord.append(e3_ligase['coords'])
        
        # max_length = max(max_length, len(poi['seq']))
        # max_length = max(max_length, len(e3_ligase['seq']))
        
        warhead.append(x['warhead_sdf'])
        linker.append(x['linker_sdf'])
        e3_ligand.append(x['e3_ligand_sdf'])

        label.append(x['protac_label']) # Store the target label.
       
    dataset = {
        'poi_seq' : poi_seq,
        "poi_coord" : poi_coord,
        "e3_ligase_seq" : e3_ligase_seq,
        "e3_ligase_coord" : e3_ligase_coord,
        "warhead" : warhead,
        "linker" : linker,
        "e3_ligand" : e3_ligand,
        'label' : label
    }
    dataset = Dataset.from_dict(dataset)
    print(dataset[0]) # {'seq', 'coords', 'drug'}
    print(len(dataset))
    print(max_length) # 2527
    
    # # Tokenize the amino acid sequence.
    print("Tokenizing the data...")
    # Padding to the max length: 85.
    tokenizer = ProtacTokenizer(max_seq_length=768, dataset=dataset, padding_to_longest=False)
    dataset = ProtacDataset(tokenizer=tokenizer, generate=True)
    
    dataset = load_from_disk('protac_1')
    print(dataset[0])

    poi_input_ids, poi_coords, poi_masks = [], [], []
    e3_ligase_input_ids, e3_ligase_coords, e3_ligase_masks = [], [], []
    warhead, linker, e3_ligand = [], [], []
    label = []
    
    def extract(path):
        underscore_idx = path.rfind('_')
        dot_idx = path.rfind('.')
        return int(path[underscore_idx + 1 : dot_idx])
    
    for protein in tqdm(dataset):
        poi_input_ids.append(torch.tensor(protein['poi_input_ids']))
        poi_coords.append(torch.tensor(protein['poi_coords']))
        poi_masks.append(torch.tensor(protein['poi_masks']).bool())
        e3_ligase_input_ids.append(torch.tensor(protein['e3_ligase_input_ids']))
        e3_ligase_coords.append(torch.tensor(protein['e3_ligase_coords']))
        e3_ligase_masks.append(torch.tensor(protein['e3_ligase_masks']).bool())
        warhead.append(torch.tensor(extract(protein['warhead'])))
        linker.append(torch.tensor(extract(protein['linker'])))
        e3_ligand.append(torch.tensor(extract(protein['e3_ligand'])))
        label.append(torch.tensor(protein['label']))

    poi_input_ids = torch.stack(poi_input_ids)
    poi_coords = torch.stack(poi_coords)
    poi_masks = torch.stack(poi_masks)
    e3_ligase_input_ids = torch.stack(e3_ligase_input_ids)
    e3_ligase_coords = torch.stack(e3_ligase_coords)
    e3_ligase_masks = torch.stack(e3_ligase_masks)
    warhead = torch.stack(warhead)
    linker = torch.stack(linker)
    e3_ligand = torch.stack(e3_ligand)
    label = torch.stack(label)
    
    dataset = {
        "poi_input_ids": poi_input_ids,
        "poi_coords": poi_coords,
        "poi_masks": poi_masks,
        "e3_ligase_input_ids": e3_ligase_input_ids,
        "e3_ligase_coords": e3_ligase_coords,
        "e3_ligase_masks": e3_ligase_masks,
        "warhead": warhead,
        "linker": linker,
        "e3_ligand": e3_ligand,
        "label": label,
    }

    dataset = Dataset.from_dict(dataset)
    
    dataset.set_format(type='torch', columns=[
        'poi_input_ids', 
        'poi_coords', 
        'poi_masks', 
        'e3_ligase_input_ids', 
        'e3_ligase_coords', 
        'e3_ligase_masks', 
        'warhead', 
        'linker', 
        'e3_ligand', 
        'label'
    ])
    
    dataset = dataset.save_to_disk("protac_2")
    
    return dataset

def generate_function_prediction_data(go_term="biological_process", train_dir=None, test_dir=None):
    
    test_simil = pd.read_csv("GO_data_processed_0502/nrPDB-GO_test.csv", sep=',')
    df_test50 = test_simil[test_simil["<50%"] == 1]
    test50 = df_test50['PDB-chain'].tolist()

    with open("GO_data_processed_0502/data_splits.json", "r") as fs:
        data_split = json.load(fs)
        
    train_protein = data_split["train"] # IDs
    test_protein = data_split["test"]

    with open("GO_data_processed_0502/processed_data.json", "r") as f:
        protein = json.load(f)

    train_list = []
    test_list = []
    for i in protein:
        terms = np.array(i[go_term])
        if i['name'] in train_protein and not np.all(terms == 0) and "X" not in i['seq'] and len(i['seq'])==len(i['coords']):
            train_list.append(i)
    for j in protein:
        terms = np.array(j[go_term])
        if j['name'] in test_protein and not np.all(terms == 0) and j['name'] in test50 and "X" not in j['seq'] and len(j['seq'])==len(j['coords']):
            test_list.append(j)
            
    seq = []
    coords = []
    go = []
    for data in train_list:
        seq.append(data['seq'])
        coords.append(data['coords'])
        go.append(data[go_term])
    
    train_set = {
        'seq' : seq,
        'coords' : coords,
        'go' : go
    }
    train_set = Dataset.from_dict(train_set)
    train_set = train_set.save_to_disk(go_term + "_train_0")
    
    seq = []
    coords = []
    go = []
    for data in test_list:
        seq.append(data['seq'])
        coords.append(data['coords'])
        go.append(data[go_term])
    
    test_set = {
        'seq' : seq,
        'coords' : coords,
        'go' : go
    }

    test_set = Dataset.from_dict(test_set)
    test_set = test_set.save_to_disk(go_term + '_test_0')
    
    print("Tokenizing the data...")
    train_set = load_from_disk('./data/go_data/' + go_term + "_train_0")
    test_set = load_from_disk('./data/go_data/' + go_term + "_test_0")
    # Padding to the max length: 85.
    tokenizer = ProteinTokenizer(max_seq_length=768, dataset=train_set, padding_to_longest=False)
    dataset = EgnnDataset(tokenizer=tokenizer, generate=True, include_go=True, save_dir='./data/go_data/' + go_term + "_train_1")
    tokenizer = ProteinTokenizer(max_seq_length=768, dataset=test_set, padding_to_longest=False)
    dataset = EgnnDataset(tokenizer=tokenizer, generate=True, include_go=True, save_dir='./data/go_data/' + go_term + "_test_1")
    
    # dataset = load_from_disk('protein_drug_1')
    # print(dataset[0])
    
    train_set = load_from_disk('./data/go_data/' + go_term + "_train_1")
    test_set = load_from_disk('./data/go_data/' + go_term + "_test_1")

    input_ids, coords, masks, go = [], [], [], []
    for protein in tqdm(train_set):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
        go.append(torch.tensor(protein['go']))

    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    go = torch.stack(go)
    train_set = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks,
        "go" : go
    }

    train_set = Dataset.from_dict(train_set)
    train_set.set_format(type='torch', columns=['input_ids', 'coords', 'masks', 'go'])
    train_set = train_set.save_to_disk(go_term + "_train_2")
    
    input_ids, coords, masks, go = [], [], [], []
    for protein in tqdm(test_set):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
        go.append(torch.tensor(protein['go']))

    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    go = torch.stack(go)
    test_set = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks,
        "go" : go
    }

    test_set = Dataset.from_dict(test_set)
    test_set.set_format(type='torch', columns=['input_ids', 'coords', 'masks', 'go'])
    test_set = test_set.save_to_disk(go_term + "_test_2")
            
    return train_set, test_set

def get_cleavage_data(dataset_name):
    
    train_raw = "./data/cleavage_data/train_C14005.pkl"
    test_raw = "./data/cleavage_data/test_C14005.pkl"

    with open('./data/cleavage_data/Substrate.pickle', 'rb') as fsub:
        sub_protein = pickle.load(fsub)

    data_list = []
    for protein_id, protein_data in sub_protein.items():
        protein_data['name'] = protein_id
        data_list.append(protein_data)

    with open(train_raw, "rb") as ftr:
        train_raw_data = pickle.load(ftr)
    with open(test_raw, "rb") as fte:
        test_raw_data = pickle.load(fte)

    train_label_dict = {k.split("_")[0]: v for k, v in train_raw_data.items()}
    test_label_dict  = {k.split("_")[0]: v for k, v in test_raw_data.items()}

    train_list, test_list = [], []
    for p in data_list:
        pid = p['name']
        if pid in train_label_dict and len(p["seq"]) >=np.max(train_label_dict[pid]):
            p['cleave_site'] = train_label_dict[pid]
            train_list.append(p)
        elif pid in test_label_dict and len(p["seq"]) >=np.max(test_label_dict[pid]):
            p['cleave_site'] = test_label_dict[pid]
            test_list.append(p)
    
    seq = []
    coords = []
    site = []
    for data in train_list:
        seq.append(data['seq'])
        coords.append(data['coords'])
        site.append(data['cleave_site'])
    
    train_set = {
        'seq' : seq,
        'coords' : coords,
        'site' : site
    }
    train_set = Dataset.from_dict(train_set)
    train_set = train_set.save_to_disk(dataset_name + "_train_0")
    
    seq = []
    coords = []
    site = []
    for data in test_list:
        seq.append(data['seq'])
        coords.append(data['coords'])
        site.append(data['cleave_site'])
    
    test_set = {
        'seq' : seq,
        'coords' : coords,
        'site' : site
    }

    test_set = Dataset.from_dict(test_set)
    test_set = test_set.save_to_disk(dataset_name + '_test_0')
    
    train_set = load_from_disk(dataset_name + '_train_0') # Max number of sites: 18 & 56
    test_set = load_from_disk(dataset_name + '_test_0') # Max site position: 1356.
    
    l = 0
    k = 0
    for x in train_set['site']:
        l = max(l, len(x))
        for y in x:
            k = max(k, y)
    print(l)
    print(k)
    for x in test_set['site']:
        l = max(l, len(x))
        for y in x:
            k = max(k, y)
    print(l)
    print(k)
    print(train_set['site'])
    
    # Construct the multi labels.
    labels = train_set['site']
    multi_hot = torch.zeros(len(labels), 1357, dtype=torch.float)

    for i, label in enumerate(labels):
        multi_hot[i, label] = 1.0
    
    train_set = train_set.add_column('labels', multi_hot.tolist())
    
    labels = test_set['site']
    multi_hot = torch.zeros(len(labels), 1357, dtype=torch.float)

    for i, label in enumerate(labels):
        multi_hot[i, label] = 1.0
    
    test_set = test_set.add_column('labels', multi_hot.tolist())
    
    train_set = train_set.save_to_disk(dataset_name + '_train_1')
    test_set = test_set.save_to_disk(dataset_name + '_test_1')
    
    train_set = load_from_disk(dataset_name + '_train_1')
    test_set = load_from_disk(dataset_name + '_test_1')
    
    # print(train_set[100]['labels'])
    
    print("Tokenizing the data...")

    # Padding to the max length: 768.
    tokenizer = ProteinTokenizer(max_seq_length=768, dataset=train_set, padding_to_longest=False)
    dataset = EgnnDataset(tokenizer=tokenizer, generate=True, include_site=True, save_dir='./data/cleavage_data/' + dataset_name + "_train_2")
    tokenizer = ProteinTokenizer(max_seq_length=768, dataset=test_set, padding_to_longest=False)
    dataset = EgnnDataset(tokenizer=tokenizer, generate=True, include_site=True, save_dir='./data/cleavage_data/' + dataset_name + "_test_2")
    
    train_set = load_from_disk('./data/cleavage_data/' + dataset_name + "_train_2")
    test_set = load_from_disk('./data/cleavage_data/' + dataset_name + "_test_2")
    
    print(train_set[0])

    input_ids, coords, masks, site = [], [], [], []
    for protein in tqdm(train_set):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
        site.append(torch.tensor(protein['site']))

    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    site = torch.stack(site)
    train_set = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks,
        "site" : site
    }

    train_set = Dataset.from_dict(train_set)
    train_set.set_format(type='torch', columns=['input_ids', 'coords', 'masks', 'site'])
    train_set = train_set.save_to_disk(dataset_name + "_train_3")
    
    input_ids, coords, masks, site = [], [], [], []
    for protein in tqdm(test_set):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
        site.append(torch.tensor(protein['site']))

    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    site = torch.stack(site)
    test_set = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks,
        "site" : site
    }

    test_set = Dataset.from_dict(test_set)
    test_set.set_format(type='torch', columns=['input_ids', 'coords', 'masks', 'site'])
    test_set = test_set.save_to_disk(dataset_name + "_test_3")
            
    return train_set, test_set

def get_drug_graph():
    
    dataset = load_from_disk("protein_drug_1") # {input_ids, coords, masks, drugs}
    # print(dataset[0])
    
    drug_dir = '/mnt/data/protein_data/pli_data/davis_mol3d_sdf'
    drug = dataset['drug']
    drug_ids = list(set(drug))
    
    graph_list = {}
    for drug_id in drug_ids:
        data_dir = os.path.join(drug_dir, str(drug_id) + '.sdf')
        graph_list[drug_id] = data_dir
    
    drug_graphs = sdf_to_graphs(graph_list)
    
    return drug_graphs

if __name__ == '__main__':
    
    print("Generating......")
    
    # generate_pretrain_dataset()
    
    # dataset = load_from_disk("protein_drug_2")
    # print(len(dataset))
    # print(dataset[1000])
    
    # data = load_from_disk('./data/protein_family_2')
    # print(data['input_ids'][155])
    
    # input_ids = data['input_ids'][155]
    # seq = """
    # MVFPIRILVLFALLAFPACVHGAIRKYTFNVVTKQVTRICSTKQIVTVNGKFPGPTIYANEDDTILVNVVNNVKYNVSIHWHGIRQLRTGWADGPAYITQCPIKPGHSYVYNFTVTGQRGTLWWHAHVLWLRATVHGAIVILPKLGLPYPFPKPHREEVIILGEWWKSDTETVVNEALKSGLAPNVSDAHVINGHPGFVPNCPSQGNFKLAVESGKTYMLRLINAALNEELFFKIAGHRFTVVEVDAVYVKPFNTDTILIAPGQTTTALVSAARPSGQYLIAAAPFQDSAVVAVDNRTATATVHYSGTLSATPTKTTSPPPQNATSVANTFVNSLRSLNSKTYPANVPITVDHDLLFTVGLGINRCHSCKAGNFSRVVAAINNITFKMPKTALLQAHYFNLTGIYTTDFPAKPRRVFDFTGKPPSNLATMKATKLYKLPYNSTVQVVLQDTGNVAPENHPIHLHGFNFFVVGLGTGNYNSKKDSNKFNLVDPVERNTVGVPSGGWAAIRFRADNPGVWFMHCHLEVHTTWGLKMAFLVENGKGPNQSIRPPPSDLPKC
    # """
    
    # dataset = get_drug_graph()
    # dataset = load_from_disk('protein_drug_2')
    # print(dataset[0:3])
    
    
    # generate_protac_dataset()
    # dataset = load_from_disk('./data/protac_1')
    # print(dataset[0])
    # print(dataset[0]['warhead'])
    # print(dataset[100]['warhead'])
    # print(dataset[1000]['warhead'])
    
    # generate_function_prediction_data(go_term='cellular_component') # 320
    # generate_function_prediction_data(go_term='biological_process') # 1943
    # generate_function_prediction_data(go_term='molecular_function') # 489
    # data = load_from_disk('./data/go_data/cellular_component_train_2')
    # dataset = load_from_disk('./data/go_data/molecular_function_test_2')
    # dataset = load_from_disk('./data/go_data/biological_process_test_2')
    # count_all_zero = 0
    
    # print(len(dataset))

    # for data in dataset:
    #     go = data['go']  # Tensor or list
    #     # go_tensor = torch.tensor(go) if not isinstance(go, torch.Tensor) else go
    #     if torch.sum(go) < 1:
    #         count_all_zero += 1
    #         print(data)

    
    # dataset = load_from_disk('./data/swiss-protein-540k-tensor')
    # print(len(dataset))
    
    # get_cleavage_data('m10003')
    # get_cleavage_data('c14005')
    dataset = load_from_disk('./data/cleavage_data/c14005_test_3')
    for data in dataset:
        if torch.sum(data['site']) < 1:
            print(data)