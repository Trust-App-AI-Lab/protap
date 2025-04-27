import os
import json
from datasets import Dataset, load_from_disk

from data.dataset import EgnnDataset, ProtacDataset
from data.tokenizers import ProteinTokenizer, ProtacTokenizer
from data.drug_graph import sdf_to_graphs

import torch
from tqdm import tqdm
import pandas as pd

from models.drug_gvp.drug_gvp import DrugGVPModel

def generate_pretrain_dataset():
    raw_data = '/mnt/data/protein_data/swiss_540k_family_index_list.json'
    # raw_data = '/mnt/data/shuoyan/swiss_540k_list.json'

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
    tokenizer = ProtacTokenizer(max_seq_length=768, dataset=dataset, padding_to_longest=False)
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
    dataset = load_from_disk('./data/protac_1')
    print(dataset[0])
    print(dataset[0]['warhead'])
    print(dataset[100]['warhead'])
    print(dataset[1000]['warhead'])