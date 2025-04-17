import os
import json
from datasets import Dataset, load_from_disk

from data.dataset import EgnnDataset
from data.tokenizers import ProteinTokenizer
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
    tokenizer = ProteinTokenizer(max_seq_length=768, dataset=dataset, padding_to_longest=False)
    dataset = EgnnDataset(tokenizer=tokenizer, generate=True, include_family=True)

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
    for x in tqdm(dataset):
       protein = x['protein_pdb'] # Query the protein coords and seq with the PDB id.
       protein_data = proteins[protein]
       seq.append(protein_data['seq'])
       coords = protein_data['coord']
    #    coords_ = [coord[1] for coord in coords]
       coord.append(coords)
       drug.append(x['drug']) # Store the drug id for joint training.
    dataset = {
        'seq' : seq,
        'coords' : coord,
        'drug' : drug
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

    input_ids, coords, masks = [], [], []
    drugs = []
    for protein in tqdm(dataset):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
        drugs.append(torch.tensor(protein['drug']))

    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    drugs = torch.stack(drugs)
    dataset = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks,
        "drugs" : drugs
    }

    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type='torch', columns=['input_ids', 'coords', 'masks', 'drugs'])
    dataset = dataset.save_to_disk("protein_drug_2")

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
    
    model = DrugGVPModel()
    
    for k,v in drug_graphs.items():
        print(v.edge_v)
        print(model(v).shape)
        break
    
    return drug_graphs

if __name__ == '__main__':
    
    print("Generating......")
    
    # generate_pretrain_dataset()
    
    # dataset = generate_pli_dataset()
    # dataset = load_from_disk("protein_drug_2")
    # print(len(dataset))
    # print(dataset[1000])
    
    # data = load_from_disk('./data/protein_family_2')
    # print(data['input_ids'][155])
    
    # input_ids = data['input_ids'][155]
    # seq = """
    # MVFPIRILVLFALLAFPACVHGAIRKYTFNVVTKQVTRICSTKQIVTVNGKFPGPTIYANEDDTILVNVVNNVKYNVSIHWHGIRQLRTGWADGPAYITQCPIKPGHSYVYNFTVTGQRGTLWWHAHVLWLRATVHGAIVILPKLGLPYPFPKPHREEVIILGEWWKSDTETVVNEALKSGLAPNVSDAHVINGHPGFVPNCPSQGNFKLAVESGKTYMLRLINAALNEELFFKIAGHRFTVVEVDAVYVKPFNTDTILIAPGQTTTALVSAARPSGQYLIAAAPFQDSAVVAVDNRTATATVHYSGTLSATPTKTTSPPPQNATSVANTFVNSLRSLNSKTYPANVPITVDHDLLFTVGLGINRCHSCKAGNFSRVVAAINNITFKMPKTALLQAHYFNLTGIYTTDFPAKPRRVFDFTGKPPSNLATMKATKLYKLPYNSTVQVVLQDTGNVAPENHPIHLHGFNFFVVGLGTGNYNSKKDSNKFNLVDPVERNTVGVPSGGWAAIRFRADNPGVWFMHCHLEVHTTWGLKMAFLVENGKGPNQSIRPPPSDLPKC
    # """
    
    dataset = get_drug_graph()
    print(dataset)