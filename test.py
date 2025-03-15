import torch
import json
from datasets import Dataset, load_dataset
from tqdm import tqdm

from data.tokenizers import *
from data.dataset import *

if __name__ == '__main__':
    # with open('/mnt/data/shuoyan/swiss_540k_list.json', 'r') as json_file:
    #     data = json.load(json_file)
        
    # torch.save(data, './data/swiss_50k.pt')
    # data = Dataset.from_json('/mnt/data/shuoyan/swiss_540k_list.json')
    # data.save_to_disk('./data/swiss_50k')
    # data = torch.load('./data/swiss_50k.pt')
    # print(data[0])
    
    # with open('/mnt/data/shuoyan/swiss_540k_list.json') as json_file:
    #     data = json.load(json_file)
        
    # seq = []
    # coords = []
    # for i in range(len(data)):
    #     seq.append(data[i]['seq'])
    #     coords.append(data[i]['coords'])
        
    # dataset = {
    #     'seq' : seq,
    #     'coords' : coords
    # }
    
    # # dataset = Dataset.from_dict(dataset)
    # # dataset = dataset.save_to_disk('swiss-50k-hf')
    # dataset = Dataset.load_from_disk('swiss-50k-hf')
    # # dataset = load_dataset('swiss-50k-hf', streaming=True, num_proc=8)
    # print(dataset[0])
    
    # dataset = Dataset.load_from_disk('swiss-50k-hf')
    # print("Load Complete!")
    # tokenizer = ProteinTokenizer(max_seq_length=768, dataset=dataset, padding_to_longest=False)
    # dataset = EgnnDataset(tokenizer=tokenizer)
    
    dataset = load_from_disk('./data/egnn_data/swiss-540k-pre-train')
    input_ids, coords, masks = [], [], []
    for protein in tqdm(dataset):
        input_ids.append(torch.tensor(protein['input_ids']))
        coords.append(torch.tensor(protein['coords']))
        masks.append(torch.tensor(protein['masks']).bool())
    
    input_ids = torch.stack(input_ids)
    coords = torch.stack(coords)
    masks = torch.stack(masks)
    dataset = {
        "input_ids" : input_ids,
        "coords" : coords,
        "masks" : masks
    }
    
    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type='torch', columns=['input_ids', 'coords', 'masks'])
    dataset = dataset.save_to_disk("swiss-protein-540k-tensor")
    
    dataset = load_from_disk("swiss-protein-540k-tensor")
    print(dataset[0])