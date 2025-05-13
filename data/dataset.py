import torch
from torch.utils.data import Dataset as ds
from datasets import Dataset, load_from_disk

from tqdm import tqdm

from data.tokenizers import ProteinTokenizer, ProtacTokenizer

class ProteinDataset(ds):
    def __init__(self):
        """
        Dataset for protein sequences.
        
        Args:
        """
        pass
        
    def encode(self, sequence):
        """
        Encodes a protein sequence into a list of token indices.
        Pads the sequence to the maximum length.
        """
        encoded = [self.amino2dict.get(aa, self.amino2dict['<PAD>']) for aa in sequence]
        
        if len(encoded) < self.max_length:
            encoded += [self.amino2dict['<PAD>']] * (self.max_length - len(encoded))
        
        return encoded
    
    def __len__(self):
        """
        Return the length of the dataset (number of sequences).
        """
        pass
    
    def __getitem__(self, idx):
        pass
    
class EgnnDataset(ProteinDataset):
    """
    Dataset for pre-training EGNN.
    
    Args:
    """
    
    def __init__(
        self,
        tokenizer: ProteinTokenizer,
        generate: bool=False,
        include_family: bool=False,
        include_drug: bool=False,
        include_go: bool=False,
        include_site: bool=False,
        save_dir=None,
    ):  
        # Generate the dataset from scrach.
        if generate:
            self.tokenizer = tokenizer
            self.sequences = self.tokenizer.sequences
            self.data = self.tokenizer.data
            
            self.max_seq_length = self.tokenizer.max_seq_length
            
            # Obtain the 3-d C-alpha coordinate.
            self.coords = [] # (n, seq_length, 3)
            self.family = []
            self.go = []
            for protein in tqdm(self.data):
                coords = protein['coords']
                x = [coord[1] for coord in coords]  # Extract C-alpha coordinates

                # Truncate if necessary
                if len(x) > self.max_seq_length:
                    x = x[:self.max_seq_length]
                # Pad if necessary
                elif len(x) < self.max_seq_length:
                    x += [[0, 0, 0]] * (self.max_seq_length - len(x))
                
                self.coords.append(x)
                
                if include_family:
                    labels = protein['family'] + [-100] * (30 - len(protein['family']))
                    self.family.append(labels)
            
            if include_drug:
                self.drug = self.data['drug']
                self.y = self.data['y']
            
            if include_go:
                self.go = self.data['go']
                
            if include_site:
                self.site = self.data['labels']
            
            self.input_ids, self.masks = self.tokenizer.tokenize()
            
            if include_family:
                self.dataset = {"input_ids" : self.input_ids,
                                "coords" : self.coords,
                                "masks" : self.masks,
                                'family' : self.family
                            }
                
            if include_drug:
                self.dataset = {"input_ids" : self.input_ids,
                                "coords" : self.coords,
                                "masks" : self.masks,
                                'drug' : self.drug,
                                'y' : self.y
                            }
            
            if include_go:
                self.dataset = {
                    "input_ids" : self.input_ids,
                    "coords" : self.coords,
                    "masks" : self.masks,
                    "go" : self.go
                }
            
            if include_site:
                self.dataset = {
                    "input_ids" : self.input_ids,
                    "coords" : self.coords,
                    "masks" : self.masks,
                    "site" : self.site
                }
                
            self.raw_dataset = Dataset.from_dict(self.dataset)
            
            # TODO
            if save_dir:
                self.raw_dataset = self.raw_dataset.save_to_disk(save_dir)
            # else:
            #     self.raw_dataset = self.raw_dataset.save_to_disk('biological_process_1')
        else:
            # self.tokenizer = tokenizer
            # self.sequences = self.tokenizer.sequences
            self.raw_dataset = load_from_disk('swiss-540k-pre-train')
            # self.raw_dataset.set_format(type="torch", columns=["input_ids", "coords", "masks"])
            self.input_ids = self.raw_dataset['input_ids']
            self.coords = self.raw_dataset['coords']
            self.masks = self.raw_dataset['masks']
    
    def __len__(self):
        """
        Return the length of the dataset (number of sequences).
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return the tokenized protein sequence, mask, and 3D coordinates for a given index.
        
        Args:
        - idx (int): Index of the protein sequence.
        
        Returns:
        - A dictionary containing:
          - 'input_ids': The tokenized protein sequence.
          - 'masks': The mask indicating which tokens are padding or masked.
          - 'coords': The 3D coordinates for the protein sequence.
        """
        # Get the sequence and coordinates
        # sequence = self.sequences[idx]
        coords = self.coords[idx]
        
        # Get the input_ids and masks.
        input_ids = self.input_ids[idx]
        masks = self.masks[idx]
        
        
        return {
            'input_ids': torch.tensor(input_ids),
            'coords': torch.tensor(coords),
            'masks': torch.tensor(masks).bool(),
        }

class ProtacDataset(ProteinDataset):
    """
    Dataset for pre-training EGNN.
    
    Args:
    """
    
    def __init__(
        self,
        tokenizer: ProtacTokenizer,
        generate: bool=False,
    ):  
        # Generate the dataset from scrach.
        if generate:
            self.tokenizer = tokenizer
            self.poi_seq = self.tokenizer.poi_seq
            self.e3_ligase_seq = self.tokenizer.e3_ligase_seq
            self.data = self.tokenizer.data
            
            self.max_seq_length = self.tokenizer.max_seq_length
            
            # Obtain the 3-d C-alpha coordinate.
            self.poi_coords, self.e3_ligase_coords = [], [] # (n, seq_length, 3)
            self.label = []
            for protein in tqdm(self.data):
                poi_coords = protein['poi_coord']
                e3_coords = protein['e3_ligase_coord']
                poi_x = [coord[1] for coord in poi_coords]  # Extract C-alpha coordinates
                e3_ligase_x = [coord[1] for coord in e3_coords]

                # Truncate if necessary
                if len(poi_x) > self.max_seq_length:
                    poi_x = poi_x[:self.max_seq_length]
                # Pad if necessary
                elif len(poi_x) < self.max_seq_length:
                    poi_x += [[0, 0, 0]] * (self.max_seq_length - len(poi_x))
                
                                # Truncate if necessary
                if len(e3_ligase_x) > self.max_seq_length:
                    e3_ligase_x = e3_ligase_x[:self.max_seq_length]
                # Pad if necessary
                elif len(e3_ligase_x) < self.max_seq_length:
                    e3_ligase_x += [[0, 0, 0]] * (self.max_seq_length - len(e3_ligase_x))
                
                self.poi_coords.append(poi_x)
                self.e3_ligase_coords.append(e3_ligase_x)
            
            self.label = self.data['label']
            
            self.poi_input_ids, self.poi_masks, self.e3_ligase_input_ids, self.e3_ligase_masks = self.tokenizer.tokenize()
            self.warhead = self.data['warhead']
            self.linker = self.data['linker']
            self.e3_ligand = self.data['e3_ligand']
                
            self.dataset = {
                "poi_input_ids" : self.poi_input_ids,
                "poi_masks" : self.poi_masks,
                "poi_coords" : self.poi_coords,
                "e3_ligase_input_ids" : self.e3_ligase_input_ids,
                "e3_ligase_masks" : self.e3_ligase_masks,
                "e3_ligase_coords" : self.e3_ligase_coords,
                "warhead" : self.warhead,
                "linker" : self.linker,
                "e3_ligand" : self.e3_ligand,
                "label" : self.label
            }
                
            self.raw_dataset = Dataset.from_dict(self.dataset)
            # If save the dataset.
            self.raw_dataset = self.raw_dataset.save_to_disk('protac_1')
        else:
            # self.tokenizer = tokenizer
            # self.sequences = self.tokenizer.sequences
            self.raw_dataset = load_from_disk('swiss-540k-pre-train')
            # self.raw_dataset.set_format(type="torch", columns=["input_ids", "coords", "masks"])
            self.input_ids = self.raw_dataset['input_ids']
            self.coords = self.raw_dataset['coords']
            self.masks = self.raw_dataset['masks']
    
    def __len__(self):
        """
        Return the length of the dataset (number of sequences).
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return the tokenized protein sequence, mask, and 3D coordinates for a given index.
        
        Args:
        - idx (int): Index of the protein sequence.
        
        Returns:
        - A dictionary containing:
          - 'input_ids': The tokenized protein sequence.
          - 'masks': The mask indicating which tokens are padding or masked.
          - 'coords': The 3D coordinates for the protein sequence.
        """
        # Get the sequence and coordinates
        # sequence = self.sequences[idx]
        coords = self.coords[idx]
        
        # Get the input_ids and masks.
        input_ids = self.input_ids[idx]
        masks = self.masks[idx]
        
        
        return {
            'input_ids': torch.tensor(input_ids),
            'coords': torch.tensor(coords),
            'masks': torch.tensor(masks).bool(),
        }