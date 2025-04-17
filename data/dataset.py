import torch
from torch.utils.data import Dataset as ds
from datasets import Dataset, load_from_disk

from tqdm import tqdm

from data.tokenizers import ProteinTokenizer

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
                                'drug' : self.drug
                            }
                
            self.raw_dataset = Dataset.from_dict(self.dataset)
            
            # TODO
            self.raw_dataset = self.raw_dataset.save_to_disk('protein_drug_1')
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