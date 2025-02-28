import torch
from torch.utils.data import Dataset

from data.tokenizers import ProteinTokenizer

class ProteinDataset(Dataset):
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
    ):  
        self.tokenizer = tokenizer
        self.sequences = self.tokenizer.sequences
        self.data = self.tokenizer.data
        
        self.max_seq_length = self.tokenizer.max_seq_length
        # Obtain the 3-d C-alpha coordinate.
        self.coords = [] # (n, seq_length, 3)
        for protein in self.data:
            coords = protein['coords']
            x = []
            # Iterate the amino acids in the sequence.
            for coord in coords:
                x.append(coord[1]) # The second one is the C-alpha coordinate.
            self.coords.append(x)
        print(self.coords[0])

        # Padding for coordinates.
        self.coords = [coord + [[0, 0, 0]] * (self.max_seq_length - len(coord)) if len(coord) < self.max_seq_length else coord for coord in self.coords]
        
        self.input_ids, self.masks = self.tokenizer.tokenize()
    
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
        sequence = self.sequences[idx]
        coords = self.coords[idx]
        
        # Get the input_ids and masks.
        input_ids = self.input_ids[idx]
        masks = self.masks[idx]
        
        return {
            'input_ids': torch.tensor(input_ids),
            'coords': torch.tensor(coords),
            'masks': torch.tensor(masks).bool(),
        }