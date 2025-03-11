import json
import numpy as np


# ProteinTokenizer class to tokenize protein sequences
class ProteinTokenizer:
    def __init__(
        self,
        max_seq_length: int,
        dataset,
        amino_dict=None,
        padding_to_longest=False,
    ):
        if dataset == 'egnn-data':
            # keys: ['name', 'seq', 'coords']
            # with open('./data/egnn_data/ts50.json', 'r') as json_file:
            #     self.data = json.load(json_file)
            with open('/mnt/data/shuoyan/swiss_20k.json', 'r') as json_file:
                self.data = json.load(json_file)
                
        # Extract sequences
        self.sequences = [protein['seq'] for protein in self.data]
        self.max_length = np.max([len(seq) for seq in self.sequences]) # Specify the max length of the amino acids sequence in the training data.
        self.max_index = np.argmax([len(seq) for seq in self.sequences])
        
        # Pad the sequence to the max sequence length.
        if padding_to_longest:
            self.max_seq_length = self.max_length
        else:
            self.max_seq_length = max_seq_length

        # Build amino acid dictionary
        # Collect unique amino acids from the longest sequence
        self.amino_dict = amino_dict if amino_dict != None else list(set(self.sequences[self.max_index]))
        self.amino_dict.append('<MASK>') # Add mask token for residue prediction.
        self.amino_dict.append('<PAD>')  # Add padding token to pad the amino sequence to the same length for batch training.
        self.amino_numbers = len(self.amino_dict) # Obtain the amino acids types, including the <MASK> and the <PAD>.
        # Mapping from amino acid to index
        self.amino2dict = {amino: idx for idx, amino in enumerate(self.amino_dict)}
        
    def encode(self, sequence):
        """
        Encodes a protein sequence into a list of token indices.
        Pads the sequence to the maximum length.
        """
        # TODO: Obtain the max length.
        # Map each amino acid in the sequence to its corresponding index in amino2dict
        encoded = [self.amino2dict.get(aa, self.amino2dict['<PAD>']) for aa in sequence]
        
        # Pad the sequence if it's shorter than max_length
        if len(encoded) < self.max_seq_length:
            encoded += [self.amino2dict['<PAD>']] * (self.max_seq_length - len(encoded))
        
        return encoded
    
    def tokenize(self):
        """
        Tokenize the dataset.
        Return the tokens and the masks.
        """
        print(f"Tokening the data...")
        input_ids = list(map(self.encode, self.sequences))
        # Generate masks (1 for non-padding, 0 for padding)
        masks = [[1 if token != self.amino2dict['<PAD>'] else 0 for token in seq] for seq in input_ids]
        
        return input_ids, masks
    
    def decode(self, tokens):
        """
        Decodes a list of token indices back to a protein sequence.
        """
        reversed_dict = {v: k for k, v in self.amino2dict.items()}
        decoded = [reversed_dict.get(token, '<PAD>') for token in tokens]
        
        # Strip padding
        return ''.join(decoded).replace('<PAD>', '').strip()
    
    def pad(self, sequence):
        """
        Pads a sequence to the maximum length if it's shorter.
        """
        return sequence + [self.amino2dict['<PAD>']] * (self.max_length - len(sequence)) if len(sequence) < self.max_length else sequence
