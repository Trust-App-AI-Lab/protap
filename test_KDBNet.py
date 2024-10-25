

import torch
from torch_geometric.data import DataLoader, Batch
from prot_learn.models.kdbnet.dta import KIBA
from prot_learn.models.kdbnet.metrics import evaluation_metrics
from prot_learn.models.kdbnet.KDBNet_model import KDBNet

import warnings
warnings.filterwarnings('ignore')



kiba_task = KIBA(
    mmseqs_seq_cluster_file = 'data/KDBNet_data/KIBA/kiba_cluster_id50_cluster.tsv',
    data_path='data/KDBNet_data/KIBA/kiba_data.tsv',
    pdb_map='data/KDBNet_data/KIBA/kiba_uniprot2pdb.yaml',
    pdb_json='data/KDBNet_data/structure/pockets_structure.json',
    emb_dir='data/KDBNet_data/esm1b',
    drug_sdf_dir='data/KDBNet_data/structure/kiba_mol3d_sdf',
    split_method='random',
    split_frac=[0.7, 0.1, 0.2],
    seed=42,
    onthefly=False
)

split_data = kiba_task.get_split()

train_dataset = split_data['train']
valid_dataset = split_data['valid']
test_dataset = split_data['test']

def custom_collate(batch):
    drug_batch = [item['drug'] for item in batch]
    protein_batch = [item['protein'] for item in batch]
    y_batch = torch.tensor([item['y'] for item in batch], dtype=torch.float)

    drug_batch = Batch.from_data_list(drug_batch)
    protein_batch = Batch.from_data_list(protein_batch)

    return {'drug': drug_batch, 'protein': protein_batch, 'y': y_batch}

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=custom_collate)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = KDBNet(device=device)

model.train_model(train_loader, valid_loader=valid_loader, n_epochs=2, lr=0.001, eval_freq=1)

y_true, y_pred = model.predict(test_loader)
eval_metrics = evaluation_metrics(
    y_true, y_pred,
    eval_metrics=['mse', 'spearman', 'pearson']
)
print('Test Results:')
print(' | '.join([f'{k}: {v:.4f}' for k, v in eval_metrics.items()]))

