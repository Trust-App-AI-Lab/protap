

import os
from drug_graph import sdf_to_graphs
import pandas as pd
import json

sdf_dir = './data/structure/davis_mol3d_sdf'  # 你的 SDF 文件夹路径
data_list = {}

for file_name in os.listdir(sdf_dir):
    if file_name.endswith('.sdf'):
        drug_id = file_name.replace('.sdf', '')  # 去掉扩展名作为 key
        file_path = os.path.join(sdf_dir, file_name)
        data_list[drug_id] = file_path

graphs = sdf_to_graphs(data_list)
print(graphs)


df = pd.read_csv("./data/DAVIS/davis_data.tsv",sep='\t')
print(df)

with open("./data/structure/pockets_structure.json","r")as fs:
    structure = json.load(fs)

df_p2pdb = pd.read_csv("./data/DAVIS/davis_protein2pdb.yaml",sep='\t',header=None)
print(df_p2pdb)
davis_dict = dict(df_p2pdb.iloc[:, 0].str.split(': ', expand=True).values)
print(davis_dict)

df['protein_pdb'] = df['protein'].map(davis_dict)

print(df)
df.to_csv("davis_drug_pdb_data.txt",sep='\t')

#print(structure)

#davis_pdb = [davis_dict[k] for k in davis if k in davis_dict]

#print(davis_p[db)

backbone_order = ["N", "CA", "C", "O"]

protein = {}
for i in structure.items():
    prot = {}
    atom_coords_dict = i[1]["coords"]
    seq = i[1]["seq"]
    num_residue = len(seq)
    coords = [[atom_coords_dict[atom][i] for atom in backbone_order] for i in range(num_residue)]
    prot["uniprot_id"] = i[1]["UniProt_id"]
    prot["seq"] = seq
    prot["coord"] = coords
    protein[i[0]] = prot

print(protein)
print(len(protein))

with open("pli_structure.json","w")as fp:
    json.dump(protein,fp,indent=4)

