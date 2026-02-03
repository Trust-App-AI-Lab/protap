
# Protap: A Benchmark for Protein Modeling on Realistic Downstream Applications

This project is the codebase for Protap, a comprehensive benchmark that systematically compares backbone architectures, pretraining strategies, and domain-specific models across diverse and realistic downstream protein applications.

<details open><summary><b>Table of Contents</b></summary>

- [Summary of Pretraining Models](#summary-of-pretraining-models-in-protap)
- [Pretraining Strategy Illustration](#illustration-of-pretraining-strategy-in-protap)
- [Summary of Domain-Specific Models](#summary-of-domain-specific-models-in-protap)
- [Performance comparison across model architectures under different training strategies](#performance-comparison-across-model-architectures-under-different-training-strategies)
- [Environment Installation](#environment-installation)
- [Dataset](#Dataset)
- [Usage](#usage)
  - [Pretrain](#pretrain)
  - [Downstream Applications](#downstream-applications)

</details>


## Summary of Pretraining Models in Protap

|**Model** | **Input Modalities** | **Pretrain Data** | **#Params** | **Objective** | **Source** |
|:-------------:|----------------------|-------------------|-------------|---------------|------------|
| 🔴 `(*)` <br> **EGNN**          | AA Seq & 3D Coord  | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 10M  | MLM, MVCL, PFP | [ICML 2021](https://proceedings.mlr.press/v139/satorras21a.html)           |
| 🔴 `(*)` <br> **SE(3) Transformer** | AA Seq & 3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 4M   | MLM, MVCL, PFP | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html) |
| 🔴 `(*)` <br> **GVP**           | AA Seq & 3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 0.2M | MLM, MVCL, PFP | [ICLR 2021](https://openreview.net/forum?id=1YLJDvSx6J4)                    |
| 🔴 `(*)` <br> **ProteinBERT**   | AA Seq           | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 72M  | MLM, MVCL, PFP | [Bioinformatics 2022](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274) |
| 🔴 `(*)` <br> **D-Transformer** | AA Seq &  3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 3.5M | MLM, MVCL, PFP | [ArXiv 2025](https://arxiv.org/abs/2502.06914), [ICLR 2023](https://openreview.net/forum?id=vZTp1oPV3PC) |
| 🔵 `(#)` <br> **ESM2**          | AA Seq           | [UR50 70M](https://www.uniprot.org/help/uniref)                      | 650M | MLM           | [Science 2023](https://www.science.org/doi/10.1126/science.ade2574)         |  
> - 🔴 `(*)` Trained from scratch, **For models trained from scratch, we provide complete training code as well as downstream evaluation scripts.**
> - 🔵 `(#)` Uses publicly available pretrained weights  
> - **AA Seq**: amino acid sequence  
> - **3D Coord**: 3D coordinates of protein structures   

## Illustration of pretraining strategy in Protap.
![Illustration of pretraining tasks in Protap](/figures/pretrain_strategy.png) 
> (I) **Masked Language Modeling(MLM)** is a self-supervised objective designed to recover masked residues in protein sequences;  
> (II) **Multi-View Contrastive Learning(MVCL)** leverages protein structural information by aligning representations of biologically correlated substructures.  
> (III) **Protein Family Prediction(PFP)** introduces functional and structural supervision by training models to predict family labels based on protein sequences and 3D structures.


## Summary of Domain-Specific Models in Protap

| **Model** | **Input Modalities** | **Pretrain Data** | **#Params** | **Objective** | **Source** | **Github** |
|:----------:|----------------------|-------------------|-------------|---------------|------------|:--------:|
| 🟤 `($)` <br> **ClipZyme**   | AA Seq & 3D Coord & SMILES | — | 14.8M | PFS | [ICML&nbsp;2024](https://openreview.net/forum?id=0mYAK6Yhhm) | [:octocat:](https://github.com/pgmikhael/clipzyme) |
| 🟤 `($)` <br> **UniZyme**    | AA Seq & 3D Coord | [Swiss-Prot&nbsp;11k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 15.5M | PFS | [ArXiv&nbsp;2025](https://arxiv.org/abs/2502.06914) | [:octocat:](https://anonymous.4open.science/r/UniZyme-4A67) |
| 🟤 `($)` <br> **DeepProtacs**| AA Seq & 3D Coord & SMILES | — | 0.1M | PROTACs | [Nat.&nbsp;Comm&nbsp;2022](https://www.nature.com/articles/s41467-022-34807-3) |  [:octocat:](https://github.com/Fenglei104/DeepPROTACs)|
| 🟤 `($)` <br> **ET-Protacs**  | AA Seq & 3D Coord & SMILES | — | 5.4M | PROTACs | [Brief&nbsp;Bioinf&nbsp;2025](https://academic.oup.com/bib/article/26/1/bbae654/7948073) |  [:octocat:](https://github.com/GuanyuYue/ET-PROTACs)|
| 🟤 `($)` <br> **KDBNet**     | AA Seq & 3D Coord & SMILES | — | 3.4M | PLI | [Nat.&nbsp;Mach.&nbsp;Intell&nbsp;2023](https://www.nature.com/articles/s42256-023-00751-0) |  [:octocat:](https://github.com/luoyunan/KDBNet) |
| 🟤 `($)` <br> **MONN**       | AA Seq & 3D Coord | — | 1.7M | PLI | [Cell&nbsp;Systems&nbsp;2024](https://www.sciencedirect.com/science/article/pii/S2405471220300818) |  [:octocat:](https://github.com/lishuya17/MONN) |
| 🟤 `($)` <br> **DeepFRI**    | AA Seq & 3D Coord | [Pfam&nbsp;10M](https://pfam.xfam.org/) | 1.8M | AFP | [Nat.&nbsp;Comm&nbsp;2021](https://www.nature.com/articles/s41467-021-23303-9) |  [:octocat:](https://github.com/flatironinstitute/DeepFRI) |
| 🟤 `($)` <br> **DPFunc**     | AA Seq & 3D Coord & Domain | — | 110M | AFP | [Nat.&nbsp;Comm&nbsp;2025](https://www.nature.com/articles/s41467-024-54816-8) |  [:octocat:](https://github.com/CSUBioGroup/DPFunc) |


> - 🟤 `($)` domain-specific models tailored for specific biological tasks, **For Domain-Specific Models, we provide github links.**
> - **PFS**: enzyme-Catalyzed Protein Cleavage Site Prediction  
> - **PROTACs**: Targeted Protein Degradation  
> - **PLI**: Protein–Ligand Interactions  
> - **AFP**: Protein Function Annotation Prediction

## Performance comparison across model architectures under different training strategies
![Performance comparison across model architectures under different training strategies](/figures/Performance_comparison.png) 


## Environment installation
```
conda create -n protap python=3.12
conda activate protap
pip install -r requirements.txt
```

## Dataset
The dataset used for downstream evaluation is available on Hugging Face: [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-orange?label=Protap%20Dataset)](https://huggingface.co/datasets/findshuo/Protap)

You can also access it directly at:
👉 **[https://huggingface.co/datasets/findshuo/Protap](https://huggingface.co/datasets/findshuo/Protap)**


## Usage
### Pretrain
> To pretrain from scratch on the Swiss-Prot 540k dataset, simply execute the corresponding bash script for each model. The pretraining strategy and other parameters are customizable. An example of the bash script arguments is shown below:
```bash
torchrun --nproc_per_node=8  egnn_pretrain.py \
    --model_name_or_path "protap/egnn" \
    --data_path "./data/protein_family_2" \
    --bf16 True \
    --output_dir "./checkpoints/egnn/" \
    --run_name 'egnn-pretrain-family-0419' \
    --residue_prediction False \
    --subseq_length 50 \
    --max_nodes 50 \
    --temperature 0.01 \
    --task 'family_prediction' \
    --num_train_epochs 70 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --fsdp no_shard \
```

| Parameter                       | Description                                                         | Example Value                 |
| ------------------------------- | ------------------------------------------------------------------- | ----------------------------- |
| **Distributed & Hardware**      |                                                                     |                               |
| `--nproc_per_node`              | Number of GPUs for `torchrun` per node                              | `8`                           |
| `--fsdp`                        | FSDP sharding strategy: `no_shard`, `full_shard`, or `auto_wrap`    | `no_shard`                    |
| **Model & Dataset**             |                                                                     |                               |
| `--model_name_or_path`          | Hugging Face model name or local pretrained checkpoint              | `"protap/egnn"`               |
| `--data_path`                   | Path to the pretraining dataset                                     | `"./data/protein_family_2"`   |
| `--residue_prediction`          | Enable residue-level prediction (like MLM or site prediction)       | `False`                       |
| `--task`                        | Pretraining task: `family_prediction`, `residue_prediction` or `MVCL`       | `'family_prediction'`         |
| **Sequence & Graph Settings**   |                                                                     |                               |
| `--subseq_length`               | Max subsequence length (number of residues per sample)              | `50`                          |
| `--max_nodes`                   | Maximum number of graph nodes (residues)                            | `50`                          |
| `--temperature`                 | Temperature for contrastive/probabilistic losses                    | `0.01`                        |
| **Training Loop**               |                                                                     |                               |
| `--num_train_epochs`            | Total number of training epochs                                     | `70`                          |
| `--per_device_train_batch_size` | Training batch size per GPU                                         | `48`                          |
| `--per_device_eval_batch_size`  | Evaluation batch size per GPU                                       | `4`                           |
| `--learning_rate`               | Initial learning rate                                               | `1e-3`                        |
| `--weight_decay`                | Weight decay (L2 regularization)                                    | `0.`                          |
| `--warmup_ratio`                | Fraction of steps for LR warmup                                     | `0.05`                        |
| `--lr_scheduler_type`           | LR schedule: `constant`, `linear`, `cosine`, `constant_with_warmup` | `"constant_with_warmup"`      |
| **Evaluation & Logging**        |                                                                     |                               |
| `--evaluation_strategy`         | When to run evaluation: `no`, `steps`, `epoch`                      | `"no"`                        |
| `--logging_steps`               | Logging interval in steps                                           | `1`                           |
| **Checkpointing**               |                                                                     |                               |
| `--output_dir`                  | Directory to save checkpoints and weights                              | `"./checkpoints/egnn/"`       |
| `--run_name`                    | Run name for wandb logging                                      | `'egnn-pretrain-family-0419'` |
| `--save_strategy`               | When to save checkpoints: `steps` or `epoch`                        | `"steps"`                     |
| `--save_steps`                  | Save checkpoint every N steps                                       | `5000`                        |
| `--save_total_limit`            | Maximum number of checkpoints to keep (older deleted)               | `1`                           |
| **Precision & Performance**     |                                                                     |                               |
| `--bf16`                        | Enable bfloat16 mixed-precision training                            | `True`                        |

### Downstream Applications
> To evaluate pretrained models on various downstream tasks, please download the dataset from Hugging Face and run the corresponding bash script for each task. The dataset path, pretrained weights, and other parameters are customizable. An example of the bash script arguments is shown below:
```bash
torchrun --nproc_per_node=8 --master_port=23333 egnn_protac.py \
    --model_name_or_path './checkpoints/egnn_contrastive.pt' \
    --data_path "./data/protac_2" \
    --bf16 True \
    --output_dir "./checkpoints/egnn/" \
    --run_name 'egnn-protac-cl-0428' \
    --residue_prediction False \
    --subseq_length 50 \
    --max_nodes 50 \
    --temperature 0.01 \
    --num_train_epochs 50 \
    --seed 1024 \
    --load_pretrain True \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
```
| Parameter                       | Description                                                         | Example Value                         |
| ------------------------------- | ------------------------------------------------------------------- | ------------------------------------- |
| **Distributed & Hardware**      |                                                                     |                                       |
| `--master_port`                 | TCP port for multi-GPU communication                                | `23333`                               |
| **Model & Dataset**             |                                                                     |                                       |
| `--load_pretrain`               | Whether to load pretrained weights for fine-tuning                  | `True`                                |
| **Training Loop**               |                                                                     |                                       |
| `--learning_rate`               | Typically smaller than in pretraining                               | `5e-4`                                |
| `--num_train_epochs`            | Often fewer epochs than in pretraining                              | `50`                                  |
| `--per_device_train_batch_size` | Often smaller than in pretraining                                   | `24`                                  |
| **Evaluation & Logging**        |                                                                     |                                       |
| `--seed`                        | Random seed for reproducibility                                     | `1024`                                |
