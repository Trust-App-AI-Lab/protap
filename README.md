
# Protap: A Benchmark for Protein Modeling on Realistic Downstream Applications
[![Datset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-orange?label=Model)](https://huggingface.co/datasets/findshuo/Protap)

This project is the codebase for Protap, a comprehensive benchmark that systematically compares backbone architectures, pretraining strategies, and domain-specific models across diverse and realistic downstream protein applications.

## Summary of Pretraining Models in Protap

|**Model** | **Input Modalities** | **Pretrain Data** | **#Params** | **Objective** | **Source** |
|:-------------:|----------------------|-------------------|-------------|---------------|------------|
| 🔴 `(*)` <br> **EGNN**          | AA Seq & 3D Coord  | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 10M  | MLM, MVCL, PFP | [ICML 2021](https://proceedings.mlr.press/v139/satorras21a.html)           |
| 🔴 `(*)` <br> **SE(3) Transformer** | AA Seq & 3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 4M   | MLM, MVCL, PFP | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html) |
| 🔴 `(*)` <br> **GVP**           | AA Seq & 3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 0.2M | MLM, MVCL, PFP | [ICLR 2021](https://openreview.net/forum?id=1YLJDvSx6J4)                    |
| 🔴 `(*)` <br> **ProteinBERT**   | AA Seq           | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 72M  | MLM, MVCL, PFP | [Bioinformatics 2022](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274) |
| 🔴 `(*)` <br> **D-Transformer** | AA Seq &  3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 3.5M | MLM, MVCL, PFP | [ArXiv 2025](https://arxiv.org/abs/2502.06914), [ICLR 2023](https://openreview.net/forum?id=vZTp1oPV3PC) |
| 🔵 `(#)` <br> **ESM2**          | AA Seq           | [UR50 70M](https://www.uniprot.org/help/uniref)                      | 650M | MLM           | [Science 2023](https://www.science.org/doi/10.1126/science.ade2574)         |  
> - 🔴 `(*)` domain-specific models tailored for specific biological tasks, **For Domain-Specific Models, we provide github links.**
> - 🔵 `(#)` Uses publicly available pretrained weights  
> - **AA Seq**: amino acid sequence  
> - **3D Coord**: 3D coordinates of protein structures   

## Illustration of pretraining strategy in Protap.
![Illustration of pretraining tasks in Protap](/figures/pretrain_strategy.png) 
(I) **Masked Language Modeling(MLM)** is a self-supervised objective designed to recover masked residues in protein sequences;  
(II) **Multi-View Contrastive Learning(MVCL)** leverages protein structural information by aligning representations of biologically correlated substructures.  
(III) **Protein Family Prediction(PFP)** introduces functional and structural supervision by training models to predict family labels based on protein sequences and 3D structures.


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



