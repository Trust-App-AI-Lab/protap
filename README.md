
# Protap

This project is the codebase for Protap, a comprehensive benchmark that systematically compares backbone architectures, pretraining strategies, and domain-specific models across diverse and realistic downstream protein applications.

<div markdown="1">
<small>

### Summary of Pretraining Models in Protap

| **Model** | **Input Modalities** | **Pretrain Data** | **#Params** | **Objective** | **Source** |
|----------|----------------------|-------------------|-------------|---------------|------------|
| 🔴 `(*)` **EGNN**          | <small> AA Seq & 3D Coord </small> | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 10M  | MLM, MVCL, PFP | [ICML 2021](https://proceedings.mlr.press/v139/satorras21a.html)           |
| 🔴 `(*)` **SE(3) Transformer** | AA Seq & 3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 4M   | MLM, MVCL, PFP | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html) |
| 🔴 `(*)` **GVP**           | AA Seq & 3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 0.2M | MLM, MVCL, PFP | [ICLR 2021](https://openreview.net/forum?id=1YLJDvSx6J4)                    |
| 🔴 `(*)` **ProteinBERT**   | AA Seq           | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 72M  | MLM, MVCL, PFP | [Bioinformatics 2022](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274) |
| 🔴 `(*)` **D-Transformer** | AA Seq & 3D Coord | [Swiss-Prot 540k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 3.5M | MLM, MVCL, PFP | [ArXiv 2025](https://arxiv.org/abs/2502.06914), [ICLR 2023](https://openreview.net/forum?id=vZTp1oPV3PC) |
| 🔵 `(#)` **ESM2**          | AA Seq           | [UR50 70M](https://www.uniprot.org/help/uniref)                      | 650M | MLM           | [Science 2023](https://www.science.org/doi/10.1126/science.ade2574)         |

> - 🔴 `(*)` domain-specific models tailored for specific biological tasks, **For Domain-Specific Models, we provide github links.**
> - 🔵 `(#)` Uses publicly available pretrained weights  
> - **AA Seq**: amino acid sequence  
> - **3D Coord**: 3D coordinates of protein structures  
> - **MLM**: masked language modeling  
> - **MVCL**: multi-view contrastive learning  
> - **PFP**: protein function prediction

### Summary of Domain-Specific Models in Protap

| **Model** | **Input Modalities** | **Pretrain Data** | **#Params** | **Objective** | **Source** |
|----------|----------------------|-------------------|-------------|---------------|------------|
| 🟤 `(*)` **ClipZyme**   | AA Seq & 3D Coord & SMILES | —   | 14.8M  | PFS    | [ICML 2024](https://openreview.net/forum?id=0mYAK6Yhhm)                      |
| 🟤 `(*)` **UniZyme**    | AA Seq & 3D Coord           | [Swiss-Prot 11k](https://www.uniprot.org/uniprotkb?query=reviewed:true) | 15.5M  | PFS    | [ArXiv 2025](https://arxiv.org/abs/2502.06914)                               |
| 🟤 `(*)` **DeepProtacs**| AA Seq & 3D Coord & SMILES | —   | 0.1M   | PROTACs| [Nat. Comm 2022](https://www.nature.com/articles/s41467-022-34807-3)         |
| 🟤 `(*)` **ETProtacs**  | AA Seq & 3D Coord & SMILES | —   | 5.4M   | PROTACs| [Brief Bioinf 2025](https://academic.oup.com/bib/article/26/1/bbae654/7948073) |
| 🟤 `(*)` **KDBNet**     | AA Seq & 3D Coord & SMILES | —   | 3.4M   | PLI    | [Nat. Mach Intell 2023](https://www.nature.com/articles/s42256-023-00751-0)   |
| 🟤 `(*)` **MONN**       | AA Seq & 3D Coord           | —   | 1.7M   | PLI    | [Cell Systems 2024](https://www.sciencedirect.com/science/article/pii/S2405471220300818) |
| 🟤 `(*)` **DeepFRI**    | AA Seq & 3D Coord           | [Pfam 10M](https://pfam.xfam.org/)  | 1.8M   | AFP    | [Nat. Comm 2021](https://www.nature.com/articles/s41467-021-23303-9)         |
| 🟤 `(*)` **DPFunc**     | AA Seq & 3D Coord & Domain  | —   | 110M   | AFP    | [Nat. Comm 2025](https://www.nature.com/articles/s41467-024-54816-8)         |

> - 🟤 `(*)` domain-specific models tailored for specific biological tasks, **For Domain-Specific Models, we provide github links.**
> - **PFS**: enzyme-Catalyzed Protein Cleavage Site Prediction  
> - **PROTACs**: Targeted Protein Degradation  
> - **PLI**: Protein–Ligand Interactions  
> - **AFP**: Protein Function Annotation Prediction

</small>
</div>


