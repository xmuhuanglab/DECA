# **DECA**
Harnessing interpretable transformer model for cellular deconvolution of chromatin accessibility profile
## **Workflow:**
![Image text](https://github.com/xmuhuanglab/Deformer/blob/main/Description/Deformer_v2.png)
## **Introduction of DECA:**
Within the confines of this study, we introduce Deformer, a deep learning-driven deconvolution technique designed specifically for Bulk ATAC datasets. DECA utilizes vision transformers (ViT) coupled with a decoder, harnessing the power of multi-head attention mechanisms to delineate and analyze chromosomal characteristics. This approach enables the capture of long-range dependencies within non-coding regions. Through DECA, accurate predictions of cell type proportions and cell type-specific chromatin accessibility can be achieved across a wide sample types.

## **Installation of DECA:**
#### Dependencies
```
Python version: 3.9.17; PyTorch version: 2.0.1+cu117; scikit-learn version: 1.3.0;
einops version: 0.6.1; NumPy version: 1.23.5; Pandas version: 1.5.3
tqdm version: 4.66.1; Scanpy version: 1.9.5, 
other requirements see requirements.txt
```
#### Install
```
git clone https://github.com/xmuhuanglab/DECA.git
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda env create -f environment.yaml
conda activate DECA
```

## **Structure of DECA:**
#### I. Preparation of input:
![Image text](https://github.com/xmuhuanglab/DECA/blob/main/Description/Devide_chromosome.png)
```
code/devide_chromosome.py
```
#### II. Generation of pseudo bulk:
![Image text](https://github.com/xmuhuanglab/DECA/blob/main/Description/pseudo-bulk.png)
```
code/generate_pseudo_sample.py, code/correct_patch.py
```
#### III. Model structure:
![Image text](https://github.com/xmuhuanglab/DECA/blob/main/Description/biological_insight.png)
```
code/utils.py, code/train_main.py, code/train_stage.py, code/evaluate.py
```

## **Running:**
### **For training and predicting:**
```
Training.ipynb, Prediction.ipynb
```

## **Contact:**
For any inquiries or assistance, please feel free to open an issue or reach out to sluo112211@163.com or jhuang@xmu.edu.cn






