# Deformer
Harnessing Deep Learning for cellular deconvolution of chromatin accessibility data
## Workflow:
![Image text](https://github.com/xmuhuanglab/Deformer/Description/Deformer.png)

## Introduction of Deformer:
Within the confines of this study, we introduce Deformer, a deep learning-driven deconvolution technique designed specifically for Bulk ATAC datasets. Deformer utilizes vision transformers (ViT) coupled with a decoder, harnessing the power of multi-head attention mechanisms to delineate and analyze chromosomal characteristics. This approach enables the capture of long-range dependencies within non-coding regions. Through Deformer, accurate predictions of cell type proportions and cell type-specific chromatin accessibility can be achieved across a wide sample types.

## Installation of Deformer:
### Requirement
```
Python version: 3.9.17; PyTorch version: 2.0.1+cu117; scikit-learn version: 1.3.0;
einops version: 0.6.1; NumPy version: 1.23.5; Pandas version: 1.5.3
tqdm version: 4.66.1; Scanpy version: 1.9.5, 
other requirements see requirements.txt
```
### I. Install
```
git clone https://github.com/xmuhuanglab/Deformer.git
```
### II. Configure GPU environment
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
### III. Set up Deformer environment
```
conda env create -f environment.yml
conda activate Deformer
```

## Structure of Deformer:
### I. Preparation of demo datasets:
```
devide_chromosome.py
```
### II. Generation of pseudo bulk:
```
generate_pseudo_sample.py
correct_patch.py
```
### III. Model components:
```
utils.py 
train_main.py
train_stage.py; evaluate.py
```
### IV. Model training:
```
main.py
```
### V. Model prediction:
```
predict.py
```

## Running:
### For training:
```
Training.ipynb
```
### For prediction: 
```
Prediction.ipynb
```

## Contact:
For any inquiries or assistance, please feel free to open an issue or reach out to sluo112211@163.com or jhuang@xmu.edu.cn






