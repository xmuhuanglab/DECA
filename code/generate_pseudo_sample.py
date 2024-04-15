### generation of pseudo bulk samples
## requirement of parameters
#  ① scdata_path : tarining single-cell input file path
#  ② metadata_path : include celltype annotation
#  ③ d_prior: Dirichlet(True or False)
#  ④ cellnum: synthesized cell number
#  ⑤ samplenum： pseudo sample number
#  ...
#  This generation code 

import time
import pandas as pd
import numpy as np
import anndata
from tqdm import tqdm
from numpy.random import choice
import pickle
import scanpy as sc#

def generate_pesudo_sample(scdata_path, metadata_path, outname=None,
                            d_prior=None,
                            cellnum=1500, samplenum=1000,
                            random_state=114514, Dominant=True, dominant_prob=0.8,Average=False,
                            Rare=False, rare_percentage=0.2):
    ####################################
    #### Reading single-cell datasets
    ####################################
    print('Reading single-cell ATAC dataset, this may take some minutes')   
    # Reading different file types and merging metadata
    if '.csv' in scdata_path:
        start_time = time.time()
        submeta = pd.read_csv(metadata_path, index_col=0)
        scdata = pd.read_csv(scdata_path, index_col=0)
        df_transposed = scdata.T
        merged_df = df_transposed.merge(submeta, left_index=True, right_on='Cell')
        merged_df.index = merged_df['Cell']
        merged_df = merged_df.drop(columns=['Cell'])
        scdata = merged_df
        scdata.index = range(len(scdata))
        end_time = time.time()
        times = end_time - start_time
        print(f'Reading single-cell matrix execution (.csv) time: {times:.2f} seconds')       
    elif '.txt' in scdata_path:
        start_time = time.time()
        scdata = pd.read_csv(scdata_path, index_col=0, sep='\t')
        scdata.dropna(inplace=True)
        scdata['celltype'] = scdata.index
        scdata.index = range(len(scdata))
        end_time = time.time()
        times = end_time - start_time
        print(f'Reading single-cell matrix execution (.txt) time : {times:.2f} seconds')
    elif '.h5' in scdata_path:
        start_time = time.time()
        scdata = read_h5('file_name', 'file_path')  # Assuming there is a function read_h5
        print(scdata.shape)
        scdata.dropna(inplace=True)
        scdata.index = range(len(scdata))
        end_time = time.time()
        times = end_time - start_time
        print(f'Reading single-cell matrix execution (.h5) time: {times:.2f} seconds')    
    num_celltype = len(scdata['celltype'].value_counts())
    peakname = scdata.drop(columns=['celltype']).columns   
    celltype_groups = scdata.groupby('celltype').groups
    scdata.drop(columns='celltype', inplace=True)
    ####################################
    #### Generating andata and normalize 
    ####################################
    print('Normalizing raw single cell data with scanpy.pp.normalize_total')
    scdata = anndata.AnnData(scdata)
    sc.pp.normalize_total(scdata, target_sum=1e4)#
    #是否要进行标准化
    ##################################################################
    #### Generating pseudo samples based on Random distribution
    ##################################################################
    scdata = scdata.X
    scdata = np.ascontiguousarray(scdata, dtype=np.float32)
    if d_prior is None:
        print('Generating cell fractions using random distribution')
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(np.ones(num_celltype), samplenum)
        print(prop.shape)
        print('RANDOM cell fractions are generated')
    elif d_prior is not None:
        print('Generating cell fractions using Dirichlet distribution')
        assert len(d_prior) == num_celltype, 'dirichlet prior is a vector, its length should equal ' \
                                             'to the number of cell types'
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(d_prior, samplenum)
        print('Dirichlet cell fractions are generated')
    for key, value in celltype_groups.items():
        celltype_groups[key] = np.array(value)
    prop = prop / np.sum(prop, axis=1).reshape(-1, 1)   
    ##################
    ## Distribution
    ##################
    ##########################
    # Dominant cell fractions
    ##########################
    if Dominant:
        print("Dominant is True, some cell fractions will be Dominant, the probability is", dominant_prob)
        for i in range(int(prop.shape[0] * dominant_prob)):
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * dominant_prob))
            prop[i, indices] = 0
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
    ##########################
    # Average cell fractions
    ##########################
    if Average:
        print("Average is True, cell fractions will be Average")
        average_fraction = np.mean(prop, axis=0)       
        for i in range(prop.shape[0]):
            prop[i] = average_fraction
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
    ##########################
    # Rare cell fractions
    ##########################
    if Rare:
        print("Rare is True, cell fractions will be AveRarerage, the probability is", rare_prob)
        np.random.seed(0)
        indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * rare_percentage))
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf
    # precise number for each cell type
    cell_num = np.floor(cellnum * prop)
    print(f'precise number for each cell type is {cell_num.shape}')
    # precise proportion based on cell_num
    prop = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)
    prop.shape  
    # start sampling
    sample = np.zeros((prop.shape[0], scdata.shape[1]))
    allcellname = celltype_groups.keys()
    celltype_sample = {} 
    print('Sampling cells to compose pseudo-bulk ATAC samples')
    for i, sample_prop in tqdm(enumerate(cell_num)):
        celltype_sample[i] = {}  
        for j, cellname in enumerate(allcellname):
            select_index = choice(celltype_groups[cellname], size=int(sample_prop[j]), replace=True).astype(int) 
            sample[i] += scdata[select_index].sum(axis=0)
            celltype_sample[i][cellname] = scdata[select_index].sum(axis=0)
    celltype_sample
    prop = pd.DataFrame(prop, columns=celltype_groups.keys())
    pseudo_data = anndata.AnnData(X=sample,
                               obs=prop,
                               var=pd.DataFrame(index=peakname))
    print('Sampling is done')
    if outname is not None:
        pseudo_data.write_h5ad(outname + '.h5ad')
        with open(outname +'.pkl', 'wb') as file:
            pickle.dump(celltype_sample, file)
    return pseudo_data