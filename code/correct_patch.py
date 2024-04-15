### correct patch size of pseudo bulk samples
## requirement of parameters
#  ① train_data: generated training datasets
#  ② test_data
#  ③ scale: scale function; eg:mms
#  ④ patch_size： eg:100



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import anndata


def correct_patch(train_data, test_data,patch_size=100, datatype='counts', variance_threshold=0.95,
                     scaler="mms"):


    """
    Corrects and scales the input train and test data for analysis.

    Args:
    train_data (Union[AnnData, str]): Training data in AnnData format or path to an H5AD file.
    test_data (Union[AnnData, str]): Test data in AnnData format or path to an H5AD file.
    patch_size (int, optional): Size of the patch. Defaults to 100.
    datatype (str, optional): Type of data. Defaults to 'counts'.
    variance_threshold (float, optional): Variance threshold for feature selection. Defaults to 0.95.
    scaler (str, optional): Scaler to use ('ss' for StandardScaler, 'mms' for MinMaxScaler). Defaults to "mms".

    Returns:
    tuple: Tuple containing corrected and scaled train and test data along with peak names, cell types, and sample names.
    """
    
    ### read train data
    print('Reading training data (pseudo Bulk ATAC)')
    if type(train_data) is anndata.AnnData:
        pass
    elif type(train_data) is str:
        train_data = anndata.read_h5ad(train_data)
    train_data=train_data
    train_x = pd.DataFrame(train_data.X, columns=train_data.var.index)
    train_y = train_data.obs
    if type(test_data) is anndata.AnnData:
        pass
    elif type(test_data) is str:
        test_data = anndata.read_h5ad(test_data)
    # train_data.var_names_make_unique()
    test_data=test_data
    test_x = pd.DataFrame(test_data.X, columns=test_data.var.index)
    test_y = test_data.obs
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_threshold)-1]
    train_x = train_x.loc[:, train_x.var(axis=0) >= var_cutoff]
    var_cutoff = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_threshold)-1]
    test_x = test_x.loc[:, test_x.var(axis=0) >= var_cutoff]
    ### find intersected peaks
    print('Finding intersected peaks...')
    inter = train_x.columns.intersection(test_x.columns)
    num_columns = len(inter)
    if num_columns % patch_size != 0:
        num_columns = (num_columns // patch_size) * patch_size
        inter = inter[:num_columns]
    train_x = train_x[inter]
    test_x = test_x[inter]
    peakname = list(inter)
    celltypes = train_y.columns
    samplename = test_x.index
    ### MinMax process
    print('Scaling...')
    train_x = np.log(train_x + 1)
    test_x = np.log(test_x + 1)
    colors = sns.color_palette('RdYlBu', 10)
    fig = plt.figure(figsize=(3.5, 3))
    sns.histplot(data=np.mean(train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
    sns.histplot(data=np.mean(test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
    plt.legend(title='datatype', labels=['trainingdata', 'testdata'])
    plt.show()
    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        ss_train_x = ss.fit_transform(train_x.T).T
        ss_test_x = ss.fit_transform(test_x.T).T
        fig = plt.figure(figsize=(3.5, 3))
        sns.histplot(data=np.mean(ss_train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
        sns.histplot(data=np.mean(ss_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=['trainingdata', 'testdata'])
        plt.show()
        return ss_train_x, train_y.values, ss_test_x, peakname, celltypes, samplename
    elif scaler == 'mms':
        print("Using minmax scaler...")
        mms = MinMaxScaler()
        mms_train_x = mms.fit_transform(train_x.T).T
        mms_test_x = mms.fit_transform(test_x.T).T
        fig = plt.figure(figsize=(3.5, 3))
        sns.histplot(data=np.mean(mms_train_x, axis=0), kde=True, color=colors[3],edgecolor=None)
        sns.histplot(data=np.mean(mms_test_x, axis=0), kde=True, color=colors[7],edgecolor=None)
        plt.legend(title='datatype', labels=['trainingdata', 'testdata'])
        plt.show()
        return mms_train_x, train_y.values, mms_test_x, peakname, celltypes, samplename


