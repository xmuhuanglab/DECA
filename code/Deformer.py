import os
import re 
import glob
import pickle
import scipy
from scipy.io import mmread
import numpy as np
import pandas as pd
import test_bulk_func
import matplotlib.pyplot as plt
from test_bulk_func import test_generate_simulated_data, ProcessInputData, predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from generate_pseudo_sample import generate_pesudo_sample
from correct_patch import correct_patch
from train_main import train_model
from adaptive_stage import predict,train_predict
from evaluate import evaluation
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.colors as mcolors


char_list="0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def devide_chr(input_filepath,index_path,colnames_path,output_path,patch_num=100,index_name='x',barcode_name='x',mode='train',batch=None,sample=None):
    
    def extract_number(s):
        match = re.search(r'\d+', s)
        return int(match.group()) if match else float('inf')
    
    if mode == 'train':
        ##run this cell to devide chromosome
        print('Loading count matrix...')
        sparse_matrix = mmread(input_filepath)
        print('Loading peak list...')
        index_df=pd.read_csv(index_path)
        print('Loading barcode list...')
        col_df=pd.read_csv(colnames_path)

        df = pd.DataFrame(sparse_matrix.toarray())

        df.index=index_df[index_name]
        df.columns=col_df[barcode_name]

        df.index.name=None
        df.columns.name=None

        df2=df.copy()

        df2['chromosome']=df2.index

        s=df2['chromosome'][0]
        sep_list = [char for char in s if char not in char_list]

        if sep_list[0]==sep_list[1]:
            df2[['Chr','start','end']]=df2['chromosome'].str.split(sep_list[0], expand=True)
        else:
            df2[['Chr','Region']]=df2['chromosome'].str.split(sep_list[0], expand=True)
            df2[['start','end']]=df2['Region'].str.split(sep_list[1], expand=True)


        unique_chromosomes =df2['Chr'].unique()

        unique_chromosomes = sorted(unique_chromosomes, key=extract_number)
        selected_rows = []

        print('Deviding chromosomes...')

        for chromosome in unique_chromosomes:
            subset = df[df2['Chr'] == chromosome]
            row_variances = subset.var(axis=1)
            row_variances=pd.DataFrame(row_variances)
            row_variances.columns=['values']
            row_variances=row_variances.sort_values(by='values',ascending=False)    
            num_rows = len(subset)
            ###
            max_integer = num_rows - num_rows % patch_num  
            ###
            top_rows = subset.loc[row_variances[:max_integer].index]
            top_rows = pd.DataFrame(top_rows)
            selected_rows.append(top_rows)

    else:
        print('Loading count matrix...')
        
        if batch==None:
            print('Please input your project name')
        df=pd.read_csv(f'{input_filepath}/{batch}_matrix.csv',sep='\t',index_col=0)
        
        if sample == None:
            print('Please input your sample name')

        colname= sample
        df=pd.DataFrame(df)
        df[colname]=pd.to_numeric(df[colname],errors='coerce')

        df2=df
        df2['chromosome']=df2.index
        df2[['Chr','start','end']]=df2['chromosome'].str.split('_', expand=True)
        unique_chromosomes =df2['Chr'].unique()
        unique_chromosomes = sorted(unique_chromosomes, key=extract_number)
        
        print('Deviding chromosomes...')

        selected_rows = []
        # min_count = df2['Chr'].value_counts().min()
        for i,chromosome in  enumerate(unique_chromosomes):
            ref_file=index_path
            back_chr=pd.read_csv(f'{ref_file}/chr{i+1}.csv')
            inter_peak=back_chr['Unnamed: 0']
            subset = df[df2['Chr'] == chromosome]

            top_rows = subset.loc[inter_peak]
            # top_rows = subset
            top_rows = pd.DataFrame(top_rows)
            top_rows = pd.DataFrame(top_rows[colname])
            selected_rows.append(top_rows)

            
    for i, df in enumerate(selected_rows):
        df.to_csv(f'{output_path}/chr{i+1}.csv', index=True)

    file_list = glob.glob(f'{output_path}/chr*.csv')
    print('Completed!')
    
    return file_list    


def custom_sort(item):
    match = re.search(r'chr(\d+)\.csv', item)
    if match:
        chr_number = int(match.group(1))
        return chr_number
    else:
        return float('inf')



def Deformer(workpath,metapath,file_list,samplenum=100,patch_size=100,variance_threshold=1,batch_size=1,epochs=50):
    
    sorted_file_list = sorted(file_list, key=custom_sort)
    
    for pp,train_in,test_in in  zip(range(1,len(sorted_file_list)+1),sorted_file_list,sorted_file_list):        
        simudata_train = generate_pesudo_sample(scdata_path=train_in, samplenum=samplenum, metadata_path=metapath, d_prior=None, Dominant=True,
                                                outname=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/simu_h5ad/train/chr{pp}')

        simudata_test= generate_pesudo_sample(scdata_path=test_in, samplenum=samplenum, metadata_path=metapath, d_prior=None, Dominant=True,
                                              outname=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/simu_h5ad/test/chr{pp}')

        simudata_train.obs.to_csv(f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/train/Trainingsets_groundtruth_chr{pp}.csv', index=True)
        simudata_test.obs.to_csv(f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/test/Testsets_groundtruth_chr{pp}.csv', index=True)      
        train_x, train_y, test_x, peakname, celltypes, samplename = \
            correct_patch(simudata_train,simudata_test,patch_size, datatype='counts', variance_threshold=variance_threshold, scaler='mms')
        with open(f'{workpath}/single-cell_simu/devide_chr_data/simu_input_pkl/data_chr{pp}.pkl', 'wb') as file:
            data_to_save = (train_x, train_y, test_x, peakname, celltypes, samplename)
            pickle.dump(data_to_save, file)
        #batch_size=1
        ##batch_size: sample num calculated each time. smaller batch_size may make model perform better and cost more time. 
        #epochs=50
        ##epoch: train steps, bigger epoches make more steps.May make model perform better and cost more time. 
        input_size=train_x.shape[1]
        ## samples number putting into training, you should no change this
        class_size=train_y.shape[1]
        ##class_size:cell types' number, you should no change this
        output_dim=train_y.shape[1]
        ##output_dim:cell types' number, you should no change this
        state='train'
        ##model's work state.
        train_x_1 = train_x[:, np.newaxis, :]
        train_y_1= train_y[:, np.newaxis, :]
        save_model_name=f'{workpath}/single-cell_simu/devide_chr_data/model/100samples/chr{pp}'
        ## save_model_name: new model's name and save path
        model_name=f'{workpath}/single-cell_simu/devide_chr_data/model/100samples/chr{pp}'
        ## model_name: Now used model's path
        model = train_model(train_x_1, train_y_1, save_model_name, 
                            batch_size=batch_size,input_size=input_size,class_size=class_size,output_dim=output_dim,state=state,epochs=epochs)
        test_x_1 = test_x[:, np.newaxis, :]
        test_x_1.shape
        train_samplename=['Samples' + str(i) for i in range(1, train_x_1.shape[0]+1)]
        train_sigm,Train_CellTypeSigm, trainPred = \
                        train_predict(train_x_1=train_x_1, peakname=peakname, celltypes=celltypes, samplename=train_samplename,
                                model=model_name, model_name=save_model_name,
                                adaptive=True, mode="high-resolution")        
        test_sigm,CellTypeSigm, TestPred = \
                        predict(test_x_1=test_x_1, peakname=peakname, celltypes=celltypes, samplename=samplename,
                                model=model_name, model_name=save_model_name,
                                adaptive=True, mode="high-resolution")        
        with open(f'{workpath}/single-cell_simu/devide_chr_data/predict_output/train/out_chr{pp}.pkl', 'wb') as file:
            data_to_save = (train_sigm, Train_CellTypeSigm, trainPred)
            pickle.dump(data_to_save, file)
        with open(f'{workpath}/single-cell_simu/devide_chr_data/predict_output/test/out_chr{pp}.pkl', 'wb') as file:
            data_to_save = (test_sigm, CellTypeSigm, TestPred)
            pickle.dump(data_to_save, file)
        evaluation(workpath,TestPred,pp)
        with open(f'{workpath}/single-cell_simu/devide_chr_data/process_file.txt', 'a') as file:
            file.write(f'{pp} already down\n')
            
    return 0


def Deformer_Predict(workpath,sample,file_list,celltypes,variance_threshold=1):
    
    sorted_file_list_2 = sorted(file_list, key=custom_sort)

    for pp,test_in in  zip(range(1,len(sorted_file_list_2)+1),sorted_file_list_2):

        print('#############################################'+'\n'+f'       chr{pp} test'+'\n'+'#############################################')

        simudata_test,peakname= test_generate_simulated_data(test_in=test_in)    

        test_x, samplename = \
            ProcessInputData(simudata_test, test_name=None,test_path=None, sep='\t', datatype='counts', genelenfile=None,
                        variance_threshold=1, scaler='mms')

        test_x_1 = test_x[:, np.newaxis, :]
        test_x_1.shape

        model_name=f'./single-cell_simu/devide_chr_data/model/100samples/chr{pp}'  
        
        print(f'Using this celltypes:{celltypes}')
        
        test_sigm,CellTypeSigm, TestPred = \
                        predict(test_x_1=test_x_1, peakname=peakname, celltypes=celltypes, samplename=samplename,
                                model_name=model_name,
                                adaptive=True, mode="high-resolution")

        with open(f'{workpath}/bulk_pred/{sample}/out_chr{pp}.pkl', 'wb') as file:
            data_to_save = (test_sigm, CellTypeSigm, TestPred)
            pickle.dump(data_to_save, file)

        with open(f'{workpath}/bulk_pred/{sample}/process_file.txt', 'a') as file:
            file.write(f'chr {pp} already down\n')



    ####### proportion
    total_TestPred = pd.DataFrame()
    for pp in range(1,24):
        file_path = f'{workpath}/bulk_pred/{sample}/out_chr{pp}.pkl'
        if not os.path.exists(file_path):
            break   
        with open(file_path, 'rb') as file:
                data = pickle.load(file)
        TestPred = pd.DataFrame(data[2])
        total_TestPred=pd.concat([total_TestPred,TestPred])
    mean_TestPred=pd.DataFrame(total_TestPred.mean(axis=0))
    mean_TestPred.columns=['Value']
    total_value = mean_TestPred['Value'].sum()
    mean_TestPred['Scaled Value'] = mean_TestPred['Value'] / total_value
    celltypes = mean_TestPred.index
    percentage = list(mean_TestPred.iloc[:,1])
    
    fig, ax = plt.subplots(figsize=(4.5, 3))
    colors = ['#74b9ff', '#badc58', '#f6e58d', '#95afc0', '#535c68','#fab1a0','#e17055','#f05b72','#8f4b2e','#33a3dc','#fcf16e', '#72777b', '#f2eada']
    bars=plt.barh(celltypes, percentage,color=colors)  
    plt.title(f'{sample} Prediction Percentage')
    plt.gca().set_facecolor('white')
    plt.gca().set_facecolor('white')
    for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('#535c68')    
    plt.xlim(0,1)
    plt.tight_layout()
    for bar, value in zip(bars, percentage):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{value:.2%}', 
                 va='center', ha='left', fontsize=8, color='black')
    plt.savefig(f'{workpath}/bulk_pred/{sample}/figure/{sample}_percentage.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    ####### chromatin accessibility
    dataframes = []
    matrices = []
    for pp in range(1,24):
            file_path = f'{workpath}/bulk_pred/{sample}/out_chr{pp}.pkl'
            if not os.path.exists(file_path):
                break   
            with open(file_path, 'rb') as file:
                    data = pickle.load(file)        
            matrix = pd.DataFrame(data[0])
            matrix =matrix.loc[celltypes,:]
            index = matrix.index
            scaler = MinMaxScaler(feature_range=(0, 10))
            matrix= pd.DataFrame(scaler.fit_transform(matrix), columns=matrix.columns)
            matrix.index = index
            matrices.append(matrix)
    test_acc_total = pd.concat(matrices, axis=1)
    test_acc_total=test_acc_total.loc[celltypes,:]
    max_peaks =5000
    differential_genes = []
    for i in range(int(test_acc_total.shape[0])):
        other_celltypes = test_acc_total.drop(index=celltypes[i])
        print(other_celltypes.shape)
        current_celltype = test_acc_total.iloc[i, :]
        print(current_celltype.shape)
        t_statistic, p_value = ttest_ind(current_celltype, other_celltypes, axis=0)
    
        # Use absolute t-statistic values to find the most significant genes
        abs_t_statistic = np.abs(t_statistic)
        
        # Sort genes based on both absolute t-statistic values and p-values
        sorted_indices = np.argsort(abs_t_statistic)
        top_genes = test_acc_total.columns[sorted_indices[-max_peaks:]]
        
        # Filter genes based on p-value
        top_genes = top_genes[p_value[sorted_indices[-max_peaks:]] < 0.05]
    
        differential_genes.append(top_genes)
    
    for i, genes in enumerate(differential_genes):
        print(f"Celltype -- {celltypes[i]}   differential peaks: {len(genes)}")
    
    merged_index = pd.Index([])
    for index in differential_genes:
        merged_index = pd.Index.union(merged_index, index)
    
    merged_index = merged_index.unique()
    df_accmatrix=test_acc_total.loc[celltypes,merged_index]
    differential_df = pd.DataFrame()
    differential_df = pd.concat([pd.DataFrame(genes, columns=[f'{celltypes[i]}']) for i, genes in enumerate(differential_genes)], axis=1)
    
    plt.close()
    clustered_columns = sns.clustermap(df_accmatrix, cmap="coolwarm", method="average", yticklabels=False, col_cluster=True, row_cluster=True, xticklabels=False)
    plt.close()
    col_order = clustered_columns.dendrogram_col.reordered_ind
    row_order = clustered_columns.dendrogram_row.reordered_ind
    df_accmatrix_clustered = df_accmatrix.iloc[row_order, col_order]
    custom_colors = ["#0984e3","#3498db","#74b9ff","#ecf0f1","#fab1a0", "#e17055", "#e74c3c"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("CustomCmap", custom_colors, N=250)
    plt.figure(figsize=(7, 3))
    sns.heatmap(df_accmatrix_clustered, cmap=custom_cmap, annot=False, cbar=False, xticklabels=False, yticklabels=True)
    plt.title("Reconstruction Chromatin Accessibility",fontsize=15,weight='bold')
    plt.tight_layout()
    plt.savefig(f'{workpath}/bulk_pred/{sample}/figure/{sample}_chromatin_accessibility.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return 0