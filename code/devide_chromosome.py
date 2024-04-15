### requirement of input data
## ① sparse_filepath： sparse matrix file path
## ② index_path： peaks name file path
## ③ colnames_path： cell barcode name file path
## ④ patch_num; eg: 100
## ⑤ output_path： output file path
import scipy
from scipy.io import mmread
import pandas as pd
import re

sparse_matrix = mmread(sparse_filepath)

index_df=pd.read_csv(index_path)
col_df=pd.read_csv(colnames_path)

df = pd.DataFrame(sparse_matrix)#
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#AttributeError: 'numpy.ndarray' object has no attribute 'toarray'
df.index=index_df['x']
df.columns=col_df['x']

df2=df.copy()#
df2['chromosome']=df2.index#
#ValueError: Columns must be same length as key
df2[['Chr','start','end']]=df2['chromosome'].str.split('_', expand=True)


unique_chromosomes =df2['Chr'].unique()
def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')
unique_chromosomes = sorted(unique_chromosomes, key=extract_number)
selected_rows = []

for chromosome in unique_chromosomes:
    subset = df[df2['Chr'] == chromosome]
    row_variances = subset.var(axis=1)
    row_variances=pd.DataFrame(row_variances)
    row_variances.columns=['values']
    row_variances=row_variances.sort_values(by='values',ascending=False)    
    num_rows = len(subset)
    max_integer = num_rows - num_rows % patch_num    
    top_rows = subset.loc[row_variances[:max_integer].index]
    top_rows = pd.DataFrame(top_rows)
    selected_rows.append(top_rows)

for i, df in enumerate(selected_rows):
    df.to_csv(f'{output_path}/chr{i+1}.csv', index=True)