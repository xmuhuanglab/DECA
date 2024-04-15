import pandas as pd
import matplotlib.pyplot as plt

def evaluation(workpath,TestPred,pp):
    anno=pd.read_csv(f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/test/Testsets_groundtruth_chr{pp}.csv',index_col=0)
    desired_column_order = list(anno.columns)
    anno = anno[desired_column_order]
    anno.index=['Sample' + str(i) for i in range(1, anno.shape[0] + 1)]    
    TestPred = TestPred[desired_column_order]
    TestPred.index=['Sample'+str(i) for i in range(1,TestPred.shape[0]+1)]
    TestPred = TestPred.div(TestPred.sum(axis=1), axis=0)   
    combined_df = pd.concat([anno.stack(), TestPred.stack()], axis=1)
    combined_df.columns = ['GroundTruth', 'Prediction']
    combined_df=combined_df.apply(pd.to_numeric, errors='coerce')
    correlation = combined_df['GroundTruth'].corr(combined_df['Prediction'], method='pearson')   
    plt.figure(figsize=(3.5, 3))
    plt.scatter(combined_df['GroundTruth'], combined_df['Prediction'], s=6, c='#2d3436', marker='o')
    plt.title(f'Correlation: {correlation:.2f}')
    plt.xlabel('GroundTruth Values')
    plt.ylabel('Prediction Values')
    plt.grid(False)
    plt.show()