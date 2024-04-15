import glob
import os


folder_path = ''
workpath=''

file_list = glob.glob(f'{folder_path}/chr*.csv')

def custom_sort(item):
    match = re.search(r'chr(\d+)\.csv', item)
    if match:
        chr_number = int(match.group(1))
        return chr_number
    else:
        return float('inf')
sorted_file_list = sorted(file_list, key=custom_sort)

if __name__ == "__main__":
    
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/simu_h5ad/train/'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check)
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/simu_h5ad/test/'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check)
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/test/'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check)
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/train/'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check)
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/simu_input_pkl/'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check)
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/model/200samples/'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check)
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/predict_output/train'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check)        
path_to_check=f'{workpath}/single-cell_simu/devide_chr_data/predict_output/test'
if not os.path.exists(path_to_check):
     os.makedirs(path_to_check) 
    
    for pp,train_in,test_in in  zip(range(1,len(sorted_file_list)+1),sorted_file_list,sorted_file_list):        
        simudata_train = generate_pesudo_sample(scdata_path=train_in, samplenum=100, metadata_path=metapath, d_prior=None, Dominant=True,outname=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/simu_h5ad/train/chr{pp}')#变量名错误
        simudata_test= generate_pesudo_sample(scdata_path=test_in, samplenum=100, metadata_path=metapath, d_prior=None, Dominant=True,outname=f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/simu_h5ad/test/chr{pp}')#变量名错误     
        simudata_train.obs.to_csv(f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/train/Trainingsets_groundtruth_chr{pp}.csv', index=True)
        simudata_test.obs.to_csv(f'{workpath}/single-cell_simu/devide_chr_data/simu_groundtruth/test/Testsets_groundtruth_chr{pp}.csv', index=True)      
        train_x, train_y, test_x, peakname, celltypes, samplename = \
            correct_patch(simudata_train,simudata_test,patch_size=100, datatype='counts', variance_threshold=0.99, scaler='mms')
        with open(f'{workpath}/single-cell_simu/devide_chr_data/simu_input_pkl/data_chr{pp}.pkl', 'wb') as file:
            data_to_save = (train_x, train_y, test_x, peakname, celltypes, samplename)
            pickle.dump(data_to_save, file)
        batch_size=10
        epochs=50
        input_size=train_x.shape[1]
        class_size=train_y.shape[1]
        output_dim=train_y.shape[1]
        state='train'       
        train_x_1 = train_x[:, np.newaxis, :]
        train_y_1= train_y[:, np.newaxis, :]
        save_model_name=f'{workpath}/single-cell_simu/devide_chr_data/model/200samples/chr{pp}'
        model_name=f'{workpath}/single-cell_simu/devide_chr_data/model/200samples/chr{pp}'
        model = train_model(train_x_1, train_y_1, save_model_name, batch_size=batch_size, epochs=epochs)
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
        evaluation(TestPred,pp)
        with open(f'{workpath}/single-cell_simu/devide_chr_data/process_file.txt', 'a') as file:
            file.write(f'{pp} already down\n')