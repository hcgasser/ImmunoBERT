# ImmunoBERT

This repository does not include the data, since this is as of the hand in date not yet publicly available.
This folder has the following root level elements:

- **batches**
  
  This includes the batch/shell scripts that were run to: 
      
  - generate the decoys
  - run the hyperparameter search and
  - to train the final model
  
- **pMHC**
    
  This is the main code library. It contains submodules for data handling (data), training (logic) and model interpretation (interpret)
    
- **notebooks**

  This folder contains the jupyter notebooks that were used to interact with the library (pMHC). In particular to conduct the interpretation
    
- **create_decoys.py**... python script which generates the decoys (called by scripts in ./batches/create_decoys)
- **main.py**... python script which trains the model (called by scripts in ./batches/hparams and ./batches/main)
    

To generate the results, the programs need to be run in the following order:
- ./notebooks/**00_prepare_data.ipynb**... reads in the raw data files and converts them into a structure that can be processed by our program 
- ./notebooks/**01_splits.ipynb**... generates the train/val/test splits
- ./batches/create_decoys/**create_decoys_%.sh**... generates the decoys for the hits
- ./notebooks/**03_check_data.ipynb**... checks that the splitting did indeed result in a useful separation of the dataset
- ./notebooks/**04_explorative.ipynb**... generates an overview of the data
- ./batches/hparams/**hparams.sh**... trains the models for the hyperparameter search
- ./notebooks/**06_valid_hparam.ipynb**... evaluates the hparam models on the validation sets
- ./batches/main/**run.sh**... trains the final model
- ./notebooks/**08_valid_main.ipynb**... evaluates the final model on the validation set during training
- ./notebooks/**09_test.ipynb**... evaluates the final model on the test set
- ./notebooks/**10_benchmarking.ipynb**... compares our model to NetMHCpan and MHCflurry
- ./notebooks/**11_motif.ipynb**... produces the motif figures
- ./notebooks/**12_LIME.ipynb**... runs the LIME analysis
- ./notebooks/**13_SHAP.ipynb**... runs the SHAP analysis



