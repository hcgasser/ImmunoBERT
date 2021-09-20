setlocal EnableDelayedExpansion

SET CON=CONTEXT
SET MHC=PSEUDO
SET LRs=0.00001 0.000001 0.0001
SET DECOYs=19 99
SET HEADs=Attn Cls Avg


echo %LRs%

(FOR %%l in (%LRs%) do (
    (FOR %%h in (%HEADs%) do (
        (FOR %%d in (%DECOYs%) do (
            SET ME=1
            if %%d==19 SET ME=5
            python ..\..\main.py --do train ^
                        --project_directory C:\Users\tux\Documents\MScProject ^
                        --name hparam_search --version %CON%-%MHC%-HEAD_%%h-DECOY_%%d-LR_%%l ^
                        --head %%h ^
                        --input_mode CONTEXT --mhc_rep PSEUDO --max_seq_length 80 ^
                        --datasources Edi;Atlas --shuffle_data --decoys_per_obs %%d --proportion 0.1 ^
                        --batch_size 32 --learning_rate %%l ^
                        --gpus 1 --num_workers 0 --precision 16 ^
                        --log_every_n_steps 100 --flush_logs_every_n_steps 200 --progress_bar_refresh_rate 100 ^
                        --max_epochs !ME! --num_sanity_val_steps 0 --limit_val_batches 0 ^
                        --save_every_n_steps 10000 --only_SA
        ))
    ))
))


