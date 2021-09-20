h=Cls
l=0.00001
d=19
python ../../main.py --do train \
                        --project_directory "/home/tux/Documents/MScProject" \
                        --name main --version "CONTEXT-PSEUDO-HEAD_$h-DECOY_$d-LR_$l" \
                        --head $h \
                        --input_mode CONTEXT --mhc_rep PSEUDO --max_seq_length 80 \
                        --datasources "Edi;Atlas" --shuffle_data --decoys_per_obs $d --proportion 1.0 \
                        --batch_size 32 --learning_rate $l \
                        --gpus 1 --num_workers 0 --precision 16 \
                        --log_every_n_steps 100 --flush_logs_every_n_steps 200 --progress_bar_refresh_rate 100 \
                        --only_deconvolute \
                        --num_sanity_val_steps 0

