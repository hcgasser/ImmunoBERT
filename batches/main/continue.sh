h=Cls
l=0.00001
d=19
e=4
s=3648186
checkpoint="/home/tux/Documents/MScProject/output/main/CONTEXT-PSEUDO-HEAD_$h-DECOY_$d-LR_$l/checkpoints/epoch=$e-step=$s.ckpt"

python ../../main.py --do train \
                        --resume_from_checkpoint $checkpoint \
                        --project_directory "/home/tux/Documents/MScProject" \
                        --gpus 1 --num_workers 0 \
                        --shuffle_data \
                        --log_every_n_steps 100 --flush_logs_every_n_steps 200 --progress_bar_refresh_rate 100 \
                        --only_deconvolute \
                        --num_sanity_val_steps -1
