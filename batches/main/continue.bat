SET HEAD=Attn
SET LR=0.00001
SET EPOCH=0
SET STEP=642471

SET CHECKPOINT=C:\Users\tux\Documents\MScProject\output\main\CONTEXT-PSEUDO-HEAD_%HEAD%-LR_%LR%\checkpoints\epoch=%EPOCH%-step=%STEP%.ckpt

echo %CHECKPOINT%

python ..\..\main.py --do train ^
                        --resume_from_checkpoint %CHECKPOINT% ^
                        --project_directory C:\Users\tux\Documents\MScProject ^
                        --gpus 1 --num_workers 0 ^
                        --shuffle_data ^
                        --log_every_n_steps 100 --flush_logs_every_n_steps 200 --progress_bar_refresh_rate 100 ^
                        --only_deconvolute ^
                        --num_sanity_val_steps -1



