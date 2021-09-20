for l in 0.00001 0.000001 0.0001
do
  for h in Attn Cls Avg
  do
    for d in 19 99
    do
      me=1
      if [ $d -eq 19 ]
      then
        me=5
      fi
      python ../../main.py --do train \
                          --project_directory "/home/tux/Documents/MScProject" \
                          --name hparam_search --version "CONTEXT-PSEUDO-HEAD_$h-DECOY_$d-LR_$l" \
                          --head $h \
                          --input_mode CONTEXT --mhc_rep PSEUDO --max_seq_length 80 \
                          --datasources "Edi;Atlas" --shuffle_data --decoys_per_obs $d --proportion 0.1 \
                          --batch_size 32 --learning_rate $l \
                          --gpus 1 --num_workers 0 --precision 16 \
                          --log_every_n_steps 100 --flush_logs_every_n_steps 200 --progress_bar_refresh_rate 100 \
                          --max_epochs $me --num_sanity_val_steps 0 --limit_val_batches 0 \
                          --save_every_n_steps 12912 --only_SA
    done
  done
done

