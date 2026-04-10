python main_3dpw.py \
    --data_dir "./dataset/3dpw/" \
    --kernel_size 10 \
    --dct_n 35 \
    --input_n 10 \
    --output_n 25 \
    --batch_size 16 \
    --skip_rate 1 \
    --test_batch_size 32 \
    --in_features 69 \
    --cuda_idx cuda:1 \
    --d_model 16 \
    --lr_now 0.005 \
    --epoch 50 \
    --test_sample_num -1 \
    --num_stage 6 \
    --drop_out 0.4 \
    --n_tcnn_layers 4 \
    --tcnn_dropout 0.3
    