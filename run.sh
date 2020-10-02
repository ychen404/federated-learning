python3 main_fed.py --dataset foursquare --iid \
	--num_channels 1 \
	--model rnn  \
	--rounds 10 \
	--gpu 0 \
	--data_name=pub_700 \
	--num_users 2  --init 0 \
	--accuracy_mode 'top1' \
	--model_mode 'attn_local_long' \
    --print_local 0