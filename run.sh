python3 main_fed.py --dataset foursquare --iid \
	--num_channels 1 \
	--model rnn  \
	--rounds 3 \
	--gpu 0 \
	--data_name=pub_700 \
	--num_users 5 --init 1 \
	--accuracy_mode='top1'
