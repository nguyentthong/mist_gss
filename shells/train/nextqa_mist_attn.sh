mkdir -p ./save_models/nextqa/mist_nextqa_attn/
python main_nextqa.py --checkpoint_dir=nextqa \
	--feature_dir='../mist/mist_data/feats/'  \
	--dataset=nextqa \
	--mc=5 \
	--bnum=5 \
	--epochs=300 \
	--lr=0.00004 \
	--qmax_words=30 \
	--amax_words=38 \
	--max_feats=32 \
	--batch_size=64 \
	--batch_size_val=64 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=2 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--seed=400 \
	--save_dir='./save_models/nextqa/mist_nextqa_attn/' \
	--use-attn 1