python main_egoschema.py --checkpoint_dir=egoschema \
	--feature_dir='../mist/mist_data/feats/'  \
	--dataset=egoschema \
	--mc=5 \
	--bnum=5 \
	--epochs=30 \
	--lr=0.00004 \
	--qmax_words=30 \
	--amax_words=38 \
	--max_feats=32 \
	--batch_size=16 \
	--batch_size_val=64 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=2 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--seed=400 \
	--use-gss=1 \
	--test=1 \
	--save_dir='./save_models/egoschema/mist_egoschema_gss/' \
	--pretrain_path ./save_models/egoschema/mist_egoschema_gss/best_model.pth
