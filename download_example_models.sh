mkdir models/
cd models/
mv ../kaldi_tuda_de_nnet3_chain2.yaml ./
mv ../en_160k_nnet3chain_tdnn1f_2048_sp_bi.yaml ./

# German
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/de_683k_nnet3chain_tdnn1f_2048_sp_bi_smaller_fst.tar.bz2
tar xvfj de_683k_nnet3chain_tdnn1f_2048_sp_bi_smaller_fst.tar.bz2

# English
wget http://ltdata1.informatik.uni-hamburg.de/pykaldi/en_160k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
tar xvfj en_160k_nnet3chain_tdnn1f_2048_sp_bi.tar.bz2
