sudo bash -c "source ./bin/activate; while true; do nice -n -8 python nnet3_model.py -m 2 -c 4 -wait -t -mf 5 -hist -bs 5; done"
