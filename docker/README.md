## Server Addresses

  - Meeting Minute Bot UI
    http://localhost:8080

  - check ASR output
    http://localhost:5000

  - gracefully shut down ASR
    http://localhost:5000/shutdown

## Example commands:

```shell
# start up docker
docker-compose up -d

docker run --rm --device /dev/snd:/dev/snd -ti -v ${PWD}/models:/projects/kaldi-model-server/models --entrypoint bash uhhlt/kaldi-model-server:latest

# run bash within docker
docker exec -ti kamose bash

  # commands within kamose docker

  # run redis server and event-server
  $/projects/kaldi-model-server/> sh entrypoint.sh

  # run asr commands
  $> asr -l
  $> asr -m 6 -c 1 -t -mf 5 -r 48000 -cs 8192 -bs 5 -w --wait
  $> asr -m 5 -c 1 -t -mf 5 -r 48000 -cs 8192 -bs 5 -a linear -w --wait
  $> asr -m 7 -c 1 -t -mf 5 -r 48000 --yaml-config models/kaldi_tuda_de_nnet3_chain2.yaml

  # 'asr' command is an alias for
  $/projects/kaldi-model-server/> python nnet3_model.py

  # record from commandline (for testing purposes)
  $> arecord -l
  $> arecord -L
  $> arecord -f S32_LE -r 32000 -D plughw:2,0 -d 5 -c 2 /projects/kaldi-model-server/models/arecord-test.wav

# run commands from outside of docker with prefix 'docker exec kamose'
docker exec kamose asr --help

docker exec kamose asr -l

docker exec kamose asr -m 6 -c 1 -t -mf 5 -r 48000 -cs 8192 -bs 5 -w --wait

docker exec kamose asr -m 7 -c 1 -t -mf 5 -r 48000 --wait --yaml-config models/kaldi_tuda_de_nnet3_chain2.yaml --online-config models/kaldi_tuda_de_nnet3_chain2.online.conf

docker exec kamose asr -m 7 -c 1 -t -mf 5 -r 48000 --wait --yaml-config models/en_160k_nnet3chain_tdnn1f_2048_sp_bi.yaml --online-config models/en_160k_nnet3chain_tdnn1f_2048_sp_bi.online.conf

```

## Demo commands

- open 2 terminals
- connect with kamose
```
docker exec -ti kamose bash
```

terminal 1: start asr
```
asr -m 10 -e -c 1 -t -mf 5 -r 16000 --yaml-config models/en_160k_nnet3chain_tdnn1f_2048_sp_bi.yaml --online-config models/en_160k_nnet3chain_tdnn1f_2048_sp_bi.online.conf
```

terminal2:
- copy *.wav file into models directory so that kamose can access it
- read wav and stream it via redis
```
. pykaldi_env/bin/activate
python3 publish_wav.py <audiofile>
```
