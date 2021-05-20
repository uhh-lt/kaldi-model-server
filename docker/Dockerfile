# docker build -t uhhlt/kamose:latest .

# docker run -p 5000:5000 --name kamose --rm --device /dev/snd:/dev/snd -ti -v ${PWD}/models:/projects/kaldi-model-server/models --entrypoint bash uhhlt/kamose:latest
# sh entrypoint_event_server.sh

# docker exec -ti kamose bash
# asr -m 11 -c 1 -t -mf 5 -r 16000 -cs 8192 -bs 2 -a linear

FROM ubuntu:18.04

RUN apt-get update \
 && apt-get install -y portaudio19-dev redis-server autoconf automake cmake curl g++ git graphviz libatlas3-base libtool make pkg-config subversion unzip wget zlib1g-dev virtualenv python3-dev libsamplerate0 sox python3.8 python2.7 python3.8-dev software-properties-common gfortran alsa-utils

RUN mkdir projects

WORKDIR /projects

RUN git clone https://github.com/uhh-lt/kaldi-model-server.git

WORKDIR /projects/kaldi-model-server

RUN wget https://ltdata1.informatik.uni-hamburg.de/pykaldi/pykaldi-0.2.0-cp38-cp38-linux_x86_64.whl \
 && wget https://ltdata1.informatik.uni-hamburg.de/pykaldi/install_kaldi_for_pykaldi_0.2.0.sh \
 && wget https://ltdata1.informatik.uni-hamburg.de/pykaldi/install_kaldi_intel.sh \
 && wget https://ltdata1.informatik.uni-hamburg.de/pykaldi/path.sh \
 && wget https://ltdata1.informatik.uni-hamburg.de/pykaldi/install_mkl.sh \
 && chmod guo+x *.sh

RUN virtualenv -p /usr/bin/python3.8 pykaldi_env \
 && . pykaldi_env/bin/activate \
 && pip3 install pykaldi-0.2.0-cp38-cp38-linux_x86_64.whl pyparsing ninja redis pyyaml pyaudio flask flask_cors bs4 samplerate scipy

RUN ./install_mkl.sh \
 && ./install_kaldi_for_pykaldi_0.2.0.sh

RUN rm pykaldi-0.2.0-cp38-cp38-linux_x86_64.whl \
 && find . -name "*.o" -type f -delete

VOLUME ["/projects/kaldi-model-server/models"]

RUN echo "#!/bin/bash \n\
  set -e \n\
  cd /projects/kaldi-model-server \n\
  # download example model if it doesn't exist
  [ ! -d /projects/kaldi-model-server/models/en_160k_nnet3chain_tdnn1f_2048_sp_bi ] && ( \
    echo 'Example model does not exist!' \
      && sh download_example_models.sh \
    ) || echo 'Example model already exists.' \n\
  cd /projects/kaldi-model-server/ \n\
  # start redis \n\
  redis-server --daemonize yes \n\
  # run event server in background \n\
  . pykaldi_env/bin/activate \n\
  python event_server.py \n\
" > entrypoint_event_server.sh

RUN echo "#!/bin/bash \n\
  set -e \n\
  cd /projects/kaldi-model-server \n\
  . pykaldi_env/bin/activate \n\
  . path.sh \n\
  python nnet3_model.py \$* \n\
" > /usr/local/bin/asr && chmod a+rx /usr/local/bin/asr

EXPOSE 5000

CMD ["sh", "/projects/kaldi-model-server/entrypoint_event_server.sh"]
