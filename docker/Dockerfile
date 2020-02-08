###
#
# Kaldi Model Server
#
# NOTE: you might need to build the image with adjusted memory settings, e.g.:
#
#   $> docker build -f Dockerfile --tag uhh-lt/kaldi-model-server:latest --memory 6g .
#
###
FROM ubuntu:19.10

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y wget bzip2 ca-certificates curl git \
  && apt-get install -y redis-server portaudio19-dev libportaudio2 redis-server autoconf automake cmake curl g++ git graphviz libatlas3-base libtool make pkg-config subversion unzip wget zlib1g-dev sox z3 libz3-dev llvm ocaml ocaml-libs libctypes-ocaml-dev libcothreads-ocaml-dev clang gcc g++ gcc-7 g++-7 libsamplerate0 alsa-base alsa-utils \
  && apt-get autoremove \
  && apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN conda update -y -n base -c defaults conda

RUN conda install -y python=3.6

RUN pip install samplerate flask_cors bs4 doxypy numpy pyparsing ninja redis pyyaml flask pyaudio scipy

RUN mkdir -p /projects

WORKDIR /projects

RUN git clone https://github.com/pykaldi/pykaldi && cd pykaldi && git checkout 249963c2cb57d92c0027bb958280dcf8e408462a && cd ..

RUN git clone https://github.com/uhh-lt/kaldi-model-server && cd kaldi-model-server && git checkout 08b11a1955b3307d06e3a443e2cdbb88916f1229 && cd ..

RUN cd /projects/pykaldi/tools/ \
  && ./check_dependencies.sh \
  && ./install_protobuf.sh

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7

RUN cd /projects/pykaldi/tools/ \
  && ./install_clif.sh

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 70 --slave /usr/bin/g++ g++ /usr/bin/g++-9

RUN cd /projects/pykaldi/tools/ \
  && ./install_kaldi.sh

RUN cd /projects/pykaldi/ \
  && MAKE_NUM_JOBS=2 python setup.py install

WORKDIR /projects/kaldi-model-server

VOLUME ["/projects/kaldi-model-server/models"]

RUN echo "#!/bin/bash \n\
  set -e \n\
  cd /projects/kaldi-model-server \n\
  # download example model if it doesn't exist
  [ ! -d /projects/kaldi-model-server/models/de_400k_nnet3chain_tdnn1f_2048_sp_bi ] && ( \
    echo 'Example model does not exist!' \
      && sh download_example_models.sh \
      && sed -i 's/=de_400k_nnet3chain_tdnn1f_2048_sp_bi\//=models\/de_400k_nnet3chain_tdnn1f_2048_sp_bi\//g' models/de_400k_nnet3chain_tdnn1f_2048_sp_bi/ivector_extractor/ivector_extractor.conf \
    ) || echo 'Example model already exists.' \n\
  # start redis \n\
  /etc/init.d/redis-server start \n\
  # run event server in background \n\
  python event_server.py \n\
" > entrypoint_event_server.sh

RUN echo "#!/bin/bash \n\
  set -e \n\
  cd /projects/kaldi-model-server \n\
  python nnet3_model.py \$* \n\
" > /usr/local/bin/asr && chmod a+rx /usr/local/bin/asr

EXPOSE 5000

CMD ["sh", "/projects/kaldi-model-server/entrypoint_event_server.sh"]

###############
#
###############
