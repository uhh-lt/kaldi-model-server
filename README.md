# Kaldi-model-server

Kaldi-model-server is a simple Kaldi model server for online decoding with TDNN chain nnet3 models. It is written in pure Python and uses [PyKaldi](https://github.com/pykaldi/pykaldi) to interface Kaldi as a library. It is mainly meant for live decoding with real microphones and for single-user applications that need to work with realtime speech recognition locally (e.g. dictation, voice assistants) or an aggregation of multiple audio speech streams (e.g. decoding meeting speech). Computations currently happen on the device that interfaces the microphone. The [redis](https://redis.io) messaging server and a event server that can send [server-sent event notifications](https://www.w3schools.com/html/html5_serversentevents.asp) to a web browser can also be run on different devices.

Kaldi-model-server works on Linux (preferably Ubuntu / Debian based) and Mac OS X. Because redis supports a [wide range of different programming languages](https://redis.io/clients), it can easily be used to interact with decoded speech output in realtime with your favourite programming language.

For demonstration purposes we added an simple demo example application that uses a Python based event server with [Flask](https://palletsprojects.com/p/flask/) (event_server.py) to display the recognized words in a simple HTML5 app running in a browser window.

See example/ -> An example HTML5 application that visualizes decoded speech with confidence values

To start the web demo run:

```bash
/etc/init.d/redis-server start
python3 event_server.py
```

and then in a different window:

```bash
python3 nnet3_model.py
```

You can browse to http://127.0.0.1:5000/ and should see words appear as you speak into your microphone. Word confidences are computed after an utterance is decoded and visualized with different levels of greyness.

# Installation

~~Pykaldi doesn't work yet on Ubuntu 20.04 or later.~~ Check out our new and updated instruction to quickly install PyKaldi in: http://ltdata1.informatik.uni-hamburg.de/pykaldi/README.txt

To install dependencies for PyKaldi and kaldi-model-server on Ubuntu do:

```bash
# Ubuntu Linux
sudo apt-get install portaudio19-dev redis-server autoconf automake cmake curl g++ git graphviz libatlas3-base libtool make pkg-config subversion unzip wget zlib1g-dev virtualenv python3-dev libsamplerate0
```

On a Mac:

```bash
# Mac OS X, see https://brew.sh/
brew install automake cmake git graphviz libtool pkg-config wget subversion gnu-sed portaudio openblas sox

#newer scipy versions can't use apples accelerate for BLAS anymore, this tells pip/scipy where to look for openblas instead:
export LAPACK=/opt/homebrew/opt/openblas

# This makes sed = gnu-sed
PATH="/opt/homebrew/opt/gnu-sed/libexec/gnubin:$PATH"

brew upgrade python3
pip3 install virtualenv
pip3 install virtualenvwrapper

brew install redis
brew services start redis
```

The easist way to install PyKaldi and kaldi-model-server is in a virtual environment (named pykaldi_env):

```bash
mkdir ~/projects/
cd ~/projects/
git clone https://github.com/pykaldi/pykaldi
git clone https://github.com/uhh-lt/kaldi-model-server

cd kaldi-model-server

virtualenv -p python3 pykaldi_env
source ./pykaldi_env/bin/activate
```

Install Python3 pip dependencies:

```bash
pip3 install numpy pyparsing ninja redis pyyaml pyaudio flask flask_cors bs4 samplerate scipy
```

Notes for mac: If you are getting a "portaudio.h not found" error when installing pyaudio, despite portaudio being installed through brew, then you need to tell pip where to look for the include with:
```bash
pip3 install --global-option='build_ext' --global-option="-I$(brew --prefix)/include" --global-option="-L$(brew --prefix)/lib" pyaudio
```

If scipy can't find a LAPACK library then you can set:
```bash
export LAPACK=/opt/homebrew/opt/openblas
```

Compile and install Protobuf, CLIF and KALDI dependencies (compiliation can take some time unfortunatly):

```bash
cd  ~/projects/pykaldi/tools/
./check_dependencies.sh  # checks if system dependencies are installed
./install_protobuf.sh ~/projects/kaldi-model-server/pykaldi_env/bin/python3  # installs both the Protobuf C++ library and the Python package
./install_clif.sh ~/projects/kaldi-model-server/pykaldi_env/bin/python3  # installs both the CLIF C++ library and the Python package
./install_kaldi.sh ~/projects/kaldi-model-server/pykaldi_env/bin/python3 # installs the Kaldi C++ library
```

Note, if you are compiling this on Apple Silicion and ./install_kaldi.sh gets stuck right at the beginning compiling sctk, you might need to remove -march=native from tools/kaldi/tools/Makefile, e.g. by uncommeting it in this line like this:

```bash
SCTK_CXFLAGS = -w #-march=native in 
```

Now install PyKaldi:

```bash
cd ~/projects/pykaldi
~/projects/pykaldi$ python3 setup.py install
```

You can test the install with:

```bash
~/projects/pykaldi$ python3 setup.py test
```
You need to download the model:

```bash
cd ~/projects/kaldi-model-server
./download_example_models.sh
```

Whenever you want to run nnet3_model.py you have to run source ./bin/activate once per Bash session:

```bash
cd ~/projects/kaldi-model-server
source ./bin/activate
python3 nnet3_model.py
```
