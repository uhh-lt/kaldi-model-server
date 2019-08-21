# kaldi-model-server

Kaldi-model-server is a simple Kaldi model server for online decoding with TDNN chain nnet3 models. It is written in pure Python and uses [PyKaldi](https://github.com/pykaldi/pykaldi) to interface Kaldi as a library. It is mainly meant for live decoding with real microphones and for single-user applications that need to work with realtime speech recognition locally (e.g. dictation) or an aggregation of multiple audio speech streams (e.g. decoding meeting speech). Computations happen on the device that interfaces the microphone. The [redis](https://redis.io) messaging server and an event server (event_server.py) that can send events to a web browser can also be run on different devices.

Because redis supports a [wide range of different programming languages](https://redis.io/clients), it can easily be used to interact with decoded speech output in realtime with your favourite programming language.

For demonstration purposes we added an simple demo example application that uses the Python based event server written in Flask (event_server.py) to display the recognized words in a simple HTML5 app running in a browser window:

example/ An example HTML5 application that visualizes

To start the demo run 

   python3 event_server.py

and then in a different window:

   python3 nnet3_model.py

You can browse to http://127.0.0.1:5000/ and should see words appear. Word confidences are computed after an utterance is decoded and visualized with different levels of greyness.


