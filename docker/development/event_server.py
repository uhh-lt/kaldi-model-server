#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Benjamin Milde'

import flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import redis
import os
import json
import bs4
#import bridge
import codecs
import datetime
import base64
from flask import render_template
from werkzeug.serving import WSGIRequestHandler

base_path = os.getcwd() + '/example/'

server_channel = 'asr'
decode_control_channel = 'asr_control'
audio_data_channel = 'asr_audio'
async_mode = None

app = flask.Flask(__name__)
app.secret_key = 'asdf'
app._static_folder = base_path
app._static_files_root_folder_path = base_path

socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")

CORS(app)

red = redis.StrictRedis()

long_poll_timeout = 0.5
long_poll_timeout_burst = 0.08

#Send event to the event stream
def event_stream():
    print("New connection to event_stream!")
    pubsub = red.pubsub()
    pubsub.subscribe(server_channel)
    yield b'hello'
    for message in pubsub.listen():
        if not message['type'] == 'subscribe':
            #print('New message:', message)
            #print(type(message['data']))
            yield b'data: %s\n\n' % message['data']

@app.route('/reset')
def reset():
    red.publish(decode_control_channel, 'reset')
    print("reset called")
    return 'OK'

@app.route('/stop')
def stop():
    red.publish(decode_control_channel, 'stop')
    print("stop called")
    return 'OK'

@app.route('/start')
def start():
    red.publish(decode_control_channel, 'start')
    print("start called")
    return 'OK'

@app.route('/shutdown')
def shutdown():
    red.publish(decode_control_channel, 'shutdown')
    print("shutdown called")
    return 'OK'

@app.route('/status')
def status():
    red.publish(decode_control_channel, 'status')
    print("status called")
    return 'OK'

@app.route('/reset_timer')
def reset_timer():
    red.publish(decode_control_channel, 'reset_timer')
    print("reset time called")
    return 'OK'

#Event stream end point for the browser, connection is left open. Must be used with threaded Flask.
@app.route('/stream')
def stream():
    return flask.Response(event_stream(), mimetype="text/event-stream")

#Traditional long polling. This is the fall back, if a browser does not support server side events. TODO: test and handle disconnects
@app.route('/stream_poll')
def poll():
    pubsub = red.pubsub()
    pubsub.subscribe(server_channel)
    message = pubsub.get_message(timeout=long_poll_timeout)
    while(message != None):
        yield message
        message = pubsub.get_message(timeout=long_poll_timeout_burst)

#These should ideally be served with a real web server, but for developping purposes, serving static files with Flask is also ok:
#START static files

#@app.route('/')
#def root():
#    print('root called')
#    return app.send_static_file(base_path+'index.html')
#    return flask.send_from_directory(base_path, 'index.html')

@app.route('/css/<path:path>')
def send_css(path):
    return flask.send_from_directory(base_path+'css', path)

@app.route('/js/<path:path>')
def send_js(path):
    return flask.send_from_directory(base_path+'js', path)
    
@app.route('/pics/<path:path>')
def send_pics(path):
    return flask.send_from_directory(base_path+'pics', path)
    
@app.route('/fonts/<path:path>')
def send_fonts(path):
    return flask.send_from_directory(base_path+'fonts', path)

# END static files

# START socketio

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.on('connect', namespace='/test')
def connect():
    print("new connection")


@socketio.on('recording_started', namespace='/test')
def recording_started():
    print("recording started!")


@socketio.on('data_available', namespace='/test')
def data_available(message):
    bytedata = base64.b64decode(message['base64'])
    pcm_stream = bytedata[44:]
    red.publish(audio_data_channel, pcm_stream)

@socketio.on('recording_stopped', namespace='/test')
def recording_stopped(message):
    print(f"recording stopped!")


# END socketio

if __name__ == '__main__':
    print(' * Starting app with base path:',base_path)
#    new_session_outfile()
    app.debug = True
    app
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    # app.run(host='0.0.0.0', threaded=True, port=5000)
    socketio.run(app, host='0.0.0.0', port=5000)
