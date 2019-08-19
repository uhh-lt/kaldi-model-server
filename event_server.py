#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Benjamin Milde'

import flask
import redis
import os
import json
import bs4
#import bridge
import codecs
import datetime

from werkzeug.serving import WSGIRequestHandler

base_path = os.getcwd() + '/example/'

server_channel = 'asr'

app = flask.Flask(__name__)
app.secret_key = 'asdf'
app._static_folder = base_path
app._static_files_root_folder_path = base_path

red = redis.StrictRedis()

long_poll_timeout = 0.5
long_poll_timeout_burst = 0.08

#Send event to the event stream
def event_stream():
    print("New connection to event_stream!")
    pubsub = red.pubsub()
    pubsub.subscribe(server_channel)
    for message in pubsub.listen():
        print('New message:', message)
        yield 'data: %s\n\n' % message['data']

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

@app.route('/')
def root():
    print('root called')
#    return app.send_static_file(base_path+'index.html')
    return flask.send_from_directory(base_path, 'index.html')

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

if __name__ == '__main__':
    print(' * Starting app with base path:',base_path)
#    new_session_outfile()
    app.debug = True
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(threaded=True)
