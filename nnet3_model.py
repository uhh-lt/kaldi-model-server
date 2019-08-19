#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created on Wed Aug  7 15:35:15 2019

@author: Benjamin Milde (Language Technology, Universitaet Hamburg, Germany)
"""

#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import NnetLatticeFasterOnlineRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineIvectorExtractorAdaptationState,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo,
                           OnlineNnetFeaturePipeline,
                           OnlineSilenceWeighting)
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader
from kaldi.lat.sausages import MinimumBayesRisk

from kaldi.fstext import utils as fst_utils

import yaml
import os
import pyaudio

import json
import redis
from timer import Timer

chunk_size = 1440
chunk_size = 1200
models_path = 'models/'
online_config = models_path + "kaldi_tuda_de_nnet3_chain2.online.conf"
yaml_config = "models/kaldi_tuda_de_nnet3_chain2.yaml"

red = redis.StrictRedis()

#Do most of the message passing with redis, now standard version
class ASRRedisClient():

    def __init__(self, channel='asr'):
        self.channel = channel
        self.timer_started = False
        self.timer = Timer()

    def checkTimer(self):
        if not self.timer_started:
            self.timer.start()
            self.timer_started = True

    def resetTimer(self):
        self.timer_started = False
        self.timer.start()

    def partialUtterance(self, utterance, key='none', speaker='Speaker'):
        self.checkTimer()
        data = {'handle':'partialUtterance','utterance':utterance, 'key':key, 'speaker':speaker, 'time': float(self.timer.current_secs())}
        red.publish(self.channel, json.dumps(data))

    def completeUtterance(self, utterance, confidences, key='none', speaker='Speaker'):
        self.checkTimer()
        data = {'handle':'completeUtterance','utterance':utterance,'confidences':confidences,'key':key, 'speaker':speaker, 'time': float(self.timer.current_secs())}
        red.publish(self.channel, json.dumps(data))

    def reset(self):
        data = {'handle':'reset'}
        red.publish(self.channel, json.dumps(data))
        self.resetTimer()
        r = requests.post(self.server_url+'reset', data=json.dumps(data), headers=self.request_header)
        return r.status_code

def load_model(config_file):
    # Read YAML file
    with open(config_file, 'r') as stream:
        model_yaml = yaml.safe_load(stream)

    decoder_yaml_opts = model_yaml['decoder']

    print(decoder_yaml_opts)

    feat_opts = OnlineNnetFeaturePipelineConfig()
    endpoint_opts = OnlineEndpointConfig()

    if not os.path.isfile(online_config):
        print(online_config + ' does not exists. Trying to create it from yaml file settings.')
        print('See also online_config_options.info.txt for what possible settings are.')
        with open(online_config, 'w') as online_config_file:
            online_config_file.write("--add_pitch=False\n")
            online_config_file.write("--mfcc_config=" + models_path + decoder_yaml_opts['mfcc-config'] + "\n")
            online_config_file.write("--feature_type=mfcc\n")
            online_config_file.write("--ivector_extraction_config=" + models_path + decoder_yaml_opts['ivector-extraction-config'] + '\n')
            online_config_file.write("--endpoint.silence-phones=" + decoder_yaml_opts['endpoint-silence-phones'] + '\n')
    
    po = ParseOptions("")
    feat_opts.register(po)
    endpoint_opts.register(po)
    po.read_config_file(models_path + "kaldi_tuda_de_nnet3_chain2.online.conf")
    feat_info = OnlineNnetFeaturePipelineInfo.from_config(feat_opts)
    
    # Construct recognizer
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = 13
    decoder_opts.max_active = 7000
    decodable_opts = NnetSimpleLoopedComputationOptions()
    decodable_opts.acoustic_scale = 1.0
    decodable_opts.frame_subsampling_factor = 3
    decodable_opts.frames_per_chunk = 150
    asr = NnetLatticeFasterOnlineRecognizer.from_files(
        models_path + decoder_yaml_opts["model"], models_path + decoder_yaml_opts["fst"], models_path + decoder_yaml_opts["word-syms"],
        decoder_opts=decoder_opts,
        decodable_opts=decodable_opts,
        endpoint_opts=endpoint_opts)
    
    return asr, feat_info, decodable_opts

asr, feat_info, decodable_opts = load_model(yaml_config)

def decode_chunked_partial(scp):
    ## Decode (whole utterance)
    #for key, wav in SequentialWaveReader("scp:wav.scp"):
    #    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    #    asr.set_input_pipeline(feat_pipeline)
    #    feat_pipeline.accept_waveform(wav.samp_freq, wav.data()[0])
    #    feat_pipeline.input_finished()
    #    out = asr.decode()
    #    print(key, out["text"], flush=True)

    # Decode (chunked + partial output)
    for key, wav in SequentialWaveReader("scp:wav.scp"):
        feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
        asr.set_input_pipeline(feat_pipeline)
        asr.init_decoding()
        data = wav.data()[0]
        last_chunk = False
        part = 1
        prev_num_frames_decoded = 0
        for i in range(0, len(data), chunk_size):
            if i + chunk_size >= len(data):
                last_chunk = True
            feat_pipeline.accept_waveform(wav.samp_freq, data[i:i + chunk_size])
            if last_chunk:
                feat_pipeline.input_finished()
            asr.advance_decoding()
            num_frames_decoded = asr.decoder.num_frames_decoded()
            if not last_chunk:
                if num_frames_decoded > prev_num_frames_decoded:
                    prev_num_frames_decoded = num_frames_decoded
                    out = asr.get_partial_output()
                    print(key + "-part%d" % part, out["text"], flush=True)
                    part += 1
        asr.finalize_decoding()
        out = asr.get_output()
        print(key + "-final", out["text"], flush=True)

def decode_chunked_partial_endpointing(scp, compute_confidences=True,asr_client=None):
    # Decode (chunked + partial output + endpointing
    #         + ivector adaptation + silence weighting)
    adaptation_state = OnlineIvectorExtractorAdaptationState.from_info(
        feat_info.ivector_extractor_info)
    for key, wav in SequentialWaveReader("scp:wav.scp"):
        feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
        feat_pipeline.set_adaptation_state(adaptation_state)
        asr.set_input_pipeline(feat_pipeline)
        asr.init_decoding()
        sil_weighting = OnlineSilenceWeighting(
            asr.transition_model, feat_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor)
        data = wav.data()[0]
        last_chunk = False
        utt, part = 1, 1
        prev_num_frames_decoded, offset = 0, 0
        for i in range(0, len(data), chunk_size):
            if i + chunk_size >= len(data):
                last_chunk = True
            feat_pipeline.accept_waveform(wav.samp_freq, data[i:i + chunk_size])
            if last_chunk:
                feat_pipeline.input_finished()
            if sil_weighting.active():
                sil_weighting.compute_current_traceback(asr.decoder)
                feat_pipeline.ivector_feature().update_frame_weights(
                    sil_weighting.get_delta_weights(
                        feat_pipeline.num_frames_ready()))
            asr.advance_decoding()
            num_frames_decoded = asr.decoder.num_frames_decoded()
            if not last_chunk:
                if asr.endpoint_detected():
                    asr.finalize_decoding()
                    out = asr.get_output()
                    mbr = MinimumBayesRisk(out["lattice"])
                    confd = mbr.get_one_best_confidences()
                    print(confd)
                    print(key + "-utt%d-final" % utt, out["text"], flush=True)
                    if asr_client is not None:
                        asr_client.completeUtterance(utterance=out["text"],key=key +"-utt%d-part%d" % (utt, part),confidences=confd)
                    offset += int(num_frames_decoded
                                  * decodable_opts.frame_subsampling_factor
                                  * feat_pipeline.frame_shift_in_seconds()
                                  * wav.samp_freq)
                    feat_pipeline.get_adaptation_state(adaptation_state)
                    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
                    feat_pipeline.set_adaptation_state(adaptation_state)
                    asr.set_input_pipeline(feat_pipeline)
                    asr.init_decoding()
                    sil_weighting = OnlineSilenceWeighting(
                        asr.transition_model, feat_info.silence_weighting_config,
                        decodable_opts.frame_subsampling_factor)
                    remainder = data[offset:i + chunk_size]
                    feat_pipeline.accept_waveform(wav.samp_freq, remainder)
                    utt += 1
                    part = 1
                    prev_num_frames_decoded = 0
                elif num_frames_decoded > prev_num_frames_decoded:
                    prev_num_frames_decoded = num_frames_decoded
                    out = asr.get_partial_output()
                    print(key + "-utt%d-part%d" % (utt, part),
                          out["text"], flush=True)
                    if asr_client is not None:
                        asr_client.partialUtterance(utterance=out["text"],key=key + "-utt%d-part%d" % (utt, part))
                    part += 1
        asr.finalize_decoding()
        out = asr.get_output()
        mbr = MinimumBayesRisk(out["lattice"])
        confd = mbr.get_one_best_confidences()
        print(out)
        print(key + "-utt%d-final" % utt, out["text"], flush=True)
        if asr_client is not None:
            asr_client.completeUtterance(utterance=out["text"],key=key +"-utt%d-part%d" % (utt, part),confidences=confd)

        feat_pipeline.get_adaptation_state(adaptation_state)

def print_devices(paudio):
    info = paudio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
    for i in range (0,numdevices):
        if paudio.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
            print("Input Device id ", i, " - ", paudio.get_device_info_by_host_api_device_index(0,i).get('name'))

        if paudio.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
            print("Output Device id ", i, " - ", paudio.get_device_info_by_host_api_device_index(0,i).get('name'))


def decode_chunked_partial_endpointing_mic(paudio,input_microphone_id,samp_freq=16000, compute_confidences=True):
    stream = paudio.open(format=pyaudio.paInt16, channels=1, rate=samp_freq, input=True, frames_per_buffer=1024, input_device_index = input_microphone_id)  
    adaptation_state = OnlineIvectorExtractorAdaptationState.from_info(
        feat_info.ivector_extractor_info)
    key = 'mic'
    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    feat_pipeline.set_adaptation_state(adaptation_state)
    asr.set_input_pipeline(feat_pipeline)
    asr.init_decoding()
    sil_weighting = OnlineSilenceWeighting(
        asr.transition_model, feat_info.silence_weighting_config,
        decodable_opts.frame_subsampling_factor)
    #data = wav.data()[0]
    last_chunk = False
    utt, part = 1, 1
    prev_num_frames_decoded, offset = 0, 0
    while not last_chunk:
        #if i + chunk_size >= len(data):
        #    last_chunk = True
        block = stream.read(chunk_size)
        feat_pipeline.accept_waveform(samp_freq, block)
        if last_chunk:
            feat_pipeline.input_finished()
        if sil_weighting.active():
            sil_weighting.compute_current_traceback(asr.decoder)
            feat_pipeline.ivector_feature().update_frame_weights(
                sil_weighting.get_delta_weights(
                    feat_pipeline.num_frames_ready()))
        asr.advance_decoding()
        num_frames_decoded = asr.decoder.num_frames_decoded()
        if not last_chunk:
            if asr.endpoint_detected():
                asr.finalize_decoding()
                out = asr.get_output()
                print(key + "-utt%d-final" % utt, out["text"], flush=True)
                offset += int(num_frames_decoded
                              * decodable_opts.frame_subsampling_factor
                              * feat_pipeline.frame_shift_in_seconds()
                              * wav.samp_freq)
                feat_pipeline.get_adaptation_state(adaptation_state)
                feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
                feat_pipeline.set_adaptation_state(adaptation_state)
                asr.set_input_pipeline(feat_pipeline)
                asr.init_decoding()
                sil_weighting = OnlineSilenceWeighting(
                    asr.transition_model, feat_info.silence_weighting_config,
                    decodable_opts.frame_subsampling_factor)
                remainder = data[offset:i + chunk_size]
                feat_pipeline.accept_waveform(wav.samp_freq, remainder)
                utt += 1
                part = 1
                prev_num_frames_decoded = 0
            elif num_frames_decoded > prev_num_frames_decoded:
                prev_num_frames_decoded = num_frames_decoded
                out = asr.get_partial_output()
                mbr = MinimumBayesRisk(asr.get_lattice())
                confd = mbr.get_one_best_confidences()
                print(confd)
                print(key + "-utt%d-part%d" % (utt, part),
                      out["text"], flush=True)
                part += 1
    asr.finalize_decoding()
    out = asr.get_output()
    print(key + "-utt%d-final" % utt, out["text"], flush=True)
    feat_pipeline.get_adaptation_state(adaptation_state)
    
#decode_chunked_partial("scp:wav.scp")
asr_client = ASRRedisClient(channel='asr')
decode_chunked_partial_endpointing("scp:wav.scp", asr_client=asr_client)

#paudio = pyaudio.PyAudio()
#print_devices(paudio)
#decode_chunked_partial_endpointing_mic(paudio,input_microphone_id=1)
