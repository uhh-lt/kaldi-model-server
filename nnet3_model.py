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

from kaldi.matrix import Matrix, Vector

from kaldi.fstext import utils as fst_utils

import yaml
import os
import pyaudio
import time

import json
import redis
from timer import Timer

import numpy as np
import samplerate

import argparse

import scipy.io.wavfile as wavefile

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

red = redis.StrictRedis()
decode_control_channel = 'asr_control'

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
        data = {'handle': 'partialUtterance', 'utterance': utterance, 'key': key,
                'speaker': speaker, 'time': float(self.timer.current_secs())}
        red.publish(self.channel, json.dumps(data))

    def completeUtterance(self, utterance, confidences, key='none', speaker='Speaker'):
        self.checkTimer()
        data = {'handle': 'completeUtterance', 'utterance': utterance, 'confidences': confidences,
                'key': key, 'speaker': speaker, 'time': float(self.timer.current_secs())}
        red.publish(self.channel, json.dumps(data))

    def asr_loading(self, speaker):
        self.checkTimer()
        data = {'handle': 'asr_loading', 'time': float(self.timer.current_secs()), 'speaker': speaker}
        red.publish(self.channel, json.dumps(data))

    def asr_ready(self, speaker):
        self.checkTimer()
        data = {'handle': 'asr_ready', 'time': float(self.timer.current_secs()), 'speaker': speaker}
        red.publish(self.channel, json.dumps(data))

    def sendstatus(self, isDecoding, shutdown=False):
        self.checkTimer()
        data = {'handle': 'status', 'time': float(self.timer.current_secs()), 'isDecoding': isDecoding, 'shutdown': shutdown}
        red.publish(self.channel, json.dumps(data))

def load_model(config_file, online_config, models_path='models/', beam_size=10, frames_per_chunk=50):
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
    decoder_opts.beam = beam_size
    decoder_opts.max_active = 7000
    decodable_opts = NnetSimpleLoopedComputationOptions()
    decodable_opts.acoustic_scale = 1.0
    decodable_opts.frame_subsampling_factor = 3
    decodable_opts.frames_per_chunk = frames_per_chunk
    asr = NnetLatticeFasterOnlineRecognizer.from_files(
        models_path + decoder_yaml_opts["model"], models_path + decoder_yaml_opts["fst"],
        models_path + decoder_yaml_opts["word-syms"],
        decoder_opts=decoder_opts,
        decodable_opts=decodable_opts,
        endpoint_opts=endpoint_opts)

    return asr, feat_info, decodable_opts

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

def decode_chunked_partial_endpointing(asr, feat_info, decodable_opts, scp, chunk_size=1024,
                                       compute_confidences=True, asr_client=None, speaker="Speaker"):
    # Decode (chunked + partial output + endpointing
    #         + ivector adaptation + silence weighting)
    adaptation_state = OnlineIvectorExtractorAdaptationState.from_info(
        feat_info.ivector_extractor_info)
    for key, wav in SequentialWaveReader(scp):
        feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
        feat_pipeline.set_adaptation_state(adaptation_state)
        asr.set_input_pipeline(feat_pipeline)
        asr.init_decoding()
        sil_weighting = OnlineSilenceWeighting(
            asr.transition_model, feat_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor)
        data = wav.data()[0]
        print("type(data):", type(data))
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
                        asr_client.completeUtterance(utterance=out["text"], key=key +
                                                        "-utt%d-part%d" % (utt, part), confidences=confd)
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


def decode_chunked_partial_endpointing_mic(asr, feat_info, decodable_opts, paudio, input_microphone_id, channels=1,
                                           samp_freq=16000, record_samplerate=16000, chunk_size=1024, wait_for_start_command=False, compute_confidences=True, asr_client=None, speaker_str="Speaker",
                                           resample_algorithm="sinc_best", save_debug_wav=False, use_threads=False, minimum_num_frames_decoded_per_speaker=5):
    p = red.pubsub()
    p.subscribe(decode_control_channel)

    need_resample = False
    if record_samplerate != samp_freq:
        print("Activating resampler since record and decode samplerate are different:", record_samplerate, "->", samp_freq)
        resampler = samplerate.Resampler(resample_algorithm, channels=channels)
        need_resample = True
        ratio = samp_freq / record_samplerate
        print("Resample ratio:", ratio)

    print("Constructing decoding pipeline")
    adaptation_state = OnlineIvectorExtractorAdaptationState.from_info(feat_info.ivector_extractor_info)
    key = 'mic' + str(input_microphone_id)
    feat_pipeline, sil_weighting = initNnetFeatPipeline(adaptation_state, asr, decodable_opts, feat_info)
    print("Done")

    speaker = speaker_str.replace("#c#", "0")
    last_chunk = False
    utt, part = 1, 1
    prev_num_frames_decoded, offset_complete = 0, 0
    chunks_decoded = 0
    num_chunks = 0
    blocks = []
    rawblocks = []

    print("Open microphone stream with id" + str(input_microphone_id) + "...")
    stream = paudio.open(format=pyaudio.paInt16, channels=channels, rate=record_samplerate, input=True,
                         frames_per_buffer=chunk_size, input_device_index=input_microphone_id)
    print("Done!")

    do_decode = not wait_for_start_command
    need_finalize = False
    block, previous_block = None, None
    decode_future = None

    asr_client.asr_ready(speaker=speaker)

    with ThreadPoolExecutor(max_workers=1) as executor:
        while not last_chunk:
            # check if there is a message from the redis server first
            msg = p.get_message()

            # We check of there are externally send control commands
            if msg is not None:
                print('msg:', msg)
                if msg['data'] == b"start":
                    print('Start command received!')
                    do_decode = True
                    asr_client.sendstatus(isDecoding=do_decode)

                elif msg['data'] == b"stop":
                    print('Stop command received!')
                    if do_decode:
                        need_finalize = True
                    do_decode = False
                    asr_client.sendstatus(isDecoding=do_decode)

                elif msg['data'] == b"shutdown":
                    print('Shutdown command received!')
                    last_chunk = True

                elif msg['data'] == b"status":
                    print('Status command received!')
                    asr_client.sendstatus(isDecoding=do_decode)

                elif msg['data'] == b"reset_time":
                    print('Reset time command received!')
                    asr_client.resetTimer()

            # We always consume from the microphone stream, even if we do not decode
            block_raw = stream.read(chunk_size, exception_on_overflow=False)
            npblock = np.frombuffer(block_raw, dtype=np.int16)

            if need_resample:
                block = resampler.process(np.array(npblock, copy=True), ratio)
                block = np.array(block, dtype=np.int16)
            else:
                block = npblock

            if save_debug_wav:
                blocks.append(block)
                rawblocks.append(npblock)

            # block on the result of the decode if one is pending
            if use_threads and do_decode and block is not None and decode_future is not None:
                need_endpoint_finalize, prev_num_frames_decoded, part, utt = decode_future.result()

                # check if we need to finalize, disallow endpoint without a single decoded frame
                if need_endpoint_finalize and prev_num_frames_decoded > 0:
                    need_finalize = True
                    resend_previous_waveform = True

            # finalize the decoding here. We might need to finalize if we switch from do_decode=True to do_decode=False
            # that is why this is outside of that block
            if need_finalize and block is not None:
                out, confd = finalize_decode(asr, asr_client, key,
                                             part, speaker, utt)
                feat_pipeline, sil_weighting = reinitialize_asr(adaptation_state, asr, feat_info, feat_pipeline)
                utt += 1
                part = 1

                if resend_previous_waveform and previous_block is not None:
                    # offset_complete += int(num_frames_decoded
                    #              * decodable_opts.frame_subsampling_factor
                    #              * feat_pipeline.frame_shift_in_seconds()
                    #              * samp_freq)
                    # print("offset_complete:", offset_complete)
                    # offset = offset_complete - (chunks_read*chunk_size)
                    # print("offset:", offset_complete)
                    # remainder = block[offset:]

                    # we simplify the above and always resend the last block for the new utterance
                    feat_pipeline.accept_waveform(samp_freq, Vector(previous_block))
                    resend_previous_waveform = False

                need_finalize = False

            if channels > 1:
                block = np.reshape(block, (-1, channels))

                # select loudest channel
                volume_norms = []
                for i in range(channels):
                    volume_norms.append(np.linalg.norm(block[:, i] / 65536.0) * 10.0)
                    #print("|" * int(volume_norm))

                #print(volume_norms)

                volume_norm = max(volume_norms)
                max_channel = volume_norms.index(volume_norm)
                block = block[:, max_channel]

                new_speaker = speaker_str.replace("#c#", str(max_channel))

                if new_speaker != speaker and prev_num_frames_decoded >= minimum_num_frames_decoded_per_speaker:
                    print("Speaker change! Number of frames decoded for previous speaker:", str(prev_num_frames_decoded))

                    speaker = new_speaker

                    need_finalize = True
                    resend_previous_waveform = True
            else:
                volume_norm = np.linalg.norm(block / 65536.0) * 10.0


            num_chunks += 1

            # send status beacon periodically
            if num_chunks % 50 == 0:
                asr_client.sendstatus(isDecoding=do_decode)

            if do_decode:
                if not use_threads:
                    need_endpoint_finalize, prev_num_frames_decoded, part, utt = advance_mic_decoding(adaptation_state, asr, asr_client, block, chunks_decoded, feat_info, feat_pipeline, key, last_chunk,
                                                                part, prev_num_frames_decoded, samp_freq, sil_weighting, speaker, utt)
                    # check if we need to finalize, disallow endpoint without a single decoded frame
                    if need_endpoint_finalize and prev_num_frames_decoded > 0:
                        need_finalize = True
                        resend_previous_waveform = True
                else:
                    #submit a non blocking computation
                    decode_future = executor.submit(advance_mic_decoding, adaptation_state, asr, asr_client, block, chunks_decoded, feat_info, feat_pipeline, key, last_chunk,
                                                                part, prev_num_frames_decoded, samp_freq, sil_weighting, speaker, utt)
            else:
                time.sleep(0.001)

            previous_block = block

    if save_debug_wav:
        print("Saving debug output...")
        wavefile.write("debug.wav", samp_freq, np.concatenate(blocks, axis=None))
        wavefile.write("debugraw.wav", record_samplerate, np.concatenate(rawblocks, axis=None))
    else:
        print("Not writing debug wav output since --save_debug_wav is not set.")

    print("Shutdown: finalizing ASR output...")
    asr.finalize_decoding()
    out = asr.get_output()
    mbr = MinimumBayesRisk(out["lattice"])
    confd = mbr.get_one_best_confidences()
    print(out)
    print(key + "-utt%d-final" % utt, out["text"], flush=True)
    if asr_client is not None:
        asr_client.completeUtterance(utterance=out["text"], key=key + "-utt%d-part%d" % (utt, part), confidences=confd, speaker=speaker)
        asr_client.sendstatus(isDecoding=False,shutdown=True)
    print("Done, will exit now.")


def advance_mic_decoding(adaptation_state, asr, asr_client, block, chunks_decoded, feat_info, feat_pipeline, key, last_chunk, part, prev_num_frames_decoded,
                         samp_freq, sil_weighting, speaker, utt):
    need_endpoint_finalize = False
    chunks_decoded += 1
    feat_pipeline.accept_waveform(samp_freq, Vector(block))
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
            if num_frames_decoded > 0:
                need_endpoint_finalize = True
                prev_num_frames_decoded = 0
        elif num_frames_decoded > prev_num_frames_decoded:
            prev_num_frames_decoded = num_frames_decoded
            out = asr.get_partial_output()
            print(key + "-utt%d-part%d" % (utt, part),
                  out["text"], flush=True)
            if asr_client is not None:
                asr_client.partialUtterance(utterance=out["text"], key=key + "-utt%d-part%d" % (utt, part), speaker=speaker)
            part += 1
    return need_endpoint_finalize, prev_num_frames_decoded, part, utt


def initNnetFeatPipeline(adaptation_state, asr, decodable_opts, feat_info):
    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    feat_pipeline.set_adaptation_state(adaptation_state)
    asr.set_input_pipeline(feat_pipeline)
    asr.init_decoding()
    sil_weighting = OnlineSilenceWeighting(
        asr.transition_model, feat_info.silence_weighting_config,
        decodable_opts.frame_subsampling_factor)
    return feat_pipeline, sil_weighting


def finalize_decode(asr, asr_client, key, part, speaker, utt):
    asr.finalize_decoding()
    out = asr.get_output()
    mbr = MinimumBayesRisk(out["lattice"])
    confd = mbr.get_one_best_confidences()
    print(confd)
    print(key + "-utt%d-final" % utt, out["text"], flush=True)
    if asr_client is not None:
        asr_client.completeUtterance(utterance=out["text"], key=key + "-utt%d-part%d" % (utt, part), confidences=confd, speaker=speaker)

    return out, confd


def reinitialize_asr(adaptation_state, asr, feat_info, feat_pipeline):
    feat_pipeline.get_adaptation_state(adaptation_state)
    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    feat_pipeline.set_adaptation_state(adaptation_state)
    asr.set_input_pipeline(feat_pipeline)
    asr.init_decoding()
    sil_weighting = OnlineSilenceWeighting(
        asr.transition_model, feat_info.silence_weighting_config,
        decodable_opts.frame_subsampling_factor)
    return feat_pipeline, sil_weighting


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Starts a Kaldi nnet3 decoder')
    parser.add_argument('-i', '--input', dest='input', help='Input scp, simulate online decoding from wav files', type=str, default='scp:wav.scp')
    parser.add_argument('-l', '--list-audio-interfaces', dest='list_audio_interfaces', help='List all available audio interfaces on this system', action='store_true', default=False)

    parser.add_argument('-m', '--mic-id', dest='micid', help='Microphone ID, if not set to -1, do online decoding directly from the microphone.', type=int, default='-1')
    parser.add_argument('-c', '--channels', dest='channels', help='Number of channels to record from the microphone, ', type=int, default=1)

    parser.add_argument('-wait', '--wait-for-start-command', dest='wait_for_start_command', help='Do not start decoding directly, wait for a start command from the redis control channel.',
                        action='store_true', default=False)

    parser.add_argument('-s', '--speaker-name', dest='speaker_name', help='Name of the speaker, use #c# for channel', type=str, default='speaker#c#')
    parser.add_argument('-cs', '--chunk_size', dest='chunk_size', help='Default buffer size for the microphone buffer.', type=int, default=1024)

    parser.add_argument('-bs', '--beam_size', dest='beam_size', help='Beam size of the decoding beam. Defaults to 10.', type=int, default=10)

    parser.add_argument('-fpc', '--frames_per_chunk', dest='frames_per_chunk', help='Frames per (decoding) chunk. This will also have an effect on latency.', type=int, default=30)

    parser.add_argument('-red', '--redis-channel', dest='redis_channel', help='Name of the channel (for redis-server)', type=str, default='asr')
    parser.add_argument('-y', '--yaml-config', dest='yaml_config', help='Path to the yaml model config', type=str, default='models/kaldi_tuda_de_nnet3_chain2.yaml')
    parser.add_argument('-o', '--online-config', dest='online_config', help='Path to the Kaldi online config. If not available, will try to read the parameters from the yaml'
                                                                            ' file and convert it to the Kaldi online config format (See online_config_options.info.txt for details)',
                                                                            type=str, default='models/kaldi_tuda_de_nnet3_chain2.online.conf')
    parser.add_argument('-r', '--record-samplerate', dest='record_samplerate', help='The recording samplingrate if a microphone is used', type=int, default=16000)
    parser.add_argument('-d', '--decode-samplerate', dest='decode_samplerate', help='Decode samplerate, if not the same as the microphone samplerate '
                                                                                    'then the signal is automatically resampled', type=int, default=16000)

    parser.add_argument('-a', '--resample_algorithm', dest='resample_algorithm', help="One of the following: linear, sinc_best, sinc_fastest,"
                                                                                      " sinc_medium, zero_order_hold (default: sinc_best)",
                                                                                      type=str, default="sinc_best")

    parser.add_argument('-t', '--use-threads', dest='use_threads', help='Use a thread worker for realtime decoding',
                        action='store_true', default=False)

    parser.add_argument('-w', '--save_debug_wav', dest='save_debug_wav', help='This will write out a debug.wav (resampled)'
                                                                              ' and debugraw.wav (original) after decoding,'
                                                                              ' so that the recording quality can be analysed', action='store_true', default=False)

    args = parser.parse_args()

    if args.list_audio_interfaces:
        print("Listing audio interfaces...")
        paudio = pyaudio.PyAudio()
        print_devices(paudio)
    else:
        asr_client = ASRRedisClient(channel=args.redis_channel)
        asr_client.asr_loading(speaker=args.speaker_name)
        asr, feat_info, decodable_opts = load_model(args.yaml_config, args.online_config, beam_size=args.beam_size, frames_per_chunk=args.frames_per_chunk)
        if args.micid == -1:
            print("Reading from wav scp:", args.input)
            asr_client.asr_ready(speaker=args.speaker_name)
            decode_chunked_partial_endpointing(asr, feat_info, decodable_opts, args.input,
                                               asr_client=asr_client, speaker=args.speaker_name,
                                               chunk_size=args.chunk_size)
        else:
            paudio = pyaudio.PyAudio()
            decode_chunked_partial_endpointing_mic(asr, feat_info, decodable_opts, paudio, asr_client=asr_client,
                                                   input_microphone_id=args.micid, speaker_str=args.speaker_name,
                                                   samp_freq=args.decode_samplerate, record_samplerate=args.record_samplerate,
                                                   chunk_size=args.chunk_size, wait_for_start_command=args.wait_for_start_command, channels=args.channels,
                                                   resample_algorithm=args.resample_algorithm, save_debug_wav=args.save_debug_wav, use_threads=args.use_threads)
