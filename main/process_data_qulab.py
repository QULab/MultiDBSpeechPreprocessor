#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import os
import shutil
import tomllib
import torchaudio
import torch

from pyannote.audio import Pipeline
# from glob import glob
from pydub import AudioSegment
from tqdm import tqdm

origin = "Own Recording"
lang = "de"
db = "qulab"


def qulab_process_sex(s):
    match s:
        case 0:
            return "m"
        case 1:
            return "f"
        case 2:
            return "d"
        case _:
            return "u"


def write_segment(audio, pid, fnum):
    # if necessary, create dir for data
    output_path = f"data/{db}/{pid:04d}"
    os.makedirs(output_path, exist_ok=True)

    # is this conversion necessary?
    audio = audio.set_channels(1).set_frame_rate(48000)

    output_file = f"{output_path}/s{fnum:04d}.wav"
    audio.export(output_file, format="wav")
    return output_file

    
def diarize_audio(file):
    # run the pipeline on an audio file
    waveform, sample_rate = torchaudio.load(file)
    
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    n_speakers_recognized = len(diarization.labels())
    
    # here we assume that the longest-duration speaker in the audio file is the
    # speaker of interest.
    primary = diarization.argmax()
    speaker_support = diarization.label_timeline(primary).support(pause_threshold)
    # exclude overlap with other speakers
    for label in diarization.labels():
        if label != primary:
            speaker_support = speaker_support.extrude(diarization.label_timeline(label))
    
    total_duration = speaker_support.duration()
    
    return diarization, speaker_support

pipeline = Pipeline.from_pretrained("pyannote_config.yaml")
pipeline.to(torch.device("cuda"))


with open("process_config.toml", mode="rb") as fp:
    conf = tomllib.load(fp)
    do_overwrite = conf["general"]["do_overwrite"]
    path_to_data_dir = conf["path"]["path_to_data_dir"]
    duration_min = conf["segmentation"]["duration_min"]
    duration_max = conf["segmentation"]["duration_max"]
    ignore_too_short = conf["segmentation"]["ignore_too_short"]
    pause_threshold = conf["segmentation"]["pause_threshold"]

if os.path.exists(f"data/{db}"):
    if do_overwrite:
        shutil.rmtree(f"data/{db}")
    else:
        raise FileExistsError(f"data/{db} already exists! use do_overwrite in the config to start fresh")


# get information on database
info = pd.read_csv(f"{path_to_data_dir}/{origin}/data/responses.csv")
info_additional = pd.read_csv(f"{path_to_data_dir}/{origin}/post.csv", delimiter=';')
info_additional.set_index("vp_id", inplace=True)

# process and split (simple context-agnostic fractional splitting) data
data = []
pid = 1
for i in tqdm(info.index):
    # with the QULab data, only the full "converted WAVs" exist (entire recording)
    file = f"{path_to_data_dir}/{origin}/data/converted_wavs/{info.loc[i, "pID"]}.wav"
    full_audio = AudioSegment.from_file(file)

    try:
        sex = qulab_process_sex(info_additional.loc[info.loc[i, "pID"], "gender_identification"])
    except KeyError:
        # some of the Versuchspersonencodes don't exist in the post.csv file...
        sex = 'u'

    # process audio
    diarization, support = diarize_audio(file)

    segment_num = 0

    # loop over all portions of the primary speaker in the main recording
    for portion in support:
        start = portion.start
        end = portion.end
        audio = full_audio[start * 1000:end * 1000]
        
        if audio.duration_seconds < duration_min:
            # audio needs to be padded or concatenated
            if ignore_too_short:
                tqdm.write(f"ignore_too_short enabled, skipping {portion} of {file}")
                tqdm.write(f"> portion is {audio.duration_seconds:.2f} seconds\r")
            else:
                raise NotImplementedError
    
        elif audio.duration_seconds > duration_max:
            # audio needs to be split
            n_segments = np.ceil(audio.duration_seconds / duration_max).astype(int)
            seg_duration = audio.duration_seconds / n_segments
            seg_duration_ms = (seg_duration * 1000).astype(int)
        
            for s in range(n_segments):
                segment = audio[s * seg_duration_ms: (s + 1) * seg_duration_ms]
        
                segment_num += 1
                fname = write_segment(segment, pid, segment_num)
                data.append({
                    "db": db,
                    "lang": lang,
                    "pid": f"{pid:04d}",
                    "file_num": f"{segment_num:04d}",
                    "file_path": fname,
                    "sex": sex,
                    "phq-9": info.loc[i, "PHQ-9"],
                    "gad-7": info.loc[i, "GAD-7"]
                })
    
        else:
            # audio duration is acceptable
            segment_num += 1
            fname = write_segment(audio, pid, segment_num)
            data.append({
                "db": db,
                "lang": lang,
                "pid": f"{pid:04d}",
                "file_num": f"{segment_num:04d}",
                "file_path": fname,
                "sex": sex,
                "phq-9": info.loc[i, "PHQ-9"],
                "gad-7": info.loc[i, "GAD-7"]
            })
    
    # only increment pid if files were written
    if segment_num > 1:
        pid += 1  # use a simple numerical pid for each db

pd.DataFrame(data).to_csv(f"db-{db}.csv", index=False)
