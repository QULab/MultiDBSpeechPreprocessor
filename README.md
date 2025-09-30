# MultiDBSpeechPreprocessor

## Introduction
This set of scripts is designed to preprocess speech audio files from various
sources or contexts (henceforth "databases" or DBs).
It makes of use of the
[`pyannote.audio`](https://github.com/pyannote/pyannote-audio) speaker
diarization toolkit to identify the primary speaker in a recording (the person
of interest) in order to extract only the relevant parts of the recording.

Specifically, they use the automatic speech recognition (ASR) pipeline
[`speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1).
(Note that this has since been superseded by the `pyannote`
[`community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
open-source diarization model, which is an improvement upon the previous
version, written by the same authors, and should be drop-in compatible with the
current scripts.)
This pre-trained pipeline can be set up for local (offline) use and is a
combination of ML models for speaker segmentation, speaker embedding, and
speaker clustering.

## Processing scripts
For each source database, there exists a `process_data_<db>.py` script inside
the directory `main/`.
These are almost all identical, and are only split for convenience, and because
the preprocessing of a database is usually a one-time operation.
The differences are the location of the metadata (often called
`responses.csv`), which contains the information on a speakers PHQ-9 score,
gender identity, and location and structure of the recordings (one single file
per speaker? multiple files?).
As a result, it would be fairly straightforward to have simply one
preprocessing script that is a little smarter, more configurable, and more
adaptable.

With the `speaker-diarization-3.1` pipeline, and presumably also with the newer
`community-1` pipeline, audio files of arbitrary length can be processed, and
due to the simple segmentation procedure in the processing script(s), one
single file for speaker is preferable and will lead to the most possible
extracted audio (since, by default, segments under a certain duration are
simply discarded; loading the full audio from the speaker reduces the number of
artificial pauses to zero).

In `doc/` there is a file detailing the desired "database design" for the
output of the preprocessing scripts.
In other words, it describes how the output files, which these scripts
generate, should be structured, as well as an explanation of some commonly-used
terms.

Currently, scripts output the preprocessed data inside `main/data/`.
This currently cannot be configured, and is "hard-coded" into the scripts.

### Configuration
The scripts can be configured with the file `process_config.toml` also inside
`main/`.
There are currently six options in three categories, all of which are currently
mandatory.
The categories and options are as follows:

#### `[general]`
- `do_overwrite`: boolean (`true|false`)
    - If `true`, delete any existing files (located at `main/data/<db>/` for
      the database being processed.
      If `false`, refuse to overwrite any files and exit the script.

#### `[path`]
- `path_to_data_dir`: string
    - The path relative to the script's working directory (default: `.`), or an
      absolute path, at which the original (non-preprocessed) data can be
      found.

#### `[segmentation]`
- `duration_min`: float
    - The minimum duration (in seconds) for a single audio segment.
- `duration_max`: float
    - The maximum duration (in seconds) for a single audio segment.
- `ignore_too_short`: boolean (`true|false`)
    - If `true`, any audio portion under `duration_min` will be discarded by
      the script, and not concatenated with any other portions to form a usable
      segment.
      If `false`, raise a `NotImplementedError` when a portion is too short (as
      handling them is not yet implemented).
- `pause_threshold`: float
    - The maximum time allowed (in seconds), during which no speech is
      recognized in an audio recording, that can be still included in a single
      audio portion.
      For example, if speech is detected from 0:53 to 0:56 of a recording, then
      silence from 0:56 to 0:58, then speech again from 0:58 to 1:04, the
      entire portion from 0:53–1:04 can be saved as a single segment,
      preserving the natural pause in the speaker's speech.

## Training-validation-testing split
After all of the databases have been preprocessed, the data must be divided
into training, validation, and testing groups to be used for training machine
learning models.
The script `train_split.py` uses an integer linear programming (ILP) approach
to solve this distribution problem.
The data are categorized on the per-speaker level, which means that each
speaker with their $n$ segments is assigned to either the `train` or `val`
group.

The goals of this optimization problem are to have approximately 85% of all
segments belong to speakers assigned to the `train` group, and the remaining
15% belong to speakers assigned to the `val` group.
We also want both the `train` and `val` sets to have distributions of PHQ-9
scores as similar as possible 1. to each other and 2. to the uniform
distribution.

Furthermore, we want each set to have representation of gender identity as
balanced as possible.

The `test` group is selected on a _per-segment_ level by randomly choosing 10% 
of the segments belonging to `val` speakers (if they have more then ten
segments).

## Docker container
To obtain some degree of portability, a Docker image containing all
prerequisites for running the preprocessing scripts can be generated using the
Dockerfile included in `docker/`.
It generates an image based on a Debian 13 distribution and installs the
`pyannote.audio` dependencies using both `micromamba` and `pip`.

A template command for running the image is provided (`docker_resonanz_cmd`).
The generated image is expected to be named `resonanz:latest`, but this is
naturally very simply modified.
Running `docker_resonanz_cmd` will present the user with a CLI inside the
container, from which the processing scripts or the training-validation split
script can be run.

## Operation procedure
0. (Ensure the Docker image is built and can be run.)
1. Run the Docker image using `docker_resonanz_cmd` or via other means.
2. Run the preprocessing scripts to act on data located at
   `path_to_data_dir/<db-orig-name>`.
   This places the audio segments inside `<script-wording-directory>/data/` and
   generates a `db-<db>.csv` file.
3. After preprocessing all the data, the `cat_dbs.sh` script concatenates the
   `csv` files from the various databases into a primary `db.csv` file.
4. Run the `train_split.py` script to distribute the speakers / segments to the
   required `train`, `val` and `test` groups.
