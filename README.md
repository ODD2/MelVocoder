# Artist20 Singer Prediction
## Description
This repository includes the project for the second homework of the course "Deep Learning for Music Analysis and Generation" lectured by Prof. Yang at the National Taiwan University. The main goals of this work is to train a melvocoder on the [M4Singer](https://m4singer.github.io/) dataset. Given an mel-spectrogram of a singing segment, the model should generate the waveform corresponding to the sepctrogram. This project relies on [HiFi-GAN](https://github.com/jik876/hifi-gan) and [sobel-operator-pytorch](https://github.com/chaddy1004/sobel-operator-pytorch), big thanks to the authors.

## Create Environment 
```bash
pip install -r requirements.txt
```

## Training
The file structure of the dataset is expected to be something similar like this:
```
./dataset
    |- audios/
        |- 0001.mp3
        |- 0002.mp3
        |- 0003.mp3
        ...
    |- split
        |- train.txt
        |- valid.txt
```
The following command starts the training process with configuration 'config/config_v1.json' and save the checkpoint to the 'checkpoint/test/' folder:
```bash
python train.py \
--config=configs/config_v1.json \
--input_wavs_dir=dataset/audios \
--input_training_file=./dataset/split/train.txt \
--input_validation_file=./dataset/split/valid.txt \
--checkpoint_path=./checkpoints/test
```
## Inference
### Preprocess Audio Files for Mel-Spectrogram
Please change the source and destination folders and run the 'preprocess.py' file to derive the mel-spectrograms.
```bash
python -m preprocess
```
### Generate the Waveforms
- Please download the model weights and config from Google Drive: [config.json](https://drive.google.com/file/d/1sUmhYvu5oOiLlacI_GVGLv34hUSy5JOe/view?usp=sharing), [weights](https://drive.google.com/file/d/1lI5X329HsT-4vDLJFEIkhUooo-DotGqG/view?usp=sharing)
- Inference with the following command:
```bash
python -m inference \
# the folder path for the mel-spectrograms
--input_mels_dir=$input_mel_spec_folder \ 
# the folder path to save the generated audios
--output_dir=$output_audio_folder \ 
# the path that includes both the weight and the config
--checkpoint_file=$generator_checkpoint_path 
```
