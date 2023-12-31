# SpikingNN
Playground for spiking neural network
# GSC_speech2spike.ipynb

## Overview
This notebook provides a simple example of converting the Google Speech Commands dataset (GSC) into spiking data using the Speech2Spikes package. It demonstrates taking one audio command and encoding it with a spiking representation as a basic demonstration of how to use the package.

### Source:
- Google Speech Commands Dataset: [Tensorflow Speech Commands Dataset v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
- Speech2Spikes Package: [DOI Reference](https://doi.org/10.1145/3584954.3584995)

## Conversion Process
The conversion from raw audio to a spiking representation involves several steps as outlined in the associated paper. Here's a brief overview of each step:

### 1. Preprocessing the Raw Audio:
The raw audio, assumed to be a 1-second clip sampled at 16 kHz, starts as a 1-dimensional array of 16,000 samples. The audio is first centered and scaled to normalize it for further processing. In the implementation, this involves padding and transposing.

### 2. Frequency Domain Mapping using SDFT:
The Sliding Discrete Fourier Transform (SDFT) incrementally maps the raw audio to the frequency domain, capturing the spectral components over time.

### 3. Mapping to Mel-frequency Bands:
The frequency domain data is then mapped onto Mel-frequency bands, focusing on the perceptually relevant aspects of sound.

### 4. Log Transformation and Feature Stacking:
A log transformation is applied to the Mel-frequency band data, followed by stacking to prepare it for spike encoding.

### 5. Spike Encoding with Step-Forward Algorithm:
The continuous-valued features are encoded into spikes using the Step-Forward algorithm, resulting in a binary representation where each element indicates the presence or absence of a spike. This mimics the firing patterns of neurons. The implementation uses `tensor_to_events` for this step.

## Implementation Notes
In the provided implementation code, `_default_spec_kwargs` specifies an `n_mels` value of 20, which likely contributes to the 20 units/neurons dimension in the final spike data representation. The conversion process involves decomposing the audio into these Mel frequency bands to represent different frequency components as "neurons."

## Usage
The notebook is a demonstration for educational and illustrative purposes, showcasing a simple use case of the Speech2Spikes package for converting audio into a neuromorphic-friendly format. It serves as a starting point for more complex applications and experiments with spiking neural networks or related fields.

