# -*- coding: utf-8 -*-
"""
Audio processing utilities for Stego-AI.

This module provides functions for loading, saving, and processing
audio files used in steganographic operations.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
import torch
from scipy import signal

# Set up logging
logger = logging.getLogger(__name__)


def load_audio(file_path, sr=44100, mono=True):
    """
    Load an audio file.
    
    Args:
        file_path (str): Path to audio file
        sr (int): Target sample rate (default: 44100 Hz)
        mono (bool): Convert to mono if True (default: True)
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        y, sr_orig = librosa.load(file_path, sr=sr, mono=mono)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def save_audio(audio, file_path, sr=44100, format=None, bit_depth=16):
    """
    Save audio data to a file.
    
    Args:
        audio (numpy.ndarray): Audio data
        file_path (str): Path to save audio
        sr (int): Sample rate (default: 44100 Hz)
        format (str): Audio format (default: inferred from extension)
        bit_depth (int): Bit depth (8, 16, 24, 32) for PCM formats
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine subtype based on bit depth
        if bit_depth == 8:
            subtype = 'PCM_S8'
        elif bit_depth == 16:
            subtype = 'PCM_16'
        elif bit_depth == 24:
            subtype = 'PCM_24'
        else:  # 32
            subtype = 'FLOAT'
            
        # Save using soundfile
        sf.write(file_path, audio, sr, format=format, subtype=subtype)
        
    except Exception as e:
        logger.error(f"Error saving audio to {file_path}: {e}")
        raise


def audio_to_spectrogram(audio, n_fft=2048, hop_length=512, win_length=None):
    """
    Convert audio waveform to magnitude spectrogram.
    
    Args:
        audio (numpy.ndarray): Audio waveform
        n_fft (int): FFT window size (default: 2048)
        hop_length (int): Hop length (default: 512)
        win_length (int): Window length (default: n_fft)
        
    Returns:
        numpy.ndarray: Reconstructed audio waveform
    """
    # Denormalize magnitude from [-1, 1] to dB scale
    db_magnitude = ((magnitude - 1.0) / 2.0) * 80.0
    
    # Convert from dB to magnitude
    magnitude = librosa.db_to_amplitude(db_magnitude)
    
    if phase is not None:
        # Denormalize phase from [-1, 1] to [-π, π]
        phase = phase * np.pi
        
        # Create complex spectrogram
        complex_spectrogram = magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        audio = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length)
    else:
        # Griffin-Lim algorithm for phase reconstruction
        audio = librosa.griffinlim(magnitude, n_iter=10, hop_length=hop_length, win_length=win_length)
    
    return audio


def apply_phase_reconstruction(modified_phase, original_audio, n_fft=2048, hop_length=512, win_length=None):
    """
    Apply modified phase to original audio.
    
    Args:
        modified_phase (numpy.ndarray): Modified phase (normalized to [-1, 1])
        original_audio (numpy.ndarray): Original audio waveform
        n_fft (int): FFT window size (default: 2048)
        hop_length (int): Hop length (default: 512)
        win_length (int): Window length (default: n_fft)
        
    Returns:
        numpy.ndarray: Reconstructed audio with modified phase
    """
    # Compute STFT of original audio
    stft = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Get magnitude from original
    magnitude = np.abs(stft)
    
    # Denormalize modified phase from [-1, 1] to [-π, π]
    phase = modified_phase * np.pi
    
    # Apply modified phase to original magnitude
    complex_spectrogram = magnitude * np.exp(1j * phase)
    
    # Inverse STFT
    audio = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length)
    
    return audio


def add_noise(audio, snr_db):
    """
    Add Gaussian noise to audio signal at specified SNR.
    
    Args:
        audio (numpy.ndarray): Audio signal
        snr_db (float): Signal-to-noise ratio in dB
        
    Returns:
        numpy.ndarray: Noisy audio signal
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    # Add noise to signal
    noisy_audio = audio + noise
    
    return noisy_audio


def apply_mp3_compression_simulation(audio, sr):
    """
    Simulate MP3 compression artifacts.
    
    This is a simplified simulation by applying low-pass filtering.
    For real MP3 compression, you would need to save and reload using an MP3 codec.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sample rate
        
    Returns:
        numpy.ndarray: Audio with simulated compression artifacts
    """
    # Simple approximation of MP3 compression artifacts
    # by using a low-pass filter (MP3 typically cuts high frequencies)
    cutoff = 16000  # MP3 often cuts around 16kHz
    nyquist = sr // 2
    normal_cutoff = cutoff / nyquist
    
    # Design a low-pass filter
    b, a = signal.butter(10, normal_cutoff, btype='lowpass')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    # Also add some quantization noise
    bits = 10  # MP3 uses psychoacoustic model, but we'll use simple bit reduction
    max_val = np.max(np.abs(filtered_audio))
    if max_val > 0:
        steps = 2 ** bits
        quantized = np.round(filtered_audio / max_val * steps) / steps * max_val
        compressed_audio = quantized
    else:
        compressed_audio = filtered_audio
    
    return compressed_audio


def apply_time_stretch(audio, rate):
    """
    Apply time stretching to audio.
    
    Args:
        audio (numpy.ndarray): Audio signal
        rate (float): Stretch factor (>1 for slower, <1 for faster)
        
    Returns:
        numpy.ndarray: Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def apply_pitch_shift(audio, sr, n_steps):
    """
    Apply pitch shifting to audio.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sample rate
        n_steps (float): Number of semitones to shift
        
    Returns:
        numpy.ndarray: Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def extract_audio_features(audio, sr):
    """
    Extract common audio features.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sample rate
        
    Returns:
        dict: Dictionary of audio features
    """
    features = {}
    
    # Basic statistics
    features['duration'] = len(audio) / sr
    features['rms'] = np.sqrt(np.mean(audio ** 2))
    features['peak'] = np.max(np.abs(audio))
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    features['mel_mean'] = np.mean(mel_spec)
    features['mel_std'] = np.std(mel_spec)
    
    # Zero crossing rate
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    
    # MFCC (first few coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features['mfccs'] = np.mean(mfccs, axis=1)
    
    return features


def convert_to_mono(audio):
    """
    Convert multi-channel audio to mono.
    
    Args:
        audio (numpy.ndarray): Audio signal (can be multi-channel)
        
    Returns:
        numpy.ndarray: Mono audio signal
    """
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio


def split_into_chunks(audio, chunk_size, overlap=0):
    """
    Split audio into fixed-size chunks with optional overlap.
    
    Args:
        audio (numpy.ndarray): Audio signal
        chunk_size (int): Size of each chunk in samples
        overlap (float): Overlap between chunks (0 to 1)
        
    Returns:
        list: List of audio chunks
    """
    # Calculate step size
    step = int(chunk_size * (1 - overlap))
    
    # Calculate number of chunks
    num_chunks = (len(audio) - chunk_size) // step + 1
    
    # Create chunks
    chunks = []
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_size
        if end <= len(audio):
            chunks.append(audio[start:end])
    
    return chunks


def calculate_snr(original, noisy):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    Args:
        original (numpy.ndarray): Original clean signal
        noisy (numpy.ndarray): Noisy signal
        
    Returns:
        float: SNR in dB
    """
    # Ensure signals are the same length
    min_len = min(len(original), len(noisy))
    original = original[:min_len]
    noisy = noisy[:min_len]
    
    # Calculate noise
    noise = original - noisy
    
    # Calculate powers
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')
    
    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr


def calculate_pesq(original, degraded, sr):
    """
    Calculate Perceptual Evaluation of Speech Quality (PESQ).
    
    This is a placeholder for the real PESQ implementation,
    which requires external libraries.
    
    Args:
        original (numpy.ndarray): Original clean signal
        degraded (numpy.ndarray): Degraded signal
        sr (int): Sample rate
        
    Returns:
        float: PESQ score (or None if library not available)
    """
    try:
        # This would use a actual PESQ implementation
        # In a real system, you would use:
        # from pesq import pesq
        # return pesq(sr, original, degraded, 'wb')
        
        # For now, return a placeholder based on SNR
        snr = calculate_snr(original, degraded)
        # Approximate mapping from SNR to PESQ
        pesq_score = 1.0 + 3.0 * (1 - np.exp(-snr / 20))
        return min(max(pesq_score, 1.0), 4.5)  # PESQ is between 1.0 and 4.5
    except ImportError:
        logger.warning("PESQ calculation requires the 'pesq' library. Using SNR approximation.")
        return None


def waveform_to_tensor(waveform, add_batch=True):
    """
    Convert numpy waveform to PyTorch tensor.
    
    Args:
        waveform (numpy.ndarray): Audio waveform
        add_batch (bool): Whether to add batch dimension
        
    Returns:
        torch.Tensor: Waveform tensor
    """
    # Convert to tensor
    tensor = torch.from_numpy(waveform).float()
    
    # Add batch dimension if requested
    if add_batch:
        tensor = tensor.unsqueeze(0)
        
    return tensor


def spectrogram_to_tensor(spectrogram, add_batch=True, add_channel=True):
    """
    Convert numpy spectrogram to PyTorch tensor.
    
    Args:
        spectrogram (numpy.ndarray): Spectrogram
        add_batch (bool): Whether to add batch dimension
        add_channel (bool): Whether to add channel dimension
        
    Returns:
        torch.Tensor: Spectrogram tensor
    """
    # Convert to tensor
    tensor = torch.from_numpy(spectrogram).float()
    
    # Add channel dimension if requested
    if add_channel:
        tensor = tensor.unsqueeze(0)
        
    # Add batch dimension if requested
    if add_batch:
        tensor = tensor.unsqueeze(0)
        
    return tensor


def tensor_to_waveform(tensor):
    """
    Convert PyTorch tensor to numpy waveform.
    
    Args:
        tensor (torch.Tensor): Waveform tensor
        
    Returns:
        numpy.ndarray: Audio waveform
    """
    # Handle batch dimension
    if tensor.dim() > 1:
        tensor = tensor[0]  # Take first sample
    
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy
    waveform = tensor.numpy()
        
    return waveform


def tensor_to_spectrogram(tensor):
    """
    Convert PyTorch tensor to numpy spectrogram.
    
    Args:
        tensor (torch.Tensor): Spectrogram tensor
        
    Returns:
        numpy.ndarray: Spectrogram
    """
    # Handle batch and channel dimensions
    if tensor.dim() > 2:
        tensor = tensor[0]  # Take first sample
    if tensor.dim() > 2:
        tensor = tensor[0]  # Take first channel
    
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy
    spectrogram = tensor.numpy()
        
    return spectrogram512

def spectrogram_to_audio(spectrogram, n_fft=2048, hop_length=512, win_length=None, 
                        original_phase=None, num_iterations=10):
    """
    Convert magnitude spectrogram back to audio waveform.
    
    Args:
        spectrogram (numpy.ndarray): Magnitude spectrogram
        n_fft (int): FFT window size (default: 2048)
        hop_length (int): Hop length (default: 512)
        win_length (int): Window length (default: n_fft)
        original_phase (numpy.ndarray): Original phase (default: None, random phase)
        num_iterations (int): Number of iterations for Griffin-Lim (default: 10)
        
    Returns:
        numpy.ndarray: Audio waveform
    """
    # Denormalize from [-1, 1] to dB scale
    db_magnitude = ((spectrogram - 1.0) / 2.0) * 80.0
    
    # Convert from dB to magnitude
    magnitude = librosa.db_to_amplitude(db_magnitude)
    
    # If original phase is provided, use it
    if original_phase is not None:
        # Get phase from original audio
        if isinstance(original_phase, np.ndarray) and original_phase.ndim == 1:
            # If we got an audio waveform, compute its phase
            stft_original = librosa.stft(original_phase, n_fft=n_fft, hop_length=hop_length, 
                                        win_length=win_length)
            phase = np.angle(stft_original)
        else:
            # If we already have complex STFT or phase
            phase = np.angle(original_phase) if np.iscomplexobj(original_phase) else original_phase
            
        # Apply phase
        complex_spectrogram = magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        audio = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=win_length)
    else:
        # Griffin-Lim algorithm for phase reconstruction
        audio = librosa.griffinlim(magnitude, n_iter=num_iterations, hop_length=hop_length, 
                                  win_length=win_length)
    
    return audio


def calculate_spectrogram(audio, n_fft=2048, hop_length=512, win_length=None, return_phase=False):
    """
    Calculate spectrogram and optionally phase.
    
    Args:
        audio (numpy.ndarray): Audio waveform
        n_fft (int): FFT window size (default: 2048)
        hop_length (int): Hop length (default: 512)
        win_length (int): Window length (default: n_fft)
        return_phase (bool): Whether to return phase (default: False)
        
    Returns:
        numpy.ndarray or tuple: Magnitude spectrogram or (magnitude, phase)
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Separate magnitude and phase
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Convert to log scale
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Normalize to [-1, 1]
    normalized_magnitude = (log_magnitude / 80.0) * 2.0 + 1.0
    normalized_magnitude = np.clip(normalized_magnitude, -1.0, 1.0)
    
    # Normalize phase to [-1, 1]
    normalized_phase = phase / np.pi
    
    if return_phase:
        return normalized_magnitude, normalized_phase
    else:
        return normalized_magnitude


def reconstruct_from_spectrogram(magnitude, phase=None, n_fft=2048, hop_length=512, win_length=None):
    """
    Reconstruct audio from magnitude spectrogram and optional phase.
    
    Args:
        magnitude (numpy.ndarray): Magnitude spectrogram (normalized to [-1, 1])
        phase (numpy.ndarray): Phase information (normalized to [-1, 1])
        n_fft (int): FFT window size (default: 2048)
        hop_length (int): Hop length (default:"
        "    Returns:
        numpy.ndarray: Magnitude spectrogram
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Convert to magnitude spectrogram
    magnitude = np.abs(stft)
    
    # Convert to log scale
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Normalize to [-1, 1]
    normalized = (log_magnitude / 80.0) * 2.0 + 1.0
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized