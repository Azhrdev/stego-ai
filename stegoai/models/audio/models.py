# -*- coding: utf-8 -*-
"""
Audio steganography models for Stego-AI.

This module contains the main AudioStegoNet class for audio steganography,
which orchestrates the hiding and extraction of data in audio signals.
"""

import os
import gc
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

import torch
import numpy as np
from torch.optim import Adam
from tqdm import tqdm

from stegoai.models.base import BaseStegoNet
from stegoai.models.audio.encoders import SpectrogramEncoder, WaveformEncoder, PhaseEncoder
from stegoai.models.audio.decoders import SpectrogramDecoder, WaveformDecoder, PhaseDecoder
from stegoai.models.audio.critics import AudioCritic
from stegoai.utils.audio_utils import (
    load_audio, save_audio, audio_to_spectrogram, spectrogram_to_audio,
    calculate_spectrogram, reconstruct_from_spectrogram, apply_phase_reconstruction,
)
from stegoai.utils.text_utils import text_to_bits, bits_to_bytearray, bytearray_to_text
from stegoai.metrics.audio_metrics import calculate_snr, calculate_pesq

# Set up logging
logger = logging.getLogger(__name__)

# Define training metrics to track
METRICS = [
    'val.encoder_mse',    # Mean squared error of the encoder
    'val.decoder_loss',   # Binary cross entropy loss of the decoder
    'val.decoder_acc',    # Bit accuracy of the decoder
    'val.cover_score',    # Critic score for cover audio
    'val.stego_score',    # Critic score for steganographic audio
    'val.snr',            # Signal-to-noise ratio
    'val.pesq',           # Perceptual evaluation of speech quality
    'val.bpf',            # Bits per frame (capacity)
    'val.detection_rate', # Theoretical detection rate
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.stego_score',
]


class AudioStegoNet(BaseStegoNet):
    """
    Neural network-based audio steganography model.
    
    This class handles the creation, training, and usage of steganographic models
    that can hide messages in audio signals using adversarial training techniques.
    
    The model can operate in three modes:
    - Spectrogram domain: Hide data in the magnitude spectrogram
    - Waveform domain: Hide data directly in the waveform
    - Phase domain: Hide data in the phase component of the spectrogram
    
    Attributes:
        data_depth (int): Bits per frame/sample to hide
        encoder (nn.Module): Network for hiding data in audio
        decoder (nn.Module): Network for extracting hidden data
        critic (nn.Module): Network for detecting steganography
        mode (str): Operating mode (spectrogram, waveform, phase)
        device (torch.device): Device to use for computation
        verbose (bool): Whether to print verbose output
    """

    ARCHITECTURES = {
        'spectrogram': (SpectrogramEncoder, SpectrogramDecoder, AudioCritic, 64),
        'waveform': (WaveformEncoder, WaveformDecoder, AudioCritic, 64),
        'phase': (PhaseEncoder, PhaseDecoder, AudioCritic, 64),
    }

    def __init__(
        self, 
        data_depth: int = 1, 
        encoder: Optional[Any] = None, 
        decoder: Optional[Any] = None, 
        critic: Optional[Any] = None, 
        hidden_size: int = 64,
        mode: str = 'spectrogram',
        sample_rate: int = 44100,
        cuda: bool = True, 
        verbose: bool = False, 
        log_dir: Optional[str] = None,
    ):
        """
        Initialize AudioStegoNet with encoder, decoder, and critic networks.
        
        Args:
            data_depth: Number of bits to hide per frame/sample (default: 1)
            encoder: Encoder network class or instance
            decoder: Decoder network class or instance
            critic: Critic network class or instance
            hidden_size: Size of hidden layers in networks (default: 64)
            mode: Operating mode, one of 'spectrogram', 'waveform', 'phase' (default: 'spectrogram')
            sample_rate: Audio sample rate in Hz (default: 44100)
            cuda: Whether to use GPU acceleration (default: True)
            verbose: Whether to print verbose output (default: False)
            log_dir: Directory to save logs and checkpoints (default: None)
        """
        super().__init__(
            data_depth=data_depth,
            hidden_size=hidden_size,
            cuda=cuda,
            verbose=verbose,
            log_dir=log_dir,
        )
        
        # Set mode and sample rate
        self.mode = mode
        self.sample_rate = sample_rate
        
        # Set appropriate encoder/decoder based on mode if not provided
        if encoder is None and decoder is None:
            if mode == 'spectrogram':
                encoder = SpectrogramEncoder
                decoder = SpectrogramDecoder
            elif mode == 'waveform':
                encoder = WaveformEncoder
                decoder = WaveformDecoder
            elif mode == 'phase':
                encoder = PhaseEncoder
                decoder = PhaseDecoder
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        # Setup networks
        if encoder is None:
            encoder = SpectrogramEncoder
        if decoder is None:
            decoder = SpectrogramDecoder
        if critic is None:
            critic = AudioCritic
            
        # Initialize networks (accepting either classes or instances)
        kwargs = {
            'data_depth': data_depth, 
            'hidden_size': hidden_size,
            'mode': mode,
        }
        self.encoder = self._get_network(encoder, kwargs)
        self.decoder = self._get_network(decoder, kwargs)
        self.critic = self._get_network(critic, {'hidden_size': hidden_size, 'mode': mode})
        
        # Move networks to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)
        
        # Training state
        self.critic_optimizer = None
        self.decoder_optimizer = None

    def _get_network(self, class_or_instance, kwargs):
        """
        Get a network instance from a class or return the instance directly.
        
        Args:
            class_or_instance: Network class or instance
            kwargs: Arguments to pass to the constructor
            
        Returns:
            nn.Module: Instantiated network
        """
        if isinstance(class_or_instance, torch.nn.Module):
            return class_or_instance
            
        return class_or_instance(**kwargs)

    def _get_random_payload(self, batch_size, frames, freq_bins=None):
        """
        Generate random binary data for training.
        
        Args:
            batch_size: Number of audio clips in batch
            frames: Number of time frames
            freq_bins: Number of frequency bins (for spectrogram mode)
            
        Returns:
            torch.Tensor: Random binary data tensor
        """
        if self.mode == 'spectrogram' or self.mode == 'phase':
            if freq_bins is None:
                raise ValueError("freq_bins must be provided for spectrogram mode")
            shape = (batch_size, self.data_depth, frames, freq_bins)
        else:  # waveform mode
            shape = (batch_size, self.data_depth, frames)
            
        return torch.zeros(shape, device=self.device).random_(0, 2)

    def _encode_decode(self, cover, payload=None, quantize=False):
        """
        Run encoding and decoding process.
        
        Args:
            cover: Cover audio tensor
            payload: Data to hide (or None for random data)
            quantize: Whether to simulate quantization
            
        Returns:
            tuple: (stego_audio, payload, decoded_data)
        """
        # Generate random payload if not provided
        if payload is None:
            if self.mode == 'spectrogram' or self.mode == 'phase':
                batch_size, _, frames, freq_bins = cover.size()
                payload = self._get_random_payload(batch_size, frames, freq_bins)
            else:  # waveform mode
                batch_size, frames = cover.size()
                payload = self._get_random_payload(batch_size, frames)
        
        # Encode the payload into the cover audio
        stego = self.encoder(cover, payload)
        
        # Simulate audio processing with quantization
        if quantize:
            if self.mode == 'waveform':
                # Simulate 16-bit quantization
                stego = (stego * 32767).round() / 32767
            else:
                # For spectrograms, simulate limited precision
                stego = torch.round(stego * 1000) / 1000

        # Decode the payload from the stego audio
        decoded = self.decoder(stego)

        return stego, payload, decoded

    def _compute_losses(self, cover, stego, payload, decoded):
        """
        Compute all losses for training.
        
        Args:
            cover: Original cover audio
            stego: Steganographic audio
            payload: Original payload data
            decoded: Decoded payload data
            
        Returns:
            tuple: (encoder_mse, decoder_loss, decoder_acc)
        """
        # Audio reconstruction loss
        encoder_mse = torch.nn.functional.mse_loss(stego, cover)
        
        # Message reconstruction loss
        decoder_loss = torch.nn.functional.binary_cross_entropy_with_logits(decoded, payload)
        
        # Message reconstruction accuracy
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoder_mse, decoder_loss, decoder_acc

    def _get_optimizers(self, learning_rate=1e-4):
        """
        Create optimizers for training.
        
        Args:
            learning_rate: Learning rate for optimizers
            
        Returns:
            tuple: (critic_optimizer, decoder_optimizer)
        """
        encoder_decoder_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)
        decoder_optimizer = Adam(encoder_decoder_params, lr=learning_rate)
        
        return critic_optimizer, decoder_optimizer

    def fit(self, train_loader, val_loader, epochs=5, learning_rate=1e-4, 
            alpha=100.0, beta=1.0, save_freq=1):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train (default: 5)
            learning_rate: Learning rate for optimizers (default: 1e-4)
            alpha: Weight for encoder MSE loss (default: 100.0)
            beta: Weight for critic loss (default: 1.0)
            save_freq: Frequency of model checkpoints (default: 1)
            
        Returns:
            dict: Training history
        """
        # Initialize optimizers if needed
        if self.critic_optimizer is None or self.decoder_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers(learning_rate)
            
        # Train for specified number of epochs
        start_epoch = self.epochs + 1
        end_epoch = start_epoch + epochs
        
        for epoch in range(start_epoch, end_epoch):
            if self.verbose:
                logger.info(f"Epoch {epoch}/{end_epoch-1}")
            
            # Initialize metrics for this epoch
            metrics = {k: [] for k in METRICS}
            
            # Train one epoch (implementation depends on specific model)
            self._train_epoch(train_loader, val_loader, metrics, alpha, beta)
            
            # Update epoch counter
            self.epochs = epoch
            
            # Compute average metrics for this epoch
            self.fit_metrics = {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch
            
            # Log results
            if self.log_dir:
                # Add to history
                self.history.append(self.fit_metrics)
                
                # Write metrics to file
                metrics_path = os.path.join(self.log_dir, 'metrics.json')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=2)
                
                # Save model checkpoint
                if epoch % save_freq == 0 or epoch == end_epoch - 1:
                    bpf = self.fit_metrics.get("val.bpf", 0)
                    acc = self.fit_metrics.get("val.decoder_acc", 0)
                    save_name = f'checkpoint_epoch{epoch:03d}_bpf{bpf:.3f}_acc{acc:.3f}.pt'
                    self.save(os.path.join(self.log_dir, save_name))
            
            # Print summary
            if self.verbose:
                val_acc = self.fit_metrics.get('val.decoder_acc', 0) * 100
                val_snr = self.fit_metrics.get('val.snr', 0)
                val_bpf = self.fit_metrics.get('val.bpf', 0)
                logger.info(f"Val accuracy: {val_acc:.2f}%, SNR: {val_snr:.2f} dB, BPF: {val_bpf:.3f}")
                
            # Clear memory
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
        
        return self.history

    def _train_epoch(self, train_loader, val_loader, metrics, alpha=100.0, beta=1.0):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            metrics: Dictionary to store metrics
            alpha: Weight for encoder MSE loss
            beta: Weight for critic loss
        """
        # Train critic
        self.encoder.eval()
        self.decoder.eval()
        self.critic.train()
        
        for cover, _ in tqdm(train_loader, disable=not self.verbose, desc="Training critic"):
            # Move to device
            cover = cover.to(self.device)
            
            # Generate stego audio with random payloads
            with torch.no_grad():
                if self.mode == 'spectrogram' or self.mode == 'phase':
                    batch_size, _, frames, freq_bins = cover.size()
                    payload = self._get_random_payload(batch_size, frames, freq_bins)
                else:  # waveform mode
                    batch_size, frames = cover.size()
                    payload = self._get_random_payload(batch_size, frames)
                    
                stego = self.encoder(cover, payload)
            
            # Get critic scores for real and fake audio
            cover_score = self.critic(cover).mean()
            stego_score = self.critic(stego).mean()
            
            # Compute critic loss (want to maximize score difference)
            critic_loss = stego_score - cover_score
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Apply weight clipping (WGAN-style)
            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)
            
            # Record metrics
            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.stego_score'].append(stego_score.item())
            
        # Train encoder/decoder
        self.encoder.train()
        self.decoder.train()
        self.critic.eval()
        
        for cover, _ in tqdm(train_loader, disable=not self.verbose, desc="Training encoder/decoder"):
            # Move to device
            cover = cover.to(self.device)
            
            # Encode and decode
            stego, payload, decoded = self._encode_decode(cover)
            
            # Calculate losses
            encoder_mse, decoder_loss, decoder_acc = self._compute_losses(
                cover, stego, payload, decoded)
                
            # Get critic score (want stego audio to sound like cover audio)
            with torch.no_grad():
                stego_score = self.critic(stego).mean()
            
            # Combined loss (weighted sum)
            combined_loss = (alpha * encoder_mse) + decoder_loss + (beta * stego_score)
            
            # Update encoder and decoder
            self.decoder_optimizer.zero_grad()
            combined_loss.backward()
            self.decoder_optimizer.step()
            
            # Record metrics
            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())
            
        # Validate
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()
        
        with torch.no_grad():
            for cover, _ in tqdm(val_loader, disable=not self.verbose, desc="Validating"):
                # Move to device
                cover = cover.to(self.device)
                
                # Encode and decode with quantization
                stego, payload, decoded = self._encode_decode(cover, quantize=True)
                
                # Calculate standard metrics
                encoder_mse, decoder_loss, decoder_acc = self._compute_losses(
                    cover, stego, payload, decoded)
                
                # Critic scores
                cover_score = self.critic(cover).mean()
                stego_score = self.critic(stego).mean()
                
                # Audio quality metrics (simplified for batched data)
                # In a real implementation, would process individual samples
                if self.mode == 'waveform':
                    # For waveform, we can directly compute SNR
                    snr_val = calculate_snr(cover.cpu().numpy(), stego.cpu().numpy())
                    pesq_val = 0  # Placeholder, would compute for real audio
                else:
                    # For spectrogram modes, a placeholder
                    snr_val = 20 * torch.log10(torch.norm(cover) / torch.norm(cover - stego)).item()
                    pesq_val = 0
                
                # Capacity metric (effective bits per frame)
                bpf = self.data_depth * (2 * decoder_acc.item() - 1)
                
                # Detection rate
                detection_rate = torch.sigmoid(stego_score - cover_score).item()
                
                # Record metrics
                metrics['val.encoder_mse'].append(encoder_mse.item())
                metrics['val.decoder_loss'].append(decoder_loss.item())
                metrics['val.decoder_acc'].append(decoder_acc.item())
                metrics['val.cover_score'].append(cover_score.item())
                metrics['val.stego_score'].append(stego_score.item())
                metrics['val.snr'].append(snr_val)
                metrics['val.pesq'].append(pesq_val)
                metrics['val.bpf'].append(bpf)
                metrics['val.detection_rate'].append(detection_rate)

    def _prepare_payload(self, audio_length, message):
        """
        Convert text message to a payload tensor suitable for encoding in audio.
        
        Args:
            audio_length: Length of audio in samples/frames
            message: Text message to encode
            
        Returns:
            torch.Tensor: Payload tensor ready for encoding
        """
        # Convert text to bits
        message_bits = text_to_bits(message)
        
        # Add termination sequence (32 zeros)
        message_bits = message_bits + [0] * 32
        
        # Calculate capacity based on mode and audio length
        if self.mode == 'waveform':
            # For waveform, each sample can hold data_depth bits
            # But we use a more conservative estimate for robustness
            capacity = int(audio_length * self.data_depth * 0.8)
        else:
            # For spectrogram, we need to estimate frames
            frames = audio_length // 512  # Assuming 512-sample frames with no overlap
            capacity = int(frames * 128 * self.data_depth * 0.8)  # 128 freq bins is a common size
        
        if len(message_bits) > capacity:
            raise ValueError(
                f"Message too large ({len(message_bits)} bits) for audio capacity ({capacity} bits)"
            )
        
        # Pad with zeros
        payload = message_bits + [0] * (capacity - len(message_bits))
        
        # For waveform mode, reshape for time domain
        if self.mode == 'waveform':
            # Reshape to match expected input (1, data_depth, time)
            payload_tensor = torch.FloatTensor(payload).reshape(1, self.data_depth, -1)
            
            # Ensure it matches the audio length
            if payload_tensor.shape[2] < audio_length:
                padding = torch.zeros(1, self.data_depth, audio_length - payload_tensor.shape[2])
                payload_tensor = torch.cat([payload_tensor, padding], dim=2)
            elif payload_tensor.shape[2] > audio_length:
                payload_tensor = payload_tensor[:, :, :audio_length]
        else:
            # For spectrogram modes, reshape for time-frequency domain
            frames = audio_length // 512
            freq_bins = 128  # Common size for spectrograms
            
            # Ensure we have enough bits
            if len(payload) < frames * freq_bins * self.data_depth:
                payload = payload + [0] * (frames * freq_bins * self.data_depth - len(payload))
            elif len(payload) > frames * freq_bins * self.data_depth:
                payload = payload[:frames * freq_bins * self.data_depth]
                
            payload_tensor = torch.FloatTensor(payload).reshape(1, self.data_depth, frames, freq_bins)
            
        return payload_tensor

    def encode(self, cover_path, output_path, message, quality='high'):
        """
        Encode a message into a cover audio file.
        
        Args:
            cover_path: Path to cover audio file
            output_path: Path for output steganographic audio
            message: Message to hide
            quality: Audio quality setting ('high', 'medium', 'low')
        """
        # Load the cover audio
        try:
            audio, sample_rate = load_audio(cover_path)
            if sample_rate != self.sample_rate:
                if self.verbose:
                    logger.warning(f"Resampling audio from {sample_rate} Hz to {self.sample_rate} Hz")
                # In a real implementation, would resample here
        except Exception as e:
            raise ValueError(f"Failed to load cover audio: {e}")
            
        # Prepare audio tensor based on mode
        if self.mode == 'spectrogram' or self.mode == 'phase':
            # Convert to spectrogram
            spectrogram = audio_to_spectrogram(audio)
            # Convert to tensor
            audio_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)  # Add batch dimension
        else:  # waveform mode
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Add batch dimension
            
        # Create message payload
        payload = self._prepare_payload(len(audio), message)
        
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        payload = payload.to(self.device)
        
        # Encode message
        self.encoder.eval()
        with torch.no_grad():
            stego_tensor = self.encoder(audio_tensor, payload).clamp(-1.0, 1.0)
            
        # Convert back to audio
        if self.mode == 'spectrogram':
            # Convert spectrogram tensor back to waveform
            stego_spectrogram = stego_tensor[0].cpu().numpy()
            stego_audio = spectrogram_to_audio(stego_spectrogram, original_phase=audio)
        elif self.mode == 'phase':
            # For phase encoding, need to handle phase reconstruction
            stego_spectrogram = stego_tensor[0].cpu().numpy()
            stego_audio = apply_phase_reconstruction(stego_spectrogram, audio)
        else:  # waveform mode
            stego_audio = stego_tensor[0].cpu().numpy()
            
        # Save audio with appropriate format and quality
        bit_depth = 24 if quality == 'high' else 16 if quality == 'medium' else 8
        save_audio(stego_audio, output_path, self.sample_rate, bit_depth=bit_depth)
        
        if self.verbose:
            msg_size = len(text_to_bits(message))
            if self.mode == 'waveform':
                capacity = len(audio) * self.data_depth
            else:
                frames = len(audio) // 512
                capacity = frames * 128 * self.data_depth
            usage = msg_size / capacity * 100
            logger.info(f"Message encoded successfully ({msg_size} bits, {usage:.1f}% of capacity)")

    def decode(self, stego_path, max_length=None):
        """
        Decode a message from a steganographic audio file.
        
        Args:
            stego_path: Path to steganographic audio
            max_length: Maximum message length to extract (in bits)
            
        Returns:
            str: Decoded message
        """
        # Load the stego audio
        try:
            audio, sample_rate = load_audio(stego_path)
            if sample_rate != self.sample_rate:
                if self.verbose:
                    logger.warning(f"Resampling audio from {sample_rate} Hz to {self.sample_rate} Hz")
                # In a real implementation, would resample here
        except Exception as e:
            raise ValueError(f"Failed to load stego audio: {e}")
        
        # Prepare audio tensor based on mode
        if self.mode == 'spectrogram' or self.mode == 'phase':
            # Convert to spectrogram
            spectrogram = audio_to_spectrogram(audio)
            # Convert to tensor
            audio_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)  # Add batch dimension
        else:  # waveform mode
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Add batch dimension
            
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Decode message bits
        self.decoder.eval()
        with torch.no_grad():
            decoded = self.decoder(audio_tensor).view(-1) > 0
        
        # Convert bits to messages
        bits = decoded.cpu().numpy().astype(int).tolist()
        
        # Apply max_length if specified
        if max_length:
            bits = bits[:max_length]
            
        # Find terminator sequences (runs of 32+ zeros)
        zero_positions = []
        run_length = 0
        for i, bit in enumerate(bits):
            if bit == 0:
                run_length += 1
            else:
                if run_length >= 32:
                    zero_positions.append(i - run_length)
                run_length = 0
                
        # Add end position if we end with zeros
        if run_length >= 32:
            zero_positions.append(len(bits) - run_length)
            
        # If no terminator found, try the whole sequence
        if not zero_positions:
            zero_positions = [len(bits)]
            
        # Extract messages at different cut points
        messages = []
        start_pos = 0
        for end_pos in zero_positions:
            message_bits = bits[start_pos:end_pos]
            if message_bits:
                # Convert bits to bytes
                byte_data = bits_to_bytearray(message_bits)
                
                # Try to decode as text
                text = bytearray_to_text(byte_data)
                
                # Only add if valid text was decoded
                if text:
                    messages.append(text)
                    
            start_pos = end_pos + 32  # Skip the terminator
            
        # Return the longest valid message as a heuristic
        if not messages:
            # Try one more time with the whole sequence
            byte_data = bits_to_bytearray(bits)
            text = bytearray_to_text(byte_data)
            if text:
                messages.append(text)
        
        if not messages:
            raise ValueError("Failed to decode any valid message from the audio")
            
        # Sort by length (prefer longer messages as they're less likely to be noise)
        messages.sort(key=len, reverse=True)
        
        # Log success
        if self.verbose:
            logger.info(f"Message decoded successfully ({len(messages[0])} characters)")
            
        return messages[0]

    def analyze_audio(self, audio_path):
        """
        Analyze audio to determine if it contains hidden data.
        
        Args:
            audio_path: Path to audio for analysis
            
        Returns:
            dict: Analysis results including detection probability
        """
        # Load audio
        audio, sample_rate = load_audio(audio_path)
        
        # Prepare audio tensor based on mode
        if self.mode == 'spectrogram' or self.mode == 'phase':
            spectrogram = audio_to_spectrogram(audio)
            audio_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)
        else:  # waveform mode
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Run through critic network
        self.critic.eval()
        with torch.no_grad():
            score = self.critic(audio_tensor).item()
        
        # Calculate probability of steganography (sigmoid of score)
        probability = 1 / (1 + np.exp(-score))
        
        # Create analysis result
        result = {
            'score': score,
            'probability': probability,
            'assessment': 'Likely contains hidden data' if probability > 0.7 else 'Likely clean',
            'confidence': 'High' if abs(probability - 0.5) > 0.4 else 'Medium' if abs(probability - 0.5) > 0.2 else 'Low'
        }
        
        return result

    @classmethod
    def load(cls, path=None, architecture=None, cuda=True, verbose=False):
        """
        Load a model from disk or use a predefined architecture.
        
        Args:
            path: Path to saved model (optional)
            architecture: Name of predefined architecture (optional)
            cuda: Whether to use GPU if available
            verbose: Whether to print verbose output
            
        Returns:
            AudioStegoNet: Loaded model
            
        Note:
            Either path or architecture must be provided, but not both.
        """
        if path is None and architecture is None:
            raise ValueError("Either path or architecture must be provided")
            
        if path is not None and architecture is not None:
            raise ValueError("Only one of path or architecture should be provided")
        
        # Load from predefined architecture
        if architecture is not None:
            if architecture not in cls.ARCHITECTURES:
                raise ValueError(f"Unknown architecture: {architecture}. Available: {list(cls.ARCHITECTURES.keys())}")
                
            # Get architecture components
            encoder_cls, decoder_cls, critic_cls, hidden_size = cls.ARCHITECTURES[architecture]
            
            # Create model
            model = cls(
                encoder=encoder_cls,
                decoder=decoder_cls,
                critic=critic_cls,
                hidden_size=hidden_size,
                mode=architecture,  # Use architecture name as mode
                cuda=cuda,
                verbose=verbose
            )
            
            # Load weights if available
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            weights_path = os.path.join(models_dir, f"audio_{architecture}.pt")
            
            if os.path.exists(weights_path):
                # Load state dictionary
                state_dict = torch.load(weights_path, map_location='cpu')
                model.encoder.load_state_dict(state_dict['encoder'])
                model.decoder.load_state_dict(state_dict['decoder'])
                model.critic.load_state_dict(state_dict['critic'])
                
                if verbose:
                    logger.info(f"Loaded weights for {architecture} architecture")
        
        # Load from path
        else:
            try:
                model = torch.load(path, map_location='cpu')
                model.verbose = verbose
                model.set_device(cuda)
                
                if verbose:
                    logger.info(f"Model loaded from {path}")
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")
        
        return model