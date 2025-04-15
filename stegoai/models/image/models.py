# -*- coding: utf-8 -*-
"""
Image steganography models for Stego-AI.

This module contains the main ImageStegoNet class for image steganography,
which orchestrates the training and usage of neural networks for hiding
and extracting data in images.
"""

import gc
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import imageio
from torch.optim import Adam
from tqdm import tqdm

from stegoai.models.base import BaseStegoNet
from stegoai.models.image.encoders import SimpleEncoder, ResidualEncoder, DenseEncoder, UNetEncoder
from stegoai.models.image.decoders import SimpleDecoder, DenseDecoder, UNetDecoder
from stegoai.models.image.critics import Critic, AdvancedCritic
from stegoai.utils.image_utils import normalize_image, denormalize_image, read_image, save_image
from stegoai.utils.text_utils import text_to_bits, bits_to_bytearray, bytearray_to_text
from stegoai.utils.image_utils import ssim, psnr
from stegoai.utils.error_correction import encode_robust_message, decode_robust_message

# Set up logging
logger = logging.getLogger(__name__)

# Define training metrics to track
METRICS = [
    'val.encoder_mse',    # Mean squared error of the encoder
    'val.decoder_loss',   # Binary cross entropy loss of the decoder
    'val.decoder_acc',    # Bit accuracy of the decoder
    'val.cover_score',    # Critic score for cover images
    'val.stego_score',    # Critic score for steganographic images
    'val.ssim',           # Structural similarity index
    'val.psnr',           # Peak signal-to-noise ratio
    'val.bpp',            # Bits per pixel (capacity)
    'val.detection_rate', # Theoretical detection rate
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.stego_score',
]


class ImageStegoNet(BaseStegoNet):
    """
    Neural network-based image steganography model.
    
    This class handles the creation, training, and usage of steganographic models
    that can hide messages in images using adversarial training techniques.
    
    Attributes:
        data_depth (int): Bits per pixel to hide
        encoder (nn.Module): Network for hiding data in images
        decoder (nn.Module): Network for extracting hidden data
        critic (nn.Module): Network for detecting steganography
        device (torch.device): Device to use for computation
        verbose (bool): Whether to print verbose output
    """

    ARCHITECTURES = {
        'simple': (SimpleEncoder, SimpleDecoder, Critic, 32),
        'residual': (ResidualEncoder, SimpleDecoder, Critic, 64),
        'dense': (DenseEncoder, DenseDecoder, Critic, 64),
        'unet': (UNetEncoder, UNetDecoder, AdvancedCritic, 64),
        'attention': (ResidualEncoder, SimpleDecoder, AdvancedCritic, 64),
    }

    def __init__(
        self, 
        data_depth: int = 1, 
        encoder: Optional[Any] = None, 
        decoder: Optional[Any] = None, 
        critic: Optional[Any] = None, 
        hidden_size: int = 64, 
        cuda: bool = True, 
        verbose: bool = False, 
        log_dir: Optional[str] = None,
    ):
        """
        Initialize ImageStegoNet with encoder, decoder, and critic networks.
        
        Args:
            data_depth: Number of bits to hide per pixel (default: 1)
            encoder: Encoder network class or instance
            decoder: Decoder network class or instance
            critic: Critic network class or instance
            hidden_size: Size of hidden layers in networks (default: 64)
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
        
        # Setup networks
        if encoder is None:
            encoder = SimpleEncoder
        if decoder is None:
            decoder = SimpleDecoder
        if critic is None:
            critic = Critic
            
        # Initialize networks (accepting either classes or instances)
        kwargs = {'data_depth': data_depth, 'hidden_size': hidden_size}
        self.encoder = self._get_network(encoder, kwargs)
        self.decoder = self._get_network(decoder, kwargs)
        self.critic = self._get_network(critic, {'hidden_size': hidden_size})
        
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

    def _get_random_payload(self, batch_size, height, width):
        """
        Generate random binary data for training.
        
        Args:
            batch_size: Number of images in batch
            height: Image height
            width: Image width
            
        Returns:
            torch.Tensor: Random binary data tensor
        """
        shape = (batch_size, self.data_depth, height, width)
        return torch.zeros(shape, device=self.device).random_(0, 2)

    def _encode_decode(self, cover, payload=None, quantize=False):
        """
        Run encoding and decoding process.
        
        Args:
            cover: Cover images tensor
            payload: Data to hide (or None for random data)
            quantize: Whether to simulate 8-bit quantization
            
        Returns:
            tuple: (stego_images, payload, decoded_data)
        """
        batch_size, _, height, width = cover.size()
        
        # Generate random data if not provided
        if payload is None:
            payload = self._get_random_payload(batch_size, height, width)
        
        # Encode the payload into the cover images
        stego = self.encoder(cover, payload)
        
        # Simulate image saving/loading with 8-bit quantization
        if quantize:
            stego = (255.0 * (stego + 1.0) / 2.0).round()
            stego = 2.0 * stego / 255.0 - 1.0

        # Decode the payload from the stego images
        decoded = self.decoder(stego)

        return stego, payload, decoded

    def _compute_losses(self, cover, stego, payload, decoded):
        """
        Compute all losses for training.
        
        Args:
            cover: Original cover images
            stego: Steganographic images
            payload: Original payload data
            decoded: Decoded payload data
            
        Returns:
            tuple: (encoder_mse, decoder_loss, decoder_acc)
        """
        # Image reconstruction loss
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

    def _train_critic(self, train_loader, metrics):
        """
        Train the critic network for one epoch.
        
        Args:
            train_loader: DataLoader providing training images
            metrics: Dictionary to store training metrics
        """
        # Set models to training mode
        self.encoder.eval()  # Freeze encoder during critic training
        self.decoder.eval()  # Freeze decoder during critic training
        self.critic.train()
        
        for cover, _ in tqdm(train_loader, disable=not self.verbose, desc="Training critic"):
            # Clear memory
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
                
            # Move to device
            cover = cover.to(self.device)
            
            # Generate stego images with random payloads
            with torch.no_grad():
                payload = self._get_random_payload(*cover.shape[::2])
                stego = self.encoder(cover, payload)
            
            # Get critic scores for real and fake images
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

    def _train_encoder_decoder(self, train_loader, metrics, alpha=100.0, beta=1.0):
        """
        Train the encoder and decoder networks for one epoch.
        
        Args:
            train_loader: DataLoader providing training images
            metrics: Dictionary to store training metrics
            alpha: Weight for the encoder MSE loss
            beta: Weight for the critic loss
        """
        # Set models to training mode
        self.encoder.train()
        self.decoder.train()
        self.critic.eval()  # Freeze critic during encoder/decoder training
        
        for cover, _ in tqdm(train_loader, disable=not self.verbose, desc="Training encoder/decoder"):
            # Clear memory
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
                
            # Move to device
            cover = cover.to(self.device)
            
            # Encode and decode
            stego, payload, decoded = self._encode_decode(cover)
            
            # Calculate losses
            encoder_mse, decoder_loss, decoder_acc = self._compute_losses(
                cover, stego, payload, decoded)
                
            # Get critic score (want stego images to look like cover images)
            with torch.no_grad():
                stego_score = self.critic(stego).mean()
            
            # Combined loss (weighted sum):
            # - High alpha: prioritize image quality
            # - High beta: prioritize undetectability
            combined_loss = (alpha * encoder_mse) + decoder_loss + (beta * stego_score)
            
            # Update encoder and decoder
            self.decoder_optimizer.zero_grad()
            combined_loss.backward()
            self.decoder_optimizer.step()
            
            # Record metrics
            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

    def _validate(self, val_loader, metrics):
        """
        Validate the model on validation data.
        
        Args:
            val_loader: DataLoader providing validation images
            metrics: Dictionary to store validation metrics
        """
        # Set all models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()
        
        with torch.no_grad():
            for cover, _ in tqdm(val_loader, disable=not self.verbose, desc="Validating"):
                # Clear memory
                gc.collect()
                if self.cuda:
                    torch.cuda.empty_cache()
                    
                # Move to device
                cover = cover.to(self.device)
                
                # Encode and decode with quantization (simulating real-world usage)
                stego, payload, decoded = self._encode_decode(cover, quantize=True)
                
                # Calculate standard metrics
                encoder_mse, decoder_loss, decoder_acc = self._compute_losses(
                    cover, stego, payload, decoded)
                
                # Critic scores for cover and stego images
                cover_score = self.critic(cover).mean()
                stego_score = self.critic(stego).mean()
                
                # Image quality metrics
                ssim_val = ssim(cover, stego)
                psnr_val = psnr(cover, stego)
                
                # Capacity metric (effective bits per pixel)
                bpp = self.data_depth * (2 * decoder_acc.item() - 1)
                
                # Simple theoretical detection rate based on critic scores
                detection_rate = torch.sigmoid(stego_score - cover_score).item()
                
                # Record all metrics
                metrics['val.encoder_mse'].append(encoder_mse.item())
                metrics['val.decoder_loss'].append(decoder_loss.item())
                metrics['val.decoder_acc'].append(decoder_acc.item())
                metrics['val.cover_score'].append(cover_score.item())
                metrics['val.stego_score'].append(stego_score.item())
                metrics['val.ssim'].append(ssim_val.item())
                metrics['val.psnr'].append(psnr_val.item())
                metrics['val.bpp'].append(bpp)
                metrics['val.detection_rate'].append(detection_rate)

    def _save_samples(self, cover, epoch):
        """
        Generate and save sample images for visual inspection.
        
        Args:
            cover: Cover images to use
            epoch: Current epoch number
        """
        if not self.log_dir:
            return
            
        with torch.no_grad():
            self.encoder.eval()
            cover = cover.to(self.device)
            
            # Generate steganographic images with random data
            payload = self._get_random_payload(*cover.shape[::2])
            stego = self.encoder(cover, payload)
            
            # Save sample images
            for i in range(min(4, cover.size(0))):  # Save up to 4 samples
                # Save cover image
                cover_path = os.path.join(self.samples_path, f'sample{i}_cover.png')
                cover_img = denormalize_image(cover[i].cpu())
                imageio.imwrite(cover_path, cover_img)
                
                # Save steganographic image
                stego_path = os.path.join(self.samples_path, f'sample{i}_stego_epoch{epoch:03d}.png')
                stego_img = denormalize_image(stego[i].clamp(-1.0, 1.0).cpu())
                imageio.imwrite(stego_path, stego_img)
                
                # Save difference image (enhanced for visibility)
                diff_path = os.path.join(self.samples_path, f'sample{i}_diff_epoch{epoch:03d}.png')
                diff = torch.abs(stego[i] - cover[i]) * 10  # Enhance differences
                diff_img = denormalize_image(diff.clamp(-1.0, 1.0).cpu())
                imageio.imwrite(diff_path, diff_img)

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
        
        # Get sample images for visualization
        if self.log_dir:
            val_iter = iter(val_loader)
            sample_cover, _ = next(val_iter)
        
        # Train for specified number of epochs
        start_epoch = self.epochs + 1
        end_epoch = start_epoch + epochs
        
        for epoch in range(start_epoch, end_epoch):
            if self.verbose:
                logger.info(f"Epoch {epoch}/{end_epoch-1}")
            
            # Initialize metrics for this epoch
            metrics = {k: [] for k in METRICS}
            
            # Train critic (discriminator)
            self._train_critic(train_loader, metrics)
            
            # Train encoder/decoder (generator)
            self._train_encoder_decoder(train_loader, metrics, alpha, beta)
            
            # Validate
            self._validate(val_loader, metrics)
            
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
                
                # Save sample images
                self._save_samples(sample_cover, epoch)
                
                # Save model checkpoint
                if epoch % save_freq == 0 or epoch == end_epoch - 1:
                    bpp = self.fit_metrics["val.bpp"]
                    acc = self.fit_metrics["val.decoder_acc"]
                    save_name = f'checkpoint_epoch{epoch:03d}_bpp{bpp:.3f}_acc{acc:.3f}.pt'
                    self.save(os.path.join(self.log_dir, save_name))
            
            # Print summary
            if self.verbose:
                val_acc = self.fit_metrics['val.decoder_acc'] * 100
                val_psnr = self.fit_metrics['val.psnr']
                val_bpp = self.fit_metrics['val.bpp']
                logger.info(f"Val accuracy: {val_acc:.2f}%, PSNR: {val_psnr:.2f} dB, BPP: {val_bpp:.3f}")
                
            # Clear memory
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
        
        return self.history

    def _prepare_payload(self, width, height, message, multiple=False, max_capacity=None):
        """
        Convert text message to a payload tensor suitable for encoding.
        
        Args:
            width: Image width
            height: Image height
            message: Text message to encode
            multiple: Whether to repeat the message to fill capacity
            max_capacity: Maximum capacity in bits (optional)
            
        Returns:
            torch.Tensor: Payload tensor ready for encoding
        """
        # Convert text to bits
        message_bits = text_to_bits(message)
        
        # Add termination sequence (32 zeros)
        message_bits = message_bits + [0] * 32
        
        # Calculate total capacity
        capacity = width * height * self.data_depth
        max_capacity = capacity if max_capacity is None else min(capacity, max_capacity)
        
        if len(message_bits) > max_capacity:
            raise ValueError(
                f"Message too large ({len(message_bits)} bits) for image capacity ({max_capacity} bits)"
            )
        
        # Create payload (optionally repeated to fill capacity)
        if multiple:
            payload = []
            while len(payload) < max_capacity:
                payload.extend(message_bits)
            payload = payload[:max_capacity]  # Trim to exact size
        else:
            # Pad with zeros
            payload = message_bits + [0] * (max_capacity - len(message_bits))
            
        # Convert to tensor and reshape
        payload_tensor = torch.FloatTensor(payload).reshape(1, self.data_depth, height, width)
        
        return payload_tensor

    def encode(self, cover_path, output_path, message, multiple=False, quality=95):
        """
        Encode a message into a cover image.
        
        Args:
            cover_path: Path to cover image
            output_path: Path for output steganographic image
            message: Message to hide
            multiple: Whether to repeat the message to fill capacity
            quality: JPEG quality if saving as JPEG (default: 95)
        """
        # Load the cover image
        try:
            # Use our utility function to handle various image formats
            img = read_image(cover_path)
        except Exception as e:
            raise ValueError(f"Failed to load cover image: {e}")
            
        # Convert to tensor
        img_tensor = normalize_image(img)
        img_tensor = torch.FloatTensor(img_tensor).permute(2, 0, 1).unsqueeze(0)
        
        # Get image dimensions
        _, _, height, width = img_tensor.shape
        
        # Create message payload
        payload = self._prepare_payload(width, height, message, multiple)
        
        # Move to device
        img_tensor = img_tensor.to(self.device)
        payload = payload.to(self.device)
        
        # Encode message
        self.encoder.eval()
        with torch.no_grad():
            stego_img = self.encoder(img_tensor, payload)[0].clamp(-1.0, 1.0)
        
        # Convert back to numpy and save
        stego_img = denormalize_image(stego_img.cpu().numpy())
        
        # Use our utility function to save with appropriate format
        save_image(stego_img, output_path, quality=quality)
        
        if self.verbose:
            msg_size = len(text_to_bits(message))
            capacity = width * height * self.data_depth
            usage = msg_size / capacity * 100
            logger.info(f"Message encoded successfully ({msg_size} bits, {usage:.1f}% of capacity)")

    def decode(self, stego_path, max_length=None):
        """
        Decode a message from a steganographic image.
        
        Args:
            stego_path: Path to steganographic image
            max_length: Maximum message length to extract
            
        Returns:
            str: Decoded message
        """
        # Load the stego image
        try:
            img = read_image(stego_path)
        except Exception as e:
            raise ValueError(f"Failed to load stego image: {e}")
        
        # Normalize image to [-1, 1]
        img_tensor = normalize_image(img)
        img_tensor = torch.FloatTensor(img_tensor).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Decode message bits
        self.decoder.eval()
        with torch.no_grad():
            decoded = self.decoder(img_tensor).view(-1) > 0
        
        # Convert bits to messages
        bits = decoded.cpu().numpy().astype(int).tolist()
        
        # Apply max_length if specified
        if max_length:
            bits = bits[:max_length]
        
        # Try to find the message by looking for terminator sequences
        # Find all zero runs of length 32 or more
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
            raise ValueError("Failed to decode any valid message from the image")
            
        # Sort by length (prefer longer messages as they're less likely to be noise)
        messages.sort(key=len, reverse=True)
        
        # Log success
        if self.verbose:
            logger.info(f"Message decoded successfully ({len(messages[0])} characters)")
            
        return messages[0]

    def analyze_image(self, image_path):
        """
        Analyze an image to determine if it contains hidden data.
        
        Args:
            image_path: Path to image for analysis
            
        Returns:
            dict: Analysis results including detection probability
        """
        # Load image
        img = read_image(image_path)
        
        # Normalize and convert to tensor
        img_tensor = normalize_image(img)
        img_tensor = torch.FloatTensor(img_tensor).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Run through critic network
        self.critic.eval()
        with torch.no_grad():
            score = self.critic(img_tensor).item()
        
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

    def compare_images(self, original_path, modified_path):
        """
        Compare original and potentially modified images.
        
        Args:
            original_path: Path to original image
            modified_path: Path to potentially modified image
            
        Returns:
            dict: Comparison results including quality metrics
        """
        # Load images
        original = read_image(original_path)
        modified = read_image(modified_path)
        
        # Ensure same shape
        if original.shape != modified.shape:
            raise ValueError("Images have different dimensions")
            
        # Convert to tensors
        original_tensor = normalize_image(original)
        modified_tensor = normalize_image(modified)
        original_tensor = torch.FloatTensor(original_tensor).permute(2, 0, 1).unsqueeze(0)
        modified_tensor = torch.FloatTensor(modified_tensor).permute(2, 0, 1).unsqueeze(0)
        
        # Calculate metrics
        mse_val = torch.nn.functional.mse_loss(modified_tensor, original_tensor).item()
        ssim_val = ssim(original_tensor, modified_tensor).item()
        psnr_val = psnr(original_tensor, modified_tensor).item()
        
        # Calculate absolute difference
        diff = np.abs(original.astype(np.float32) - modified.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Create result dictionary
        result = {
            'mse': mse_val,
            'ssim': ssim_val,
            'psnr': psnr_val,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'quality_assessment': 'Excellent' if psnr_val > 40 else 'Good' if psnr_val > 30 else 'Fair' if psnr_val > 20 else 'Poor'
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
            ImageStegoNet: Loaded model
            
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
                cuda=cuda,
                verbose=verbose
            )
            
            # Load weights if available
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
            weights_path = os.path.join(models_dir, f"image_{architecture}.pt")
            
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