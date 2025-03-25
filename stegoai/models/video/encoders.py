# -*- coding: utf-8 -*-
"""
Video steganography encoders for Stego-AI.

This module implements encoder architectures for hiding messages
in videos using different steganographic techniques.
"""

import os
import logging
import tempfile
import random
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from stegoai.models.base import BaseEncoder
from stegoai.models.image.encoders import ImageStegoEncoder
from stegoai.utils.text_utils import text_to_bits, bits_to_bytearray

# Set up logging
logger = logging.getLogger(__name__)


class VideoFrameLSBEncoder(BaseEncoder):
    """
    Encoder for hiding messages in video frames using LSB steganography.
    
    This encoder hides data in the least significant bits of pixel values
    across selected frames of a video.
    """
    
    def __init__(self, data_depth: int = 1, cuda: bool = True):
        """
        Initialize the Frame LSB encoder.
        
        Args:
            data_depth: Number of bits to hide per pixel component
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.data_depth = min(data_depth, 3)  # Limit to 3 bits for quality
        self.cuda = cuda
        
        # Define parameters
        self.bit_mapping = nn.Parameter(torch.eye(2), requires_grad=False)
        
        # Optional GPU acceleration
        if self.cuda and torch.cuda.is_available():
            self.bit_mapping = self.bit_mapping.cuda()
    
    def forward(self, frames: List[np.ndarray], message_bits: List[int]) -> List[np.ndarray]:
        """
        Hide a message in video frames using LSB steganography.
        
        Args:
            frames: List of video frames
            message_bits: Bits to hide
            
        Returns:
            list: Modified frames with hidden message
        """
        if not frames or not message_bits:
            return frames
        
        # Create a copy of frames to avoid modifying originals
        stego_frames = [frame.copy() for frame in frames]
        
        # Calculate total capacity across all frames
        total_pixels = sum(frame.shape[0] * frame.shape[1] for frame in stego_frames)
        total_capacity = total_pixels * 3 * self.data_depth  # 3 channels per pixel
        
        if len(message_bits) > total_capacity:
            logger.warning(f"Message size ({len(message_bits)} bits) exceeds capacity ({total_capacity} bits)")
            # Truncate message
            message_bits = message_bits[:total_capacity]
        
        # Determine distribution strategy
        # Spread message evenly across frames
        bits_per_frame = [len(message_bits) // len(stego_frames)] * len(stego_frames)
        remaining_bits = len(message_bits) % len(stego_frames)
        
        # Distribute remaining bits
        for i in range(remaining_bits):
            bits_per_frame[i] += 1
        
        # Hide message across frames
        bit_idx = 0
        
        for i, frame in enumerate(stego_frames):
            bits_to_hide = bits_per_frame[i]
            if bits_to_hide == 0:
                continue
            
            # Use modified frame for embedding
            stego_frame = self._encode_frame(
                frame, 
                message_bits[bit_idx:bit_idx + bits_to_hide]
            )
            
            # Update stego_frames with modified frame
            stego_frames[i] = stego_frame
            
            # Move to next bits
            bit_idx += bits_to_hide
        
        return stego_frames
    
    def _encode_frame(self, frame: np.ndarray, bits: List[int]) -> np.ndarray:
        """
        Hide bits in a single frame.
        
        Args:
            frame: Video frame
            bits: Bits to hide
            
        Returns:
            np.ndarray: Modified frame with hidden bits
        """
        # Create a copy of the frame
        stego = frame.copy()
        
        # Get frame dimensions
        height, width, channels = stego.shape
        
        # Calculate capacity
        capacity = height * width * channels * self.data_depth
        
        if len(bits) > capacity:
            logger.warning(f"Too many bits ({len(bits)}) for frame capacity ({capacity})")
            # Truncate bits
            bits = bits[:capacity]
        
        # Generate random pixel indices
        all_indices = []
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    for d in range(self.data_depth):
                        all_indices.append((y, x, c, d))
        
        # Shuffle indices for security
        random.shuffle(all_indices)
        
        # Limit to the number of bits we need to hide
        indices = all_indices[:len(bits)]
        
        # Hide bits
        for i, (y, x, c, d) in enumerate(indices):
            if i >= len(bits):
                break
            
            # Get pixel value
            pixel = stego[y, x, c]
            
            # Clear the dth bit
            mask = ~(1 << d)
            pixel = pixel & mask
            
            # Set the dth bit according to the message
            pixel = pixel | (bits[i] << d)
            
            # Update pixel
            stego[y, x, c] = pixel
        
        return stego


class VideoFrameDCTEncoder(BaseEncoder):
    """
    Encoder for hiding messages in video frames using DCT steganography.
    
    This encoder hides data in the DCT coefficients of frames, similar to
    how information is hidden in JPEG compression.
    """
    
    def __init__(self, data_depth: int = 1, cuda: bool = True):
        """
        Initialize the Frame DCT encoder.
        
        Args:
            data_depth: Number of bits to modify per coefficient
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.data_depth = data_depth
        self.cuda = cuda
        
        # Define parameters
        self.coefficient_masks = [
            # Standard zigzag ordering of DCT coefficients (partial)
            # Each tuple is (row, col) of the coefficient in 8x8 block
            (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), 
            (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
            (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)
        ]
    
    def forward(self, frames: List[np.ndarray], message_bits: List[int]) -> List[np.ndarray]:
        """
        Hide a message in video frames using DCT steganography.
        
        Args:
            frames: List of video frames
            message_bits: Bits to hide
            
        Returns:
            list: Modified frames with hidden message
        """
        if not frames or not message_bits:
            return frames
        
        # Create a copy of frames to avoid modifying originals
        stego_frames = [frame.copy() for frame in frames]
        
        # Convert frames to YCrCb color space (better for DCT)
        ycrcb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) for frame in stego_frames]
        
        # Calculate total capacity across all frames
        total_blocks = 0
        for frame in ycrcb_frames:
            height, width = frame.shape[:2]
            blocks_h = height // 8
            blocks_w = width // 8
            total_blocks += blocks_h * blocks_w
        
        usable_coefficients = min(len(self.coefficient_masks), 20)  # Limit for quality
        total_capacity = total_blocks * usable_coefficients * self.data_depth
        
        if len(message_bits) > total_capacity:
            logger.warning(f"Message size ({len(message_bits)} bits) exceeds capacity ({total_capacity} bits)")
            # Truncate message
            message_bits = message_bits[:total_capacity]
        
        # Determine distribution strategy
        # Spread message evenly across frames
        bits_per_frame = [len(message_bits) // len(ycrcb_frames)] * len(ycrcb_frames)
        remaining_bits = len(message_bits) % len(ycrcb_frames)
        
        # Distribute remaining bits
        for i in range(remaining_bits):
            bits_per_frame[i] += 1
        
        # Hide message across frames
        bit_idx = 0
        
        for i, frame in enumerate(ycrcb_frames):
            bits_to_hide = bits_per_frame[i]
            if bits_to_hide == 0:
                continue
            
            # Use modified frame for embedding
            ycrcb_frames[i] = self._encode_frame_dct(
                frame, 
                message_bits[bit_idx:bit_idx + bits_to_hide]
            )
            
            # Move to next bits
            bit_idx += bits_to_hide
        
        # Convert back to BGR color space
        for i, frame in enumerate(ycrcb_frames):
            stego_frames[i] = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
        
        return stego_frames
    
    def _encode_frame_dct(self, frame: np.ndarray, bits: List[int]) -> np.ndarray:
        """
        Hide bits in a single frame using DCT.
        
        Args:
            frame: Video frame in YCrCb color space
            bits: Bits to hide
            
        Returns:
            np.ndarray: Modified frame with hidden bits
        """
        # Create a copy of the frame
        stego = frame.copy()
        
        # Get frame dimensions
        height, width = stego.shape[:2]
        
        # Work with Y (luminance) channel only
        y_channel = stego[:, :, 0].astype(np.float32)
        
        # Calculate number of 8x8 blocks
        blocks_h = height // 8
        blocks_w = width // 8
        
        # Calculate capacity
        usable_coefficients = min(len(self.coefficient_masks), 20)  # Limit for quality
        capacity = blocks_h * blocks_w * usable_coefficients * self.data_depth
        
        if len(bits) > capacity:
            logger.warning(f"Too many bits ({len(bits)}) for frame capacity ({capacity})")
            # Truncate bits
            bits = bits[:capacity]
        
        # Generate random block indices
        all_blocks = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                for coef_idx in range(usable_coefficients):
                    for d in range(self.data_depth):
                        all_blocks.append((i, j, coef_idx, d))
        
        # Shuffle indices for security
        random.shuffle(all_blocks)
        
        # Limit to the number of bits we need to hide
        blocks = all_blocks[:len(bits)]
        
        # Hide bits
        for bit_idx, (block_i, block_j, coef_idx, depth) in enumerate(blocks):
            if bit_idx >= len(bits):
                break
            
            # Extract 8x8 block
            block = y_channel[block_i*8:(block_i+1)*8, block_j*8:(block_j+1)*8]
            
            # Apply DCT
            dct_block = cv2.dct(block)
            
            # Get coefficient position
            coef_i, coef_j = self.coefficient_masks[coef_idx]
            
            # Check if coefficient position is valid
            if coef_i >= 8 or coef_j >= 8:
                continue
            
            # Modify coefficient
            coef = dct_block[coef_i, coef_j]
            
            # Quantize coefficient
            # Typical quantization step in JPEG is around 10-20 for mid-frequencies
            quant_step = 15
            quantized = round(coef / quant_step)
            
            # Modify least significant bit based on message
            if bits[bit_idx] == 0:
                quantized = (quantized // 2) * 2  # Make even
            else:
                quantized = (quantized // 2) * 2 + 1  # Make odd
            
            # Update coefficient
            dct_block[coef_i, coef_j] = quantized * quant_step
            
            # Apply inverse DCT
            block = cv2.idct(dct_block)
            
            # Update Y channel
            y_channel[block_i*8:(block_i+1)*8, block_j*8:(block_j+1)*8] = block
        
        # Update Y channel in frame
        stego[:, :, 0] = y_channel
        
        return stego


class VideoTemporalEncoder(BaseEncoder):
    """
    Encoder for hiding messages using temporal patterns between frames.
    
    This encoder modifies brightness or motion patterns between consecutive
    frames to encode the message bits.
    """
    
    def __init__(self, data_depth: int = 1, cuda: bool = True):
        """
        Initialize the Temporal encoder.
        
        Args:
            data_depth: Number of bits to hide per frame transition
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.data_depth = 1  # Temporal methods typically use 1 bit per transition
        self.cuda = cuda
    
    def forward(self, frames: List[np.ndarray], message_bits: List[int]) -> List[np.ndarray]:
        """
        Hide a message using temporal patterns between video frames.
        
        Args:
            frames: List of video frames
            message_bits: Bits to hide
            
        Returns:
            list: Modified frames with hidden message
        """
        if not frames or not message_bits or len(frames) < 2:
            return frames
        
        # Create a copy of frames to avoid modifying originals
        stego_frames = [frame.copy() for frame in frames]
        
        # Calculate total capacity (one bit per frame transition)
        total_capacity = len(stego_frames) - 1
        
        if len(message_bits) > total_capacity:
            logger.warning(f"Message size ({len(message_bits)} bits) exceeds capacity ({total_capacity} bits)")
            # Truncate message
            message_bits = message_bits[:total_capacity]
        
        # Process each bit by modifying brightness between consecutive frames
        for i, bit in enumerate(message_bits):
            if i + 1 >= len(stego_frames):
                break
            
            # Get consecutive frames
            frame1 = stego_frames[i]
            frame2 = stego_frames[i + 1]
            
            # Modify frame2 based on the bit
            stego_frames[i + 1] = self._encode_temporal_bit(frame1, frame2, bit)
        
        return stego_frames
    
    def _encode_temporal_bit(self, frame1: np.ndarray, frame2: np.ndarray, bit: int) -> np.ndarray:
        """
        Encode a single bit by modifying the brightness relation between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            bit: Bit to encode (0 or 1)
            
        Returns:
            np.ndarray: Modified second frame
        """
        # Create a copy of the second frame
        modified_frame = frame2.copy()
        
        # Convert frames to HSV for easier brightness manipulation
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2HSV)
        
        # Calculate average brightness of each frame
        brightness1 = np.mean(hsv1[:, :, 2])
        brightness2 = np.mean(hsv2[:, :, 2])
        
        # Determine the required adjustment
        if bit == 0:
            # Make second frame darker than first
            if brightness2 >= brightness1:
                # Calculate adjustment factor
                factor = 0.9  # Reduce by 10%
                adjustment = np.clip(hsv2[:, :, 2] * factor, 0, 255).astype(np.uint8)
                hsv2[:, :, 2] = adjustment
        else:  # bit == 1
            # Make second frame brighter than first
            if brightness2 <= brightness1:
                # Calculate adjustment factor
                factor = 1.1  # Increase by 10%
                adjustment = np.clip(hsv2[:, :, 2] * factor, 0, 255).astype(np.uint8)
                hsv2[:, :, 2] = adjustment
        
        # Convert back to BGR
        modified_frame = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        
        return modified_frame


class VideoNeuralEncoder(BaseEncoder):
    """
    Encoder for hiding messages in video frames using neural image steganography.
    
    This encoder leverages neural image steganography models to hide data
    in key frames of the video.
    """
    
    def __init__(self, data_depth: int = 1, image_encoder: Optional[Any] = None, cuda: bool = True):
        """
        Initialize the Neural encoder.
        
        Args:
            data_depth: Number of bits to hide per pixel component
            image_encoder: Optional ImageStegoEncoder for neural method
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.data_depth = data_depth
        self.cuda = cuda
        
        # Initialize or use provided image encoder
        self.image_encoder = image_encoder
        if self.image_encoder is None:
            try:
                from stegoai.models.image.encoders import get_image_encoder
                self.image_encoder = get_image_encoder(
                    architecture='dense',
                    data_depth=data_depth,
                    cuda=cuda
                )
            except ImportError:
                logger.warning("Could not import image encoder, using fallback")
                self.image_encoder = None
    
    def forward(self, frames: List[np.ndarray], message_bits: List[int]) -> List[np.ndarray]:
        """
        Hide a message in video frames using neural image steganography.
        
        Args:
            frames: List of video frames
            message_bits: Bits to hide
            
        Returns:
            list: Modified frames with hidden message
        """
        if not frames or not message_bits:
            return frames
        
        # If image encoder is not available, fall back to LSB
        if self.image_encoder is None:
            logger.warning("Using fallback LSB encoder instead of neural")
            fallback = VideoFrameLSBEncoder(data_depth=self.data_depth, cuda=self.cuda)
            return fallback.forward(frames, message_bits)
        
        # Create a copy of frames to avoid modifying originals
        stego_frames = [frame.copy() for frame in frames]
        
        # Calculate capacity per frame (estimate)
        # This depends on the specific neural model
        frame_capacity = []
        for frame in stego_frames:
            height, width = frame.shape[:2]
            # Estimate capacity based on resolution
            # This is an approximation and should be adjusted based on the actual model
            capacity = int(height * width * 0.05)  # Conservative estimate: 0.05 bits per pixel
            frame_capacity.append(capacity)
        
        total_capacity = sum(frame_capacity)
        
        if len(message_bits) > total_capacity:
            logger.warning(f"Message size ({len(message_bits)} bits) exceeds capacity ({total_capacity} bits)")
            # Truncate message
            message_bits = message_bits[:total_capacity]
        
        # Select key frames for hiding data
        # Strategy: Use frames with highest capacity until message is hidden
        frame_indices = sorted(range(len(stego_frames)), key=lambda i: frame_capacity[i], reverse=True)
        
        # Calculate how many frames we need
        bits_remaining = len(message_bits)
        frames_needed = 0
        
        for i in frame_indices:
            bits_remaining -= frame_capacity[i]
            frames_needed += 1
            if bits_remaining <= 0:
                break
        
        frame_indices = frame_indices[:frames_needed]
        
        # Distribute message across selected frames
        start_bit = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx in frame_indices:
                capacity = frame_capacity[idx]
                end_bit = min(start_bit + capacity, len(message_bits))
                
                # Skip if no bits to hide
                if start_bit >= end_bit:
                    continue
                
                frame_bits = message_bits[start_bit:end_bit]
                
                # Create temporary files
                cover_path = os.path.join(temp_dir, f"cover_{idx}.png")
                stego_path = os.path.join(temp_dir, f"stego_{idx}.png")
                
                # Save frame to temporary file
                cv2.imwrite(cover_path, stego_frames[idx])
                
                # Apply neural steganography
                try:
                    self._hide_bits_in_frame(cover_path, stego_path, frame_bits)
                    
                    # Read back the steganographic frame
                    stego_frame = cv2.imread(stego_path)
                    if stego_frame is not None:
                        stego_frames[idx] = stego_frame
                except Exception as e:
                    logger.error(f"Error hiding data in frame {idx}: {e}")
                
                # Move to next chunk
                start_bit = end_bit
        
        return stego_frames
    
    def _hide_bits_in_frame(self, cover_path: str, stego_path: str, bits: List[int]) -> None:
        """
        Use neural image steganography to hide bits in a frame.
        
        Args:
            cover_path: Path to cover image
            stego_path: Path to output stego image
            bits: Bits to hide
        """
        # Convert bits to binary string
        binary_message = ''.join(str(bit) for bit in bits)
        
        # Use image encoder to hide message
        try:
            if isinstance(self.image_encoder, ImageStegoEncoder):
                # Direct access to encoder
                cover_tensor = self.image_encoder.load_image(cover_path)
                stego_tensor = self.image_encoder(cover_tensor, binary_message)
                self.image_encoder.save_image(stego_tensor, stego_path)
            else:
                # Use encoder through higher-level API
                self.image_encoder.encode(cover_path, stego_path, binary_message)
        except Exception as e:
            logger.error(f"Error in neural encoding: {e}")
            # Copy original as fallback
            import shutil
            shutil.copy(cover_path, stego_path)


def get_video_encoder(method: str = 'frame_lsb', **kwargs) -> BaseEncoder:
    """
    Factory function to get the appropriate video encoder.
    
    Args:
        method: Steganography method ('frame_lsb', 'frame_dct', 'temporal', 'neural')
        **kwargs: Additional arguments to pass to the encoder
        
    Returns:
        BaseEncoder: Appropriate encoder for the method
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'frame_lsb':
        return VideoFrameLSBEncoder(**kwargs)
    elif method == 'frame_dct':
        return VideoFrameDCTEncoder(**kwargs)
    elif method == 'temporal':
        return VideoTemporalEncoder(**kwargs)
    elif method == 'neural':
        return VideoNeuralEncoder(**kwargs)
    else:
        raise ValueError(f"Unsupported video steganography method: {method}")