# -*- coding: utf-8 -*-
"""
Video steganography decoders for Stego-AI.

This module implements decoder architectures for extracting messages
from videos using different steganographic techniques.
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

from stegoai.models.base import BaseDecoder
from stegoai.models.image.decoders import ImageStegoDecoder
from stegoai.utils.text_utils import bits_to_bytearray, bytearray_to_text

# Set up logging
logger = logging.getLogger(__name__)


class VideoFrameLSBDecoder(BaseDecoder):
    """
    Decoder for extracting messages from video frames using LSB steganography.
    
    This decoder extracts data from the least significant bits of pixel values
    across selected frames of a video.
    """
    
    def __init__(self, data_depth: int = 1, cuda: bool = True):
        """
        Initialize the Frame LSB decoder.
        
        Args:
            data_depth: Number of bits hidden per pixel component
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
    
    def forward(self, frames: List[np.ndarray], max_bits: Optional[int] = None) -> List[int]:
        """
        Extract a message from video frames using LSB steganography.
        
        Args:
            frames: List of video frames
            max_bits: Maximum number of bits to extract
            
        Returns:
            list: Extracted message bits
        """
        if not frames:
            return []
        
        # Initialize extracted bits
        extracted_bits = []
        
        # Process each frame
        for frame in frames:
            frame_bits = self._decode_frame(frame, max_bits - len(extracted_bits) if max_bits else None)
            extracted_bits.extend(frame_bits)
            
            # Check if we've reached max_bits
            if max_bits and len(extracted_bits) >= max_bits:
                break
            
            # Check for termination sequence (32 zeros)
            if len(extracted_bits) >= 32 and all(bit == 0 for bit in extracted_bits[-32:]):
                # Remove termination sequence
                extracted_bits = extracted_bits[:-32]
                break
        
        return extracted_bits
    
    def _decode_frame(self, frame: np.ndarray, max_bits: Optional[int] = None) -> List[int]:
        """
        Extract bits from a single frame.
        
        Args:
            frame: Video frame
            max_bits: Maximum number of bits to extract
            
        Returns:
            list: Extracted bits
        """
        # Get frame dimensions
        height, width, channels = frame.shape
        
        # Calculate capacity
        capacity = height * width * channels * self.data_depth
        max_to_extract = min(capacity, max_bits) if max_bits else capacity
        
        # Generate pixel indices (must match encoder's random pattern)
        # For decoder, we need to use the same random seed as encoder
        # This is a simplified implementation - real systems would use
        # a seeded PRNG based on a shared key
        random.seed(42)  # Fixed seed for reproducibility
        
        all_indices = []
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    for d in range(self.data_depth):
                        all_indices.append((y, x, c, d))
        
        # Shuffle indices (same shuffle as encoder)
        random.shuffle(all_indices)
        
        # Limit to max_to_extract
        indices = all_indices[:max_to_extract]
        
        # Extract bits
        bits = []
        for y, x, c, d in indices:
            # Get pixel value
            pixel = frame[y, x, c]
            
            # Extract the dth bit
            bit = (pixel >> d) & 1
            bits.append(bit)
            
            # Check for termination sequence
            if len(bits) >= 32 and all(bit == 0 for bit in bits[-32:]):
                return bits
        
        return bits


class VideoFrameDCTDecoder(BaseDecoder):
    """
    Decoder for extracting messages from video frames using DCT steganography.
    
    This decoder extracts data from the DCT coefficients of frames, decoding
    information hidden in the frequency domain.
    """
    
    def __init__(self, data_depth: int = 1, cuda: bool = True):
        """
        Initialize the Frame DCT decoder.
        
        Args:
            data_depth: Number of bits hidden per coefficient
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
    
    def forward(self, frames: List[np.ndarray], max_bits: Optional[int] = None) -> List[int]:
        """
        Extract a message from video frames using DCT steganography.
        
        Args:
            frames: List of video frames
            max_bits: Maximum number of bits to extract
            
        Returns:
            list: Extracted message bits
        """
        if not frames:
            return []
        
        # Initialize extracted bits
        extracted_bits = []
        
        # Convert frames to YCrCb color space
        ycrcb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) for frame in frames]
        
        # Process each frame
        for frame in ycrcb_frames:
            frame_bits = self._decode_frame_dct(frame, max_bits - len(extracted_bits) if max_bits else None)
            extracted_bits.extend(frame_bits)
            
            # Check if we've reached max_bits
            if max_bits and len(extracted_bits) >= max_bits:
                break
            
            # Check for termination sequence (32 zeros)
            if len(extracted_bits) >= 32 and all(bit == 0 for bit in extracted_bits[-32:]):
                # Remove termination sequence
                extracted_bits = extracted_bits[:-32]
                break
        
        return extracted_bits
    
    def _decode_frame_dct(self, frame: np.ndarray, max_bits: Optional[int] = None) -> List[int]:
        """
        Extract bits from a single frame using DCT.
        
        Args:
            frame: Video frame in YCrCb color space
            max_bits: Maximum number of bits to extract
            
        Returns:
            list: Extracted bits
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Work with Y (luminance) channel only
        y_channel = frame[:, :, 0].astype(np.float32)
        
        # Calculate number of 8x8 blocks
        blocks_h = height // 8
        blocks_w = width // 8
        
        # Calculate capacity
        usable_coefficients = min(len(self.coefficient_masks), 20)  # Limit for quality
        capacity = blocks_h * blocks_w * usable_coefficients * self.data_depth
        max_to_extract = min(capacity, max_bits) if max_bits else capacity
        
        # Generate random block indices (must match encoder's pattern)
        # For decoder, we need to use the same random seed as encoder
        random.seed(42)  # Fixed seed for reproducibility
        
        all_blocks = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                for coef_idx in range(usable_coefficients):
                    for d in range(self.data_depth):
                        all_blocks.append((i, j, coef_idx, d))
        
        # Shuffle indices (same shuffle as encoder)
        random.shuffle(all_blocks)
        
        # Limit to max_to_extract
        blocks = all_blocks[:max_to_extract]
        
        # Extract bits
        bits = []
        for block_i, block_j, coef_idx, depth in blocks:
            # Extract 8x8 block
            block = y_channel[block_i*8:(block_i+1)*8, block_j*8:(block_j+1)*8]
            
            # Apply DCT
            dct_block = cv2.dct(block)
            
            # Get coefficient position
            coef_i, coef_j = self.coefficient_masks[coef_idx]
            
            # Check if coefficient position is valid
            if coef_i >= 8 or coef_j >= 8:
                continue
            
            # Get coefficient
            coef = dct_block[coef_i, coef_j]
            
            # Quantize coefficient
            # Typical quantization step in JPEG is around 10-20 for mid-frequencies
            quant_step = 15
            quantized = round(coef / quant_step)
            
            # Extract bit (LSB of quantized coefficient)
            bit = quantized % 2
            bits.append(bit)
            
            # Check for termination sequence
            if len(bits) >= 32 and all(bit == 0 for bit in bits[-32:]):
                return bits
        
        return bits


class VideoTemporalDecoder(BaseDecoder):
    """
    Decoder for extracting messages hidden using temporal patterns.
    
    This decoder analyzes brightness or motion patterns between consecutive
    frames to decode the hidden message bits.
    """
    
    def __init__(self, data_depth: int = 1, cuda: bool = True):
        """
        Initialize the Temporal decoder.
        
        Args:
            data_depth: Number of bits hidden per frame transition
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.data_depth = 1  # Temporal methods typically use 1 bit per transition
        self.cuda = cuda
    
    def forward(self, frames: List[np.ndarray], max_bits: Optional[int] = None) -> List[int]:
        """
        Extract a message hidden using temporal patterns between video frames.
        
        Args:
            frames: List of video frames
            max_bits: Maximum number of bits to extract
            
        Returns:
            list: Extracted message bits
        """
        if not frames or len(frames) < 2:
            return []
        
        # Initialize extracted bits
        extracted_bits = []
        
        # Process consecutive frame pairs
        for i in range(len(frames) - 1):
            if max_bits and len(extracted_bits) >= max_bits:
                break
            
            # Get consecutive frames
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Extract bit
            bit = self._decode_temporal_bit(frame1, frame2)
            extracted_bits.append(bit)
            
            # Check for termination sequence (32 zeros)
            if len(extracted_bits) >= 32 and all(bit == 0 for bit in extracted_bits[-32:]):
                # Remove termination sequence
                extracted_bits = extracted_bits[:-32]
                break
        
        return extracted_bits
    
    def _decode_temporal_bit(self, frame1: np.ndarray, frame2: np.ndarray) -> int:
        """
        Decode a single bit from the brightness relation between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            int: Extracted bit (0 or 1)
        """
        # Convert frames to HSV for easier brightness manipulation
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
        # Calculate average brightness of each frame
        brightness1 = np.mean(hsv1[:, :, 2])
        brightness2 = np.mean(hsv2[:, :, 2])
        
        # Determine the bit based on brightness relation
        if brightness2 < brightness1:
            return 0  # Second frame darker = bit 0
        else:
            return 1  # Second frame brighter = bit 1


class VideoNeuralDecoder(BaseDecoder):
    """
    Decoder for extracting messages from video frames using neural image steganography.
    
    This decoder uses neural image steganography models to extract data
    from key frames of the video.
    """
    
    def __init__(self, data_depth: int = 1, image_decoder: Optional[Any] = None, cuda: bool = True):
        """
        Initialize the Neural decoder.
        
        Args:
            data_depth: Number of bits hidden per pixel component
            image_decoder: Optional ImageStegoDecoder for neural method
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.data_depth = data_depth
        self.cuda = cuda
        
        # Initialize or use provided image decoder
        self.image_decoder = image_decoder
        if self.image_decoder is None:
            try:
                from stegoai.models.image.decoders import get_image_decoder
                self.image_decoder = get_image_decoder(
                    architecture='dense',
                    data_depth=data_depth,
                    cuda=cuda
                )
            except ImportError:
                logger.warning("Could not import image decoder, using fallback")
                self.image_decoder = None
    
    def forward(self, frames: List[np.ndarray], max_bits: Optional[int] = None) -> List[int]:
        """
        Extract a message from video frames using neural image steganography.
        
        Args:
            frames: List of video frames
            max_bits: Maximum number of bits to extract
            
        Returns:
            list: Extracted message bits
        """
        if not frames:
            return []
        
        # If image decoder is not available, fall back to LSB
        if self.image_decoder is None:
            logger.warning("Using fallback LSB decoder instead of neural")
            fallback = VideoFrameLSBDecoder(data_depth=self.data_depth, cuda=self.cuda)
            return fallback.forward(frames, max_bits)
        
        # Initialize extracted bits
        all_bits = []
        
        # Create temporary directory for processing frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each frame
            for i, frame in enumerate(frames):
                # Skip if we've already found enough bits
                if max_bits and len(all_bits) >= max_bits:
                    break
                
                # Check for termination sequence
                if len(all_bits) >= 32 and all(bit == 0 for bit in all_bits[-32:]):
                    # Remove termination sequence
                    all_bits = all_bits[:-32]
                    break
                
                # Save frame to temporary file
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame)
                
                # Extract bits using neural model
                try:
                    frame_bits = self._extract_bits_from_frame(frame_path)
                    
                    if frame_bits:
                        all_bits.extend(frame_bits)
                        
                        # Check for termination sequence
                        if len(frame_bits) >= 32 and all(bit == 0 for bit in frame_bits[-32:]):
                            # Remove termination sequence from all_bits
                            all_bits = all_bits[:-32]
                            break
                except Exception as e:
                    logger.warning(f"Error extracting from frame {i}: {e}")
        
        # Limit to max_bits if specified
        if max_bits and len(all_bits) > max_bits:
            all_bits = all_bits[:max_bits]
        
        return all_bits
    
    def _extract_bits_from_frame(self, frame_path: str) -> List[int]:
        """
        Use neural image steganography to extract bits from a frame.
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            list: Extracted bits
        """
        try:
            if isinstance(self.image_decoder, ImageStegoDecoder):
                # Direct access to decoder
                stego_tensor = self.image_decoder.load_image(frame_path)
                binary_message = self.image_decoder(stego_tensor)
            else:
                # Use decoder through higher-level API
                binary_message = self.image_decoder.decode(frame_path)
            
            # Convert binary string to bits
            bits = [int(bit) for bit in binary_message if bit in '01']
            
            return bits
        except Exception as e:
            logger.error(f"Error in neural decoding: {e}")
            return []


def get_video_decoder(method: str = 'frame_lsb', **kwargs) -> BaseDecoder:
    """
    Factory function to get the appropriate video decoder.
    
    Args:
        method: Steganography method ('frame_lsb', 'frame_dct', 'temporal', 'neural')
        **kwargs: Additional arguments to pass to the decoder
        
    Returns:
        BaseDecoder: Appropriate decoder for the method
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'frame_lsb':
        return VideoFrameLSBDecoder(**kwargs)
    elif method == 'frame_dct':
        return VideoFrameDCTDecoder(**kwargs)
    elif method == 'temporal':
        return VideoTemporalDecoder(**kwargs)
    elif method == 'neural':
        return VideoNeuralDecoder(**kwargs)
    else:
        raise ValueError(f"Unsupported video steganography method: {method}")