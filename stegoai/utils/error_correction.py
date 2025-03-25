# -*- coding: utf-8 -*-
"""
Error correction utilities for Stego-AI.

This module provides functions for adding redundancy and error correction
to hidden data, making steganography more robust against modifications.
"""

import logging
import struct
import hashlib
import random
from typing import List, Tuple, Union, Optional

import numpy as np
from reedsolo import RSCodec, ReedSolomonError

# Set up logging
logger = logging.getLogger(__name__)

# Default Reed-Solomon parameters
DEFAULT_ECC_SYMBOLS = 16  # Number of error correction symbols


def add_checksum(data: bytes) -> bytes:
    """
    Add a simple checksum to data.
    
    Args:
        data: Input data
        
    Returns:
        bytes: Data with checksum appended
    """
    # Calculate CRC-32 checksum
    checksum = binascii.crc32(data) & 0xFFFFFFFF
    
    # Append checksum to data
    return data + struct.pack('<I', checksum)


def verify_checksum(data: bytes) -> Tuple[bool, bytes]:
    """
    Verify checksum and extract original data.
    
    Args:
        data: Input data with checksum
        
    Returns:
        tuple: (is_valid, original_data)
    """
    if len(data) < 4:
        return False, data
    
    # Extract data and checksum
    original_data = data[:-4]
    checksum = struct.unpack('<I', data[-4:])[0]
    
    # Calculate checksum of original data
    calculated_checksum = binascii.crc32(original_data) & 0xFFFFFFFF
    
    # Verify checksum
    is_valid = checksum == calculated_checksum
    
    return is_valid, original_data


def encode_reed_solomon(data: bytes, ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> bytes:
    """
    Encode data using Reed-Solomon error correction.
    
    Args:
        data: Input data
        ecc_symbols: Number of error correction symbols
        
    Returns:
        bytes: Encoded data with error correction
    """
    try:
        # Create Reed-Solomon codec
        rsc = RSCodec(ecc_symbols)
        
        # Encode data
        encoded_data = rsc.encode(data)
        
        return encoded_data
    
    except Exception as e:
        logger.error(f"Error in Reed-Solomon encoding: {e}")
        # Return original data if encoding fails
        return data


def decode_reed_solomon(data: bytes, ecc_symbols: int = DEFAULT_ECC_SYMBOLS) -> Tuple[bool, bytes]:
    """
    Decode data using Reed-Solomon error correction.
    
    Args:
        data: Encoded data with error correction
        ecc_symbols: Number of error correction symbols
        
    Returns:
        tuple: (is_valid, decoded_data)
    """
    try:
        # Create Reed-Solomon codec
        rsc = RSCodec(ecc_symbols)
        
        # Decode data
        decoded_data = rsc.decode(data)
        
        return True, decoded_data
    
    except ReedSolomonError as e:
        logger.warning(f"Reed-Solomon decoding failed: {e}")
        return False, data
    
    except Exception as e:
        logger.error(f"Error in Reed-Solomon decoding: {e}")
        return False, data


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Convert a list of bits to bytes.
    
    Args:
        bits: List of bits (0s and 1s)
        
    Returns:
        bytes: Byte representation of bits
    """
    # Ensure bit count is a multiple of 8
    padded_bits = bits.copy()
    while len(padded_bits) % 8 != 0:
        padded_bits.append(0)
    
    # Convert bits to bytes
    result = bytearray()
    for i in range(0, len(padded_bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(padded_bits):
                byte = (byte << 1) | padded_bits[i + j]
        result.append(byte)
    
    return bytes(result)


def bytes_to_bits(data: bytes) -> List[int]:
    """
    Convert bytes to a list of bits.
    
    Args:
        data: Input bytes
        
    Returns:
        list: List of bits (0s and 1s)
    """
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    
    return bits


def encode_repetition(bits: List[int], repeat: int = 3) -> List[int]:
    """
    Encode bits using repetition code.
    
    Args:
        bits: Input bits
        repeat: Number of times to repeat each bit
        
    Returns:
        list: Encoded bits
    """
    encoded_bits = []
    for bit in bits:
        encoded_bits.extend([bit] * repeat)
    
    return encoded_bits


def decode_repetition(encoded_bits: List[int], repeat: int = 3) -> List[int]:
    """
    Decode bits using repetition code.
    
    Args:
        encoded_bits: Encoded bits
        repeat: Number of times each bit was repeated
        
    Returns:
        list: Decoded bits
    """
    if len(encoded_bits) % repeat != 0:
        # Pad with zeros to make length a multiple of repeat
        encoded_bits = encoded_bits + [0] * (repeat - len(encoded_bits) % repeat)
    
    decoded_bits = []
    for i in range(0, len(encoded_bits), repeat):
        # Take majority vote for each group of repeated bits
        group = encoded_bits[i:i+repeat]
        bit = 1 if sum(group) > repeat / 2 else 0
        decoded_bits.append(bit)
    
    return decoded_bits


def encode_hamming(bits: List[int]) -> List[int]:
    """
    Encode bits using Hamming code.
    
    Args:
        bits: Input bits
        
    Returns:
        list: Encoded bits with error correction
    """
    # Process 4 bits at a time
    encoded_bits = []
    
    for i in range(0, len(bits), 4):
        # Get 4 data bits, pad with zeros if needed
        data_bits = bits[i:i+4]
        while len(data_bits) < 4:
            data_bits.append(0)
        
        # Calculate parity bits
        p1 = (data_bits[0] ^ data_bits[1] ^ data_bits[3]) & 1
        p2 = (data_bits[0] ^ data_bits[2] ^ data_bits[3]) & 1
        p3 = (data_bits[1] ^ data_bits[2] ^ data_bits[3]) & 1
        
        # Create encoded sequence [p1, p2, d1, p3, d2, d3, d4]
        encoded_group = [p1, p2, data_bits[0], p3, data_bits[1], data_bits[2], data_bits[3]]
        encoded_bits.extend(encoded_group)
    
    return encoded_bits


def decode_hamming(encoded_bits: List[int]) -> List[int]:
    """
    Decode bits using Hamming code, with error correction.
    
    Args:
        encoded_bits: Encoded bits with error correction
        
    Returns:
        list: Decoded bits
    """
    # Process 7 bits at a time
    decoded_bits = []
    
    for i in range(0, len(encoded_bits), 7):
        # Get 7 encoded bits, pad with zeros if needed
        group = encoded_bits[i:i+7]
        while len(group) < 7:
            group.append(0)
        
        # Extract parity and data bits
        p1, p2, d1, p3, d2, d3, d4 = group
        
        # Calculate syndrome
        s1 = (p1 ^ d1 ^ d2 ^ d4) & 1
        s2 = (p2 ^ d1 ^ d3 ^ d4) & 1
        s3 = (p3 ^ d2 ^ d3 ^ d4) & 1
        
        syndrome = (s3 << 2) | (s2 << 1) | s1
        
        # Perform error correction
        if syndrome != 0:
            # Flip the bit at the error position
            error_pos = syndrome
            if error_pos - 1 < len(group):
                group[error_pos - 1] ^= 1
        
        # Extract corrected data bits
        d1, d2, d3, d4 = group[2], group[4], group[5], group[6]
        decoded_bits.extend([d1, d2, d3, d4])
    
    return decoded_bits


def interleave_bits(bits: List[int], block_size: int = 8) -> List[int]:
    """
    Interleave bits to protect against burst errors.
    
    Args:
        bits: Input bits
        block_size: Size of interleaving blocks
        
    Returns:
        list: Interleaved bits
    """
    # Ensure bit count is a multiple of block_size
    padded_bits = bits.copy()
    while len(padded_bits) % block_size != 0:
        padded_bits.append(0)
    
    # Determine dimensions of the interleaving matrix
    num_blocks = len(padded_bits) // block_size
    
    # Create matrix (filling by rows)
    matrix = []
    for i in range(num_blocks):
        matrix.append(padded_bits[i * block_size:(i + 1) * block_size])
    
    # Read out matrix by columns
    interleaved_bits = []
    for j in range(block_size):
        for i in range(num_blocks):
            interleaved_bits.append(matrix[i][j])
    
    return interleaved_bits


def deinterleave_bits(interleaved_bits: List[int], block_size: int = 8) -> List[int]:
    """
    Deinterleave bits.
    
    Args:
        interleaved_bits: Interleaved bits
        block_size: Size of interleaving blocks
        
    Returns:
        list: Deinterleaved bits
    """
    # Ensure bit count is a multiple of block_size
    padded_bits = interleaved_bits.copy()
    while len(padded_bits) % block_size != 0:
        padded_bits.append(0)
    
    # Determine dimensions of the interleaving matrix
    num_blocks = len(padded_bits) // block_size
    
    # Create matrix (filling by columns)
    matrix = [[0 for _ in range(block_size)] for _ in range(num_blocks)]
    for j in range(block_size):
        for i in range(num_blocks):
            idx = j * num_blocks + i
            if idx < len(padded_bits):
                matrix[i][j] = padded_bits[idx]
    
    # Read out matrix by rows
    deinterleaved_bits = []
    for i in range(num_blocks):
        deinterleaved_bits.extend(matrix[i])
    
    return deinterleaved_bits


def encode_robust_message(message: str, ecc_level: str = 'medium') -> List[int]:
    """
    Encode a message with robust error correction.
    
    Args:
        message: Message to encode
        ecc_level: Error correction level ('low', 'medium', 'high')
        
    Returns:
        list: Encoded bits
    """
    # Convert message to bytes
    message_bytes = message.encode('utf-8')
    
    # Add checksum
    data_with_checksum = add_checksum(message_bytes)
    
    # Apply Reed-Solomon encoding
    if ecc_level == 'low':
        ecc_symbols = 8
        repeat = 1
    elif ecc_level == 'medium':
        ecc_symbols = 16
        repeat = 2
    else:  # high
        ecc_symbols = 32
        repeat = 3
    
    rs_encoded = encode_reed_solomon(data_with_checksum, ecc_symbols)
    
    # Convert to bits
    bits = bytes_to_bits(rs_encoded)
    
    # Apply Hamming coding for additional protection
    hamming_encoded = encode_hamming(bits)
    
    # Apply interleaving to protect against burst errors
    interleaved = interleave_bits(hamming_encoded)
    
    # Apply repetition coding if needed
    if repeat > 1:
        repeated = encode_repetition(interleaved, repeat)
    else:
        repeated = interleaved
    
    return repeated


def decode_robust_message(encoded_bits: List[int], ecc_level: str = 'medium') -> str:
    """
    Decode a message with robust error correction.
    
    Args:
        encoded_bits: Encoded bits
        ecc_level: Error correction level ('low', 'medium', 'high')
        
    Returns:
        str: Decoded message
    """
    try:
        # Determine parameters based on ECC level
        if ecc_level == 'low':
            ecc_symbols = 8
            repeat = 1
        elif ecc_level == 'medium':
            ecc_symbols = 16
            repeat = 2
        else:  # high
            ecc_symbols = 32
            repeat = 3
        
        # Undo repetition coding if applied
        if repeat > 1:
            derepetitioned = decode_repetition(encoded_bits, repeat)
        else:
            derepetitioned = encoded_bits
        
        # Undo interleaving
        deinterleaved = deinterleave_bits(derepetitioned)
        
        # Undo Hamming coding
        hamming_decoded = decode_hamming(deinterleaved)
        
        # Convert bits to bytes
        data_bytes = bits_to_bytes(hamming_decoded)
        
        # Apply Reed-Solomon decoding
        success, rs_decoded = decode_reed_solomon(data_bytes, ecc_symbols)
        
        if not success:
            logger.warning("Reed-Solomon decoding failed")
            # Continue anyway with the partially corrected data
        
        # Verify checksum
        valid, original_data = verify_checksum(rs_decoded)
        
        if not valid:
            logger.warning("Checksum verification failed")
            # Continue anyway with the data we have
        
        # Decode as UTF-8
        message = original_data.decode('utf-8', errors='replace')
        
        return message
    
    except Exception as e:
        logger.error(f"Error in robust message decoding: {e}")
        # Try a direct conversion as last resort
        try:
            # Convert bits to bytes directly
            data_bytes = bits_to_bytes(encoded_bits)
            # Attempt to decode as UTF-8
            return data_bytes.decode('utf-8', errors='replace')
        except:
            return ""


def add_watermark(bits: List[int], watermark: str) -> List[int]:
    """
    Add a watermark signature to hidden data.
    
    Args:
        bits: Input bits
        watermark: Watermark text
        
    Returns:
        list: Bits with watermark
    """
    # Calculate watermark hash
    hash_obj = hashlib.sha256(watermark.encode('utf-8'))
    hash_value = hash_obj.digest()
    
    # Convert hash to bits
    hash_bits = bytes_to_bits(hash_value)[:32]  # Use first 32 bits of hash
    
    # Add watermark bits at the beginning
    watermarked_bits = hash_bits + bits
    
    return watermarked_bits


def verify_watermark(bits: List[int], watermark: str) -> Tuple[bool, List[int]]:
    """
    Verify watermark and extract original bits.
    
    Args:
        bits: Watermarked bits
        watermark: Expected watermark text
        
    Returns:
        tuple: (is_valid, original_bits)
    """
    if len(bits) < 32:
        return False, bits
    
    # Extract watermark hash
    watermark_bits = bits[:32]
    original_bits = bits[32:]
    
    # Calculate expected hash
    hash_obj = hashlib.sha256(watermark.encode('utf-8'))
    hash_value = hash_obj.digest()
    
    # Convert hash to bits
    expected_bits = bytes_to_bits(hash_value)[:32]
    
    # Verify watermark
    is_valid = watermark_bits == expected_bits
    
    return is_valid, original_bits


def scramble_bits(bits: List[int], seed: int) -> List[int]:
    """
    Scramble bits using a pseudo-random permutation.
    
    Args:
        bits: Input bits
        seed: Random seed for scrambling
        
    Returns:
        list: Scrambled bits
    """
    # Create a copy of the bits
    scrambled = bits.copy()
    
    # Initialize random generator with seed
    rng = random.Random(seed)
    
    # Perform Fisher-Yates shuffle
    for i in range(len(scrambled) - 1, 0, -1):
        j = rng.randint(0, i)
        scrambled[i], scrambled[j] = scrambled[j], scrambled[i]
    
    return scrambled


def unscramble_bits(scrambled_bits: List[int], seed: int) -> List[int]:
    """
    Unscramble bits using the same pseudo-random permutation.
    
    Args:
        scrambled_bits: Scrambled bits
        seed: Random seed used for scrambling
        
    Returns:
        list: Original bit order
    """
    # Create permutation map
    N = len(scrambled_bits)
    perm = list(range(N))
    
    # Initialize random generator with seed
    rng = random.Random(seed)
    
    # Perform Fisher-Yates shuffle on permutation map
    for i in range(N - 1, 0, -1):
        j = rng.randint(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    
    # Create inverse permutation
    inv_perm = [0] * N
    for i in range(N):
        inv_perm[perm[i]] = i
    
    # Apply inverse permutation
    unscrambled = [0] * N
    for i in range(N):
        unscrambled[inv_perm[i]] = scrambled_bits[i]
    
    return unscrambled


def encrypt_bits(bits: List[int], key: str) -> List[int]:
    """
    Encrypt bits using a simple XOR cipher with key expansion.
    
    Args:
        bits: Input bits
        key: Encryption key
        
    Returns:
        list: Encrypted bits
    """
    # Convert key to bytes
    key_bytes = key.encode('utf-8')
    
    # Convert key to bits
    key_bits = bytes_to_bits(key_bytes)
    
    # Expand key to match length of input
    expanded_key = []
    while len(expanded_key) < len(bits):
        expanded_key.extend(key_bits)
    
    # Trim to match input length
    expanded_key = expanded_key[:len(bits)]
    
    # Apply XOR
    encrypted = [(b ^ k) & 1 for b, k in zip(bits, expanded_key)]
    
    return encrypted


def decrypt_bits(encrypted_bits: List[int], key: str) -> List[int]:
    """
    Decrypt bits using a simple XOR cipher with key expansion.
    
    Args:
        encrypted_bits: Encrypted bits
        key: Encryption key
        
    Returns:
        list: Decrypted bits
    """
    # XOR is its own inverse, so we can use the same function
    return encrypt_bits(encrypted_bits, key)