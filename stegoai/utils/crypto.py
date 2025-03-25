# -*- coding: utf-8 -*-
"""
Cryptography utilities for Stego-AI.

This module provides functions for encrypting and decrypting data
to be hidden in steganographic carriers.
"""

import os
import logging
import base64
import hashlib
import hmac
from typing import Tuple, Optional, Union

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Set up logging
logger = logging.getLogger(__name__)


def derive_key(password: str, salt: Optional[bytes] = None,
              key_length: int = 32) -> Tuple[bytes, bytes]:
    """
    Derive key from password using PBKDF2.
    
    Args:
        password: Password for key derivation
        salt: Salt for key derivation (generated if None)
        key_length: Length of key in bytes
        
    Returns:
        tuple: (key, salt)
    """
    # Generate salt if not provided
    if salt is None:
        salt = os.urandom(16)
    
    # Create key derivation function
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    
    # Derive key
    key = kdf.derive(password.encode('utf-8'))
    
    return key, salt


def encrypt_aes(data: bytes, password: str, add_auth: bool = True) -> bytes:
    """
    Encrypt data using AES-256-GCM.
    
    Args:
        data: Data to encrypt
        password: Encryption password
        add_auth: Whether to add authentication tag
        
    Returns:
        bytes: Encrypted data (format: salt + iv + [tag] + ciphertext)
    """
    try:
        # Derive key from password
        key, salt = derive_key(password)
        
        # Generate random IV
        iv = os.urandom(12)  # 12 bytes for GCM mode
        
        # Create cipher
        if add_auth:
            # Use GCM mode for authenticated encryption
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
        else:
            # Use CBC mode
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv[:16]),  # CBC needs 16 bytes IV
                backend=default_backend()
            )
        
        # Create encryptor
        encryptor = cipher.encryptor()
        
        # Pad data for CBC mode
        if not add_auth:
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()
        else:
            padded_data = data
        
        # Encrypt data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Get authentication tag for GCM mode
        if add_auth:
            tag = encryptor.tag
            return salt + iv + tag + ciphertext
        else:
            return salt + iv[:16] + ciphertext
    
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise


def decrypt_aes(encrypted_data: bytes, password: str, add_auth: bool = True) -> bytes:
    """
    Decrypt data using AES-256-GCM.
    
    Args:
        encrypted_data: Encrypted data (format: salt + iv + [tag] + ciphertext)
        password: Encryption password
        add_auth: Whether authentication tag is present
        
    Returns:
        bytes: Decrypted data
    """
    try:
        # Extract salt, iv, tag, and ciphertext
        salt = encrypted_data[:16]
        
        if add_auth:
            iv = encrypted_data[16:28]  # 12 bytes for GCM
            tag = encrypted_data[28:44]  # 16 bytes
            ciphertext = encrypted_data[44:]
        else:
            iv = encrypted_data[16:32]  # 16 bytes for CBC
            ciphertext = encrypted_data[32:]
        
        # Derive key from password with the extracted salt
        key, _ = derive_key(password, salt)
        
        # Create cipher
        if add_auth:
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
        else:
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
        
        # Create decryptor
        decryptor = cipher.decryptor()
        
        # Decrypt data
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Unpad data for CBC mode
        if not add_auth:
            unpadder = padding.PKCS7(128).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
        else:
            data = padded_data
        
        return data
    
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise


def encrypt_data(data: bytes, password: str, method: str = 'aes-gcm') -> bytes:
    """
    Encrypt data using the specified method.
    
    Args:
        data: Data to encrypt
        password: Encryption password
        method: Encryption method ('aes-gcm', 'aes-cbc', 'xor')
        
    Returns:
        bytes: Encrypted data
    """
    if method == 'aes-gcm':
        return encrypt_aes(data, password, add_auth=True)
    elif method == 'aes-cbc':
        return encrypt_aes(data, password, add_auth=False)
    elif method == 'xor':
        return xor_encrypt(data, password)
    else:
        raise ValueError(f"Unknown encryption method: {method}")


def decrypt_data(encrypted_data: bytes, password: str, method: str = 'aes-gcm') -> bytes:
    """
    Decrypt data using the specified method.
    
    Args:
        encrypted_data: Encrypted data
        password: Encryption password
        method: Encryption method ('aes-gcm', 'aes-cbc', 'xor')
        
    Returns:
        bytes: Decrypted data
    """
    if method == 'aes-gcm':
        return decrypt_aes(encrypted_data, password, add_auth=True)
    elif method == 'aes-cbc':
        return decrypt_aes(encrypted_data, password, add_auth=False)
    elif method == 'xor':
        return xor_encrypt(encrypted_data, password)  # XOR is its own inverse
    else:
        raise ValueError(f"Unknown encryption method: {method}")


def xor_encrypt(data: bytes, password: str) -> bytes:
    """
    Encrypt data using XOR cipher.
    
    Args:
        data: Data to encrypt
        password: Encryption password
        
    Returns:
        bytes: Encrypted data
    """
    # Derive key
    key, salt = derive_key(password, key_length=32)
    
    # Expand key to match data length
    key_expanded = bytearray()
    for i in range(0, len(data), len(key)):
        key_expanded.extend(key[:min(len(key), len(data) - i)])
    
    # Apply XOR
    result = bytearray()
    for i in range(len(data)):
        result.append(data[i] ^ key_expanded[i])
    
    # Prepend salt for key derivation
    return salt + bytes(result)


def calculate_hmac(data: bytes, key: bytes) -> bytes:
    """
    Calculate HMAC for data authentication.
    
    Args:
        data: Data to authenticate
        key: Authentication key
        
    Returns:
        bytes: HMAC value
    """
    return hmac.new(key, data, hashlib.sha256).digest()


def verify_hmac(data: bytes, hmac_value: bytes, key: bytes) -> bool:
    """
    Verify HMAC for data authentication.
    
    Args:
        data: Data to verify
        hmac_value: HMAC value to check
        key: Authentication key
        
    Returns:
        bool: True if HMAC is valid
    """
    computed_hmac = calculate_hmac(data, key)
    return hmac.compare_digest(computed_hmac, hmac_value)


def encrypt_text(text: str, password: str, method: str = 'aes-gcm') -> str:
    """
    Encrypt text using the specified method.
    
    Args:
        text: Text to encrypt
        password: Encryption password
        method: Encryption method ('aes-gcm', 'aes-cbc', 'xor')
        
    Returns:
        str: Base64-encoded encrypted data
    """
    # Convert text to bytes
    data = text.encode('utf-8')
    
    # Encrypt data
    encrypted = encrypt_data(data, password, method)
    
    # Encode as Base64
    encoded = base64.b64encode(encrypted).decode('ascii')
    
    return encoded


def decrypt_text(encoded: str, password: str, method: str = 'aes-gcm') -> str:
    """
    Decrypt text using the specified method.
    
    Args:
        encoded: Base64-encoded encrypted data
        password: Encryption password
        method: Encryption method ('aes-gcm', 'aes-cbc', 'xor')
        
    Returns:
        str: Decrypted text
    """
    # Decode from Base64
    encrypted = base64.b64decode(encoded)
    
    # Decrypt data
    data = decrypt_data(encrypted, password, method)
    
    # Convert bytes to text
    text = data.decode('utf-8')
    
    return text


def secure_random(size: int) -> bytes:
    """
    Generate cryptographically secure random bytes.
    
    Args:
        size: Number of bytes to generate
        
    Returns:
        bytes: Random bytes
    """
    return os.urandom(size)


def generate_password() -> str:
    """
    Generate a secure random password.
    
    Returns:
        str: Random password
    """
    # Generate 24 random bytes for a 32-character password
    return base64.b64encode(secure_random(24)).decode('ascii')


def hash_password(password: str) -> str:
    """
    Hash password for storage.
    
    Args:
        password: Password to hash
        
    Returns:
        str: Password hash
    """
    # Generate salt
    salt = os.urandom(16)
    
    # Create key derivation function
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    
    # Derive key
    key = kdf.derive(password.encode('utf-8'))
    
    # Combine salt and key
    combined = salt + key
    
    # Encode as Base64
    return base64.b64encode(combined).decode('ascii')


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        password: Password to verify
        password_hash: Password hash
        
    Returns:
        bool: True if password matches hash
    """
    try:
        # Decode from Base64
        combined = base64.b64decode(password_hash)
        
        # Extract salt and key
        salt = combined[:16]
        key = combined[16:]
        
        # Create key derivation function
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        # Verify key
        kdf.verify(password.encode('utf-8'), key)
        
        return True
    
    except Exception:
        return False


def encrypt_file(input_path: str, output_path: str, password: str, method: str = 'aes-gcm') -> None:
    """
    Encrypt file using the specified method.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        password: Encryption password
        method: Encryption method ('aes-gcm', 'aes-cbc', 'xor')
    """
    try:
        # Read input file
        with open(input_path, 'rb') as f:
            data = f.read()
        
        # Encrypt data
        encrypted = encrypt_data(data, password, method)
        
        # Write output file
        with open(output_path, 'wb') as f:
            f.write(encrypted)
            
    except Exception as e:
        logger.error(f"Error encrypting file: {e}")
        raise


def decrypt_file(input_path: str, output_path: str, password: str, method: str = 'aes-gcm') -> None:
    """
    Decrypt file using the specified method.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        password: Encryption password
        method: Encryption method ('aes-gcm', 'aes-cbc', 'xor')
    """
    try:
        # Read input file
        with open(input_path, 'rb') as f:
            encrypted = f.read()
        
        # Decrypt data
        data = decrypt_data(encrypted, password, method)
        
        # Write output file
        with open(output_path, 'wb') as f:
            f.write(data)
            
    except Exception as e:
        logger.error(f"Error decrypting file: {e}")
        raise


def encrypt_and_sign(data: bytes, password: str) -> bytes:
    """
    Encrypt data and add signature for authentication and integrity.
    
    Args:
        data: Data to encrypt
        password: Encryption password
        
    Returns:
        bytes: Encrypted and signed data
    """
    # Derive encryption key and signing key from password
    encryption_key, salt = derive_key(password, key_length=32)
    signing_key = hashlib.sha256(encryption_key).digest()
    
    # Encrypt data
    iv = os.urandom(12)  # 12 bytes for GCM mode
    cipher = Cipher(
        algorithms.AES(encryption_key),
        modes.GCM(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    tag = encryptor.tag
    
    # Calculate signature
    signature = hmac.new(signing_key, salt + iv + tag + ciphertext, hashlib.sha256).digest()
    
    # Combine all components
    return salt + iv + tag + signature + ciphertext


def decrypt_and_verify(encrypted_data: bytes, password: str) -> bytes:
    """
    Decrypt data and verify signature.
    
    Args:
        encrypted_data: Encrypted and signed data
        password: Encryption password
        
    Returns:
        bytes: Decrypted data
    """
    # Extract components
    salt = encrypted_data[:16]
    iv = encrypted_data[16:28]
    tag = encrypted_data[28:44]
    signature = encrypted_data[44:76]  # 32 bytes for SHA-256
    ciphertext = encrypted_data[76:]
    
    # Derive keys
    encryption_key, _ = derive_key(password, salt, key_length=32)
    signing_key = hashlib.sha256(encryption_key).digest()
    
    # Verify signature
    expected_signature = hmac.new(signing_key, salt + iv + tag + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid signature")
    
    # Decrypt data
    cipher = Cipher(
        algorithms.AES(encryption_key),
        modes.GCM(iv, tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    
    return plaintext