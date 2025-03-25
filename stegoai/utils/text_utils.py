# -*- coding: utf-8 -*-
"""
Text processing utilities for Stego-AI.

This module provides functions for text processing and
conversion between text and bit representations.
"""

import re
import unicodedata
from typing import List, Optional, Tuple, Union


def text_to_bits(text: str) -> List[int]:
    """
    Convert text to a bit array.
    
    Args:
        text (str): Text to convert
        
    Returns:
        list: Array of bits (0s and 1s)
    """
    # Convert text to bytes using UTF-8 encoding
    text_bytes = text.encode('utf-8')
    
    # Convert each byte to bits
    bits = []
    for byte in text_bytes:
        # Get 8 bits for each byte
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    
    return bits


def bits_to_bytearray(bits: List[int]) -> bytearray:
    """
    Convert a bit array to bytes.
    
    Args:
        bits (list): Array of bits (0s and 1s)
        
    Returns:
        bytearray: Byte array
    """
    # Ensure bit count is a multiple of 8
    pad_len = (8 - len(bits) % 8) % 8
    padded_bits = bits + [0] * pad_len
    
    # Convert bits to bytes
    result = bytearray()
    for i in range(0, len(padded_bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(padded_bits):
                byte = (byte << 1) | padded_bits[i + j]
        result.append(byte)
    
    return result


def bytearray_to_text(data: bytearray) -> str:
    """
    Convert a byte array to text.
    
    Args:
        data (bytearray): Byte array
        
    Returns:
        str: Decoded text
    """
    try:
        # Try UTF-8 first
        return data.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Try with errors='replace'
            return data.decode('utf-8', errors='replace')
        except Exception:
            # Last resort: decode bytes individually
            return ''.join(chr(b) for b in data if 32 <= b <= 126)


def normalize_text(text: str, form: str = 'NFKC') -> str:
    """
    Normalize Unicode text.
    
    Args:
        text (str): Text to normalize
        form (str): Normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
        
    Returns:
        str: Normalized text
    """
    return unicodedata.normalize(form, text)


def ensure_unicode_printable(text: str) -> str:
    """
    Ensure text contains only printable Unicode characters.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Replace control characters
    clean_text = ''.join(c if unicodedata.category(c)[0] != 'C' else ' ' for c in text)
    
    # Normalize
    return normalize_text(clean_text)


def find_zero_width_characters(text: str) -> List[Tuple[int, str]]:
    """
    Find zero-width characters in text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        list: List of (position, character) tuples
    """
    zero_width_chars = [
        '\u200B',  # Zero width space
        '\u200C',  # Zero width non-joiner
        '\u200D',  # Zero width joiner
        '\u2060',  # Word joiner
        '\uFEFF',  # Zero width no-break space
    ]
    
    result = []
    for i, char in enumerate(text):
        if char in zero_width_chars:
            result.append((i, char))
    
    return result


def strip_zero_width_characters(text: str) -> str:
    """
    Remove zero-width characters from text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    zero_width_chars = [
        '\u200B',  # Zero width space
        '\u200C',  # Zero width non-joiner
        '\u200D',  # Zero width joiner
        '\u2060',  # Word joiner
        '\uFEFF',  # Zero width no-break space
    ]
    
    return ''.join(c for c in text if c not in zero_width_chars)


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        int: Word count
    """
    # Split by whitespace and filter out empty strings
    words = [word for word in re.split(r'\s+', text) if word]
    return len(words)


def count_sentences(text: str) -> int:
    """
    Count sentences in text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        int: Sentence count
    """
    # Split by common sentence terminators
    sentences = re.split(r'[.!?]+(?=\s|$)', text)
    # Filter out empty strings
    sentences = [s for s in sentences if s]
    return len(sentences)


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        list: List of words
    """
    # Match word characters
    return re.findall(r'\b\w+\b', text)


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        list: List of sentences
    """
    # Split by common sentence terminators
    sentences = re.split(r'([.!?]+(?=\s|$))', text)
    
    # Rejoin sentences with their punctuation
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    
    # Filter out empty strings
    return [s.strip() for s in result if s.strip()]


def detect_language(text: str) -> str:
    """
    Attempt to detect the language of text.
    
    This is a simple approximation based on character frequency.
    For better results, use a dedicated language detection library.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: ISO 639-1 language code
    """
    # This is a very simplified approach
    # Frequency of common letters varies by language
    
    # Convert to lowercase and count frequency of letters
    text = text.lower()
    letter_count = {}
    total_letters = 0
    
    for char in text:
        if char.isalpha():
            letter_count[char] = letter_count.get(char, 0) + 1
            total_letters += 1
    
    if total_letters == 0:
        return 'unknown'
    
    # Calculate frequency
    frequency = {char: count / total_letters for char, count in letter_count.items()}
    
    # Common letter frequencies in different languages
    # These are approximate and simplified
    language_signatures = {
        'en': {'e': 0.12, 't': 0.09, 'a': 0.08, 'o': 0.07, 'i': 0.07},
        'es': {'e': 0.13, 'a': 0.12, 'o': 0.09, 's': 0.08, 'r': 0.07},
        'fr': {'e': 0.15, 'a': 0.08, 's': 0.08, 'i': 0.07, 't': 0.07},
        'de': {'e': 0.16, 'n': 0.10, 'i': 0.08, 'r': 0.07, 's': 0.07},
    }
    
    # Calculate distance to each language signature
    distances = {}
    for lang, sig in language_signatures.items():
        distance = 0
        for char, freq in sig.items():
            distance += abs(frequency.get(char, 0) - freq)
        distances[lang] = distance
    
    # Return language with smallest distance
    return min(distances, key=distances.get)


def calculate_entropy(text: str) -> float:
    """
    Calculate the Shannon entropy of text.
    
    Higher entropy means more randomness/information content.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Entropy in bits per character
    """
    import math
    
    # Count frequency of each character
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0
    length = len(text)
    
    for count in freq.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    
    Higher scores indicate easier reading:
    90-100: Very easy (5th grade)
    80-89: Easy (6th grade)
    70-79: Fairly easy (7th grade)
    60-69: Standard (8th-9th grade)
    50-59: Fairly difficult (10th-12th grade)
    30-49: Difficult (college)
    0-29: Very difficult (graduate)
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Flesch Reading Ease score
    """
    # Count words, sentences, and syllables
    words = tokenize_words(text)
    sentences = tokenize_sentences(text)
    
    if not words or not sentences:
        return 0
    
    # Approximate syllable count
    # This is a simple approximation; for better results use a dedicated library
    syllable_count = 0
    for word in words:
        word = word.lower()
        if word.endswith('e'):
            word = word[:-1]
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        syllable_count += max(1, count)
    
    # Calculate score
    word_count = len(words)
    sentence_count = len(sentences)
    
    avg_words_per_sentence = word_count / sentence_count
    avg_syllables_per_word = syllable_count / word_count
    
    # Flesch Reading Ease formula
    score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    
    # Clamp score to 0-100 range
    return max(0, min(100, score))


def count_redundant_words(text: str) -> int:
    """
    Count redundant words in text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        int: Number of redundant words
    """
    # List of commonly redundant phrases and their shorter equivalents
    redundancies = {
        r'\b(at|in|during) the (present|current) time\b': 'now',
        r'\bin (the )?spite of\b': 'despite',
        r'\bin the event that\b': 'if',
        r'\bin the vicinity of\b': 'near',
        r'\bfor the purpose of\b': 'for',
        r'\bas a matter of fact\b': 'in fact',
        r'\bat this point in time\b': 'now',
        r'\bdue to the fact that\b': 'because',
        r'\bfact of the matter\b': 'fact',
        r'\bin the near future\b': 'soon',
        r'\bsubject matter\b': 'subject',
    }
    
    count = 0
    for pattern, _ in redundancies.items():
        count += len(re.findall(pattern, text, re.IGNORECASE))
    
    return count


def compress_text(text: str) -> bytes:
    """
    Compress text using zlib.
    
    Args:
        text (str): Text to compress
        
    Returns:
        bytes: Compressed data
    """
    import zlib
    return zlib.compress(text.encode('utf-8'))


def decompress_text(data: bytes) -> str:
    """
    Decompress zlib-compressed text.
    
    Args:
        data (bytes): Compressed data
        
    Returns:
        str: Decompressed text
    """
    import zlib
    try:
        return zlib.decompress(data).decode('utf-8')
    except (zlib.error, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to decompress: {e}")


def synonym_substitution(text: str, synonym_map: dict, probability: float = 0.5) -> str:
    """
    Replace words with synonyms according to a map.
    
    Args:
        text (str): Original text
        synonym_map (dict): Map of words to lists of synonyms
        probability (float): Probability of substituting each word
        
    Returns:
        str: Text with synonyms substituted
    """
    import random
    
    def replace_word(match):
        word = match.group(0)
        word_lower = word.lower()
        
        if word_lower in synonym_map and random.random() < probability:
            synonym = random.choice(synonym_map[word_lower])
            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym[0].upper() + synonym[1:]
            return synonym
        return word
    
    # Find all words
    pattern = r'\b\w+\b'
    return re.sub(pattern, replace_word, text)


def replace_with_homoglyphs(text: str, probability: float = 0.2) -> str:
    """
    Replace characters with similar-looking Unicode homoglyphs.
    
    Args:
        text (str): Original text
        probability (float): Probability of substituting each character
        
    Returns:
        str: Text with homoglyphs substituted
    """
    import random
    
    # Map of ASCII characters to homoglyphs
    homoglyphs = {
        'a': ['а', 'ɑ', 'ɐ', 'ạ'],  # Cyrillic 'а', Latin variants
        'b': ['Ь', 'ɓ', 'ḅ'],
        'c': ['с', 'ϲ', 'ċ'],  # Cyrillic 'с'
        'd': ['ԁ', 'ɗ', 'ḋ'],
        'e': ['е', 'ė', 'ë', 'ẹ'],  # Cyrillic 'е'
        'g': ['ɡ', 'ģ', 'ġ'],
        'h': ['һ', 'ῃ', 'ḩ'],
        'i': ['і', 'ӏ', 'ị', 'ı'],  # Cyrillic 'і'
        'j': ['ј', 'ʝ', 'ĵ'],  # Cyrillic 'ј'
        'k': ['ҝ', 'ķ', 'ḱ'],
        'l': ['ӏ', 'ḽ', 'ḷ'],
        'm': ['м', 'ṃ', 'ḿ'],  # Cyrillic 'м'
        'n': ['ո', 'ṅ', 'ṇ'],
        'o': ['о', 'ο', 'ȯ', 'ọ'],  # Cyrillic 'о', Greek 'ο'
        'p': ['р', 'ρ', 'ṗ'],  # Cyrillic 'р', Greek 'ρ'
        'q': ['ԛ', 'ɋ', 'q̇'],
        'r': ['г', 'ṙ', 'ŗ'],  # Cyrillic 'г'
        's': ['ѕ', 'ṡ', 'ș'],  # Cyrillic 'ѕ'
        't': ['т', 'ṫ', 'ț'],  # Cyrillic 'т'
        'u': ['υ', 'ц', 'ụ'],  # Greek 'υ'
        'v': ['ν', 'ѵ', 'ṿ'],  # Greek 'ν'
        'w': ['ԝ', 'ẇ', 'ẃ'],
        'x': ['х', 'ẋ', 'ӽ'],  # Cyrillic 'х'
        'y': ['у', 'ý', 'ỵ'],  # Cyrillic 'у'
        'z': ['z', 'ż', 'ẓ'],
    }
    
    result = ""
    for char in text:
        if char.lower() in homoglyphs and random.random() < probability:
            # Get homoglyphs for this character
            options = homoglyphs[char.lower()]
            # Choose a random homoglyph
            homoglyph = random.choice(options)
            # If original is uppercase, try to make homoglyph uppercase too
            if char.isupper():
                homoglyph = homoglyph.upper()
            result += homoglyph
        else:
            result += char
    
    return result


def base64_encode(text: str) -> str:
    """
    Encode text as base64.
    
    Args:
        text (str): Text to encode
        
    Returns:
        str: Base64-encoded text
    """
    import base64
    return base64.b64encode(text.encode('utf-8')).decode('ascii')


def base64_decode(encoded: str) -> str:
    """
    Decode base64 to text.
    
    Args:
        encoded (str): Base64-encoded text
        
    Returns:
        str: Decoded text
    """
    import base64
    try:
        return base64.b64decode(encoded).decode('utf-8')
    except (base64.binascii.Error, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to decode: {e}")


def text_to_morse(text: str) -> str:
    """
    Convert text to Morse code.
    
    Args:
        text (str): Text to convert
        
    Returns:
        str: Morse code
    """
    # Morse code mapping
    morse_map = {
        'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.',
        'g': '--.', 'h': '....', 'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
        'm': '--', 'n': '-.', 'o': '---', 'p': '.--.', 'q': '--.-', 'r': '.-.',
        's': '...', 't': '-', 'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-',
        'y': '-.--', 'z': '--..',
        '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
        '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
        '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--',
        '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...',
        ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-',
        '"': '.-..-.', '
    : '...-..-', '@': '.--.-.',
    }
    
    # Convert to lowercase
    text = text.lower()
    
    # Convert characters to Morse code
    morse = []
    for char in text:
        if char == ' ':
            morse.append('/')  # Word separator
        elif char in morse_map:
            morse.append(morse_map[char])
    
    # Join with spaces
    return ' '.join(morse)


def morse_to_text(morse: str) -> str:
    """
    Convert Morse code to text.
    
    Args:
        morse (str): Morse code
        
    Returns:
        str: Decoded text
    """
    # Inverse Morse code mapping
    inverse_morse = {
        '.-': 'a', '-...': 'b', '-.-.': 'c', '-..': 'd', '.': 'e', '..-.': 'f',
        '--.': 'g', '....': 'h', '..': 'i', '.---': 'j', '-.-': 'k', '.-..': 'l',
        '--': 'm', '-.': 'n', '---': 'o', '.--.': 'p', '--.-': 'q', '.-.': 'r',
        '...': 's', '-': 't', '..-': 'u', '...-': 'v', '.--': 'w', '-..-': 'x',
        '-.--': 'y', '--..': 'z',
        '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
        '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
        '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!',
        '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&', '---...': ':',
        '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
        '.-..-.': '"', '...-..-': '
    , '.--.-.': '@',
    }
    
    # Split into individual Morse code characters
    text = []
    for word in morse.split('/'):
        # Process each character in the word
        word_chars = []
        for char in word.strip().split():
            if char in inverse_morse:
                word_chars.append(inverse_morse[char])
        
        # Add the word to the result
        text.append(''.join(word_chars))
    
    # Join words with spaces
    return ' '.join(text)


def rot13(text: str) -> str:
    """
    Apply ROT13 substitution cipher to text.
    
    Args:
        text (str): Text to encode/decode
        
    Returns:
        str: ROT13-encoded/decoded text
    """
    result = []
    for char in text:
        if 'a' <= char.lower() <= 'z':
            # Determine whether the character is uppercase or lowercase
            is_upper = char.isupper()
            
            # Convert to zero-based index (0-25)
            base = ord('A' if is_upper else 'a')
            index = ord(char) - base
            
            # Apply ROT13 transformation (shift by 13)
            index = (index + 13) % 26
            
            # Convert back to ASCII and then to character
            char = chr(index + base)
            
        result.append(char)
    
    return ''.join(result)


def simple_encrypt(text: str, key: str) -> bytes:
    """
    Simple XOR encryption.
    
    Args:
        text (str): Text to encrypt
        key (str): Encryption key
        
    Returns:
        bytes: Encrypted data
    """
    import base64
    
    # Convert text and key to bytes
    text_bytes = text.encode('utf-8')
    key_bytes = key.encode('utf-8')
    
    # XOR encryption
    encrypted = bytearray()
    for i in range(len(text_bytes)):
        encrypted.append(text_bytes[i] ^ key_bytes[i % len(key_bytes)])
    
    # Encode as base64 for easier handling
    return base64.b64encode(encrypted)


def simple_decrypt(encrypted: bytes, key: str) -> str:
    """
    Simple XOR decryption.
    
    Args:
        encrypted (bytes): Encrypted data
        key (str): Encryption key
        
    Returns:
        str: Decrypted text
    """
    import base64
    
    # Decode from base64
    try:
        encrypted_bytes = base64.b64decode(encrypted)
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 data: {e}")
    
    # Convert key to bytes
    key_bytes = key.encode('utf-8')
    
    # XOR decryption
    decrypted = bytearray()
    for i in range(len(encrypted_bytes)):
        decrypted.append(encrypted_bytes[i] ^ key_bytes[i % len(key_bytes)])
    
    # Decode bytes to text
    try:
        return decrypted.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Decryption failed: {e}")