# -*- coding: utf-8 -*-
"""
Text steganography models for Stego-AI.

This module contains the main TextStegoNet class for text steganography,
implementing various algorithms for hiding data in text.
"""

import os
import re
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import numpy as np

from stegoai.models.base import BaseStegoNet
from stegoai.utils.text_utils import (
    text_to_bits, bits_to_bytearray, bytearray_to_text,
    ensure_unicode_printable, find_zero_width_characters,
    strip_zero_width_characters
)

# Set up logging
logger = logging.getLogger(__name__)

# Zero-width characters for whitespace method
ZERO_WIDTH_CHARS = {
    '0': '\u200B',  # Zero width space
    '1': '\u200C',  # Zero width non-joiner
    'start': '\u200D',  # Zero width joiner (used as start marker)
    'end': '\u2060',  # Word joiner (used as end marker)
}


class TextStegoNet(BaseStegoNet):
    """
    Text steganography model.
    
    This class implements various algorithms for hiding data in text:
    - whitespace: Uses zero-width characters to hide data
    - synonym: Replaces words with synonyms based on bit values
    - capitalization: Modifies capitalization patterns
    - generative: Uses language model patterns for hiding
    
    Attributes:
        method (str): Steganography method to use
        model_path (str): Path to language model files (for generative method)
    """
    
    METHODS = {
        'whitespace': {
            'capacity': 'high',
            'robustness': 'medium',
            'visibility': 'low',
        },
        'synonym': {
            'capacity': 'medium',
            'robustness': 'high',
            'visibility': 'medium',
        },
        'capitalization': {
            'capacity': 'low',
            'robustness': 'medium',
            'visibility': 'medium',
        },
        'generative': {
            'capacity': 'high',
            'robustness': 'high',
            'visibility': 'high',
        },
    }
    
    def __init__(
        self,
        method: str = 'whitespace',
        model_path: Optional[str] = None,
        data_depth: int = 1,
        cuda: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize TextStegoNet.
        
        Args:
            method: Steganography method (default: 'whitespace')
            model_path: Path to language model files (for generative method)
            data_depth: Bits per element to hide (default: 1)
            cuda: Whether to use GPU (only for generative method)
            verbose: Whether to print verbose output (default: False)
        """
        super().__init__(
            data_depth=data_depth,
            cuda=cuda,
            verbose=verbose,
        )
        
        self.method = method
        self.model_path = model_path
        
        # Validate method
        if self.method not in self.METHODS:
            raise ValueError(f"Unsupported method: {method}. Supported: {list(self.METHODS.keys())}")
        
        # For synonym method, load synonyms dictionary
        if self.method == 'synonym':
            self.synonym_dict = self._load_synonyms()
        
        # For generative method, load language model
        if self.method == 'generative':
            self.language_model = self._load_language_model()
    
    def _load_synonyms(self):
        """Load synonyms dictionary."""
        # Initialize with some common synonyms
        # In a real implementation, this would load from a larger database
        synonyms = {
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['tiny', 'little', 'miniature', 'diminutive'],
            'happy': ['glad', 'joyful', 'pleased', 'delighted'],
            'sad': ['unhappy', 'sorrowful', 'gloomy', 'depressed'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'leisurely', 'gradual', 'unhurried'],
            'good': ['great', 'excellent', 'fine', 'superb'],
            'bad': ['poor', 'terrible', 'awful', 'dreadful'],
            'beautiful': ['pretty', 'lovely', 'gorgeous', 'attractive'],
            'ugly': ['unattractive', 'hideous', 'unsightly', 'plain'],
            'smart': ['intelligent', 'clever', 'bright', 'brilliant'],
            'stupid': ['foolish', 'dumb', 'idiotic', 'brainless'],
            'old': ['ancient', 'aged', 'elderly', 'senior'],
            'new': ['recent', 'modern', 'fresh', 'current'],
            'hot': ['warm', 'heated', 'scorching', 'burning'],
            'cold': ['cool', 'chilly', 'freezing', 'icy'],
            'wet': ['damp', 'moist', 'soaked', 'soggy'],
            'dry': ['parched', 'arid', 'dehydrated', 'waterless'],
            'loud': ['noisy', 'thunderous', 'booming', 'deafening'],
            'quiet': ['silent', 'hushed', 'soft', 'muted'],
            'hard': ['difficult', 'tough', 'challenging', 'arduous'],
            'easy': ['simple', 'effortless', 'uncomplicated', 'straightforward'],
            'rich': ['wealthy', 'affluent', 'prosperous', 'opulent'],
            'poor': ['impoverished', 'destitute', 'needy', 'penniless'],
            'strong': ['powerful', 'mighty', 'sturdy', 'robust'],
            'weak': ['feeble', 'frail', 'fragile', 'flimsy'],
            'thin': ['slim', 'slender', 'lean', 'skinny'],
            'fat': ['plump', 'chubby', 'stout', 'overweight'],
            'high': ['tall', 'lofty', 'elevated', 'towering'],
            'low': ['short', 'small', 'diminutive', 'undersized'],
        }
        
        # Extend with basic function words and prepositions
        for word in ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'from', 'with', 'by', 'for', 'of', 'about']:
            synonyms[word] = [word]  # No substitution, but needed for the algorithm
        
        if self.verbose:
            logger.info(f"Loaded {len(synonyms)} synonym sets")
        
        return synonyms
    
    def _load_language_model(self):
        """Load language model for generative method."""
        try:
            # This is a placeholder - in a real implementation, this would load a transformer model
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            if self.model_path:
                tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
                model = GPT2LMHeadModel.from_pretrained(self.model_path)
            else:
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Move to GPU if available
            if self.cuda and torch.cuda.is_available():
                model = model.to('cuda')
            
            return {
                'tokenizer': tokenizer,
                'model': model,
            }
        except ImportError:
            logger.warning("Transformers library not available. Using simplified language model.")
            return None
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            return None
    
    def encode(self, cover_path: str, output_path: str, message: str) -> None:
        """
        Hide a message in a text file.
        
        Args:
            cover_path: Path to cover text file
            output_path: Path for output steganographic text
            message: Message to hide
        """
        # Read the cover text
        try:
            with open(cover_path, 'r', encoding='utf-8') as file:
                cover_text = file.read()
        except Exception as e:
            raise ValueError(f"Failed to read cover text: {e}")
        
        # Convert message to bits
        message_bits = text_to_bits(message)
        
        # Add termination marker (32 zeros)
        message_bits = message_bits + [0] * 32
        
        # Apply the appropriate encoding method
        if self.method == 'whitespace':
            stego_text = self._encode_whitespace(cover_text, message_bits)
        elif self.method == 'synonym':
            stego_text = self._encode_synonym(cover_text, message_bits)
        elif self.method == 'capitalization':
            stego_text = self._encode_capitalization(cover_text, message_bits)
        elif self.method == 'generative':
            stego_text = self._encode_generative(cover_text, message_bits)
        else:
            raise ValueError(f"Unknown steganography method: {self.method}")
        
        # Save the steganographic text
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(stego_text)
            
            if self.verbose:
                msg_len = len(message_bits)
                logger.info(f"Message encoded successfully ({msg_len} bits)")
        except Exception as e:
            raise ValueError(f"Failed to write output file: {e}")
    
    def decode(self, stego_path: str) -> str:
        """
        Extract a hidden message from a text file.
        
        Args:
            stego_path: Path to steganographic text
            
        Returns:
            str: Extracted message
        """
        # Read the steganographic text
        try:
            with open(stego_path, 'r', encoding='utf-8') as file:
                stego_text = file.read()
        except Exception as e:
            raise ValueError(f"Failed to read steganographic text: {e}")
        
        # Apply the appropriate decoding method
        if self.method == 'whitespace':
            bits = self._decode_whitespace(stego_text)
        elif self.method == 'synonym':
            bits = self._decode_synonym(stego_text)
        elif self.method == 'capitalization':
            bits = self._decode_capitalization(stego_text)
        elif self.method == 'generative':
            bits = self._decode_generative(stego_text)
        else:
            raise ValueError(f"Unknown steganography method: {self.method}")
        
        # Find termination sequence
        term_index = -1
        for i in range(len(bits) - 31):
            if all(bit == 0 for bit in bits[i:i+32]):
                term_index = i
                break
        
        # Extract message bits (up to termination if found)
        if term_index != -1:
            message_bits = bits[:term_index]
        else:
            message_bits = bits
        
        # Convert bits to text
        if not message_bits:
            raise ValueError("No hidden message found in the text")
        
        try:
            byte_data = bits_to_bytearray(message_bits)
            message = bytearray_to_text(byte_data)
            
            if self.verbose:
                logger.info(f"Message decoded successfully ({len(message)} characters)")
            
            return message
        except Exception as e:
            raise ValueError(f"Failed to decode message: {e}")
    
    def analyze_text(self, text_path: str) -> Dict:
        """
        Analyze text to determine if it contains hidden data.
        
        Args:
            text_path: Path to text for analysis
            
        Returns:
            dict: Analysis results including detection probability
        """
        # Read the text
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            raise ValueError(f"Failed to read text: {e}")
        
        # Apply the appropriate analysis method
        if self.method == 'whitespace':
            return self._analyze_whitespace(text)
        elif self.method == 'synonym':
            return self._analyze_synonym(text)
        elif self.method == 'capitalization':
            return self._analyze_capitalization(text)
        elif self.method == 'generative':
            return self._analyze_generative(text)
        else:
            raise ValueError(f"Unknown steganography method: {self.method}")
    
    def _encode_whitespace(self, cover_text: str, message_bits: List[int]) -> str:
        """
        Hide message using zero-width characters.
        
        Args:
            cover_text: Cover text
            message_bits: Message bits to encode
            
        Returns:
            str: Steganographic text
        """
        # Clean the text of any existing zero-width characters
        clean_text = strip_zero_width_characters(cover_text)
        
        # Convert bits to zero-width characters
        bit_chars = ''.join(ZERO_WIDTH_CHARS[str(bit)] for bit in message_bits)
        
        # Add start and end markers
        stego_data = ZERO_WIDTH_CHARS['start'] + bit_chars + ZERO_WIDTH_CHARS['end']
        
        # Insert at a suitable position (after a paragraph or sentence)
        positions = []
        
        # Find paragraph breaks
        for match in re.finditer(r'\n\s*\n', clean_text):
            positions.append(match.start())
        
        # Find sentence endings
        for match in re.finditer(r'[.!?]\s', clean_text):
            positions.append(match.start() + 1)
        
        if not positions:
            # If no good positions found, place at the beginning
            return stego_data + clean_text
        
        # Choose a position (middle of the text is less suspicious)
        positions.sort()
        position = positions[len(positions) // 2]
        
        # Insert the stego data
        stego_text = clean_text[:position] + stego_data + clean_text[position:]
        
        return stego_text
    
    def _decode_whitespace(self, stego_text: str) -> List[int]:
        """
        Extract message from zero-width characters.
        
        Args:
            stego_text: Steganographic text
            
        Returns:
            list: Extracted message bits
        """
        # Find zero-width characters
        zero_width_chars = find_zero_width_characters(stego_text)
        
        if not zero_width_chars:
            raise ValueError("No zero-width characters found in the text")
        
        # Extract characters
        chars = ''.join(char for _, char in zero_width_chars)
        
        # Find start and end markers
        start_marker = ZERO_WIDTH_CHARS['start']
        end_marker = ZERO_WIDTH_CHARS['end']
        
        start_idx = chars.find(start_marker)
        end_idx = chars.find(end_marker, start_idx + 1)
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Start or end marker not found")
        
        # Extract bits between markers
        bit_chars = chars[start_idx + 1:end_idx]
        
        # Convert characters to bits
        bits = []
        char_to_bit = {v: k for k, v in ZERO_WIDTH_CHARS.items() if k in ['0', '1']}
        
        for char in bit_chars:
            if char in char_to_bit:
                bits.append(int(char_to_bit[char]))
        
        return bits
    
    def _encode_synonym(self, cover_text: str, message_bits: List[int]) -> str:
        """
        Hide message using synonym substitution.
        
        Args:
            cover_text: Cover text
            message_bits: Message bits to encode
            
        Returns:
            str: Steganographic text
        """
        # Ensure we have synonyms loaded
        if not hasattr(self, 'synonym_dict') or not self.synonym_dict:
            self.synonym_dict = self._load_synonyms()
        
        # Tokenize text into words
        # This is a simple tokenization - a real implementation would use NLP
        words = re.findall(r'\b\w+\b', cover_text)
        
        # Find words that have synonyms
        eligible_words = []
        for i, word in enumerate(words):
            if word.lower() in self.synonym_dict and len(self.synonym_dict[word.lower()]) > 1:
                eligible_words.append((i, word))
        
        if not eligible_words:
            raise ValueError("No suitable words found for synonym substitution")
        
        # Check if we have enough words to encode the message
        if len(eligible_words) < len(message_bits):
            raise ValueError(
                f"Not enough suitable words ({len(eligible_words)}) to encode message ({len(message_bits)} bits)"
            )
        
        # Create a mapping of positions to original words
        word_positions = {}
        for i, (pos, word) in enumerate(eligible_words):
            if i < len(message_bits):
                word_positions[pos] = word
        
        # Create a list of all words with substitutions
        new_words = words.copy()
        bit_index = 0
        
        for pos, orig_word in word_positions.items():
            if bit_index >= len(message_bits):
                break
                
            bit = message_bits[bit_index]
            synonyms = self.synonym_dict[orig_word.lower()]
            
            # Ensure we have at least two synonyms
            if len(synonyms) < 2:
                continue
            
            # Choose synonym based on bit
            synonym_idx = bit % len(synonyms)
            synonym = synonyms[synonym_idx]
            
            # Preserve capitalization
            if orig_word[0].isupper():
                synonym = synonym[0].upper() + synonym[1:]
            
            # Replace word
            new_words[pos] = synonym
            bit_index += 1
        
        # Reconstruct text with new words
        text_parts = []
        word_idx = 0
        char_idx = 0
        
        for i, char in enumerate(cover_text):
            if char_idx < len(cover_text):
                # Check if we're at the start of a word
                if i < len(cover_text) - 1 and not cover_text[i-1:i].isalnum() and cover_text[i:i+1].isalnum():
                    # Add the word
                    if word_idx < len(new_words):
                        text_parts.append(new_words[word_idx])
                        # Skip the original word
                        while char_idx < len(cover_text) and cover_text[char_idx:char_idx+1].isalnum():
                            char_idx += 1
                        word_idx += 1
                    continue
                
                # Add non-word characters
                if char_idx < len(cover_text):
                    text_parts.append(cover_text[char_idx:char_idx+1])
                    char_idx += 1
        
        stego_text = ''.join(text_parts)
        
        return stego_text
    
    def _decode_synonym(self, stego_text: str) -> List[int]:
        """
        Extract message from synonym substitution.
        
        Args:
            stego_text: Steganographic text
            
        Returns:
            list: Extracted message bits
        """
        # Ensure we have synonyms loaded
        if not hasattr(self, 'synonym_dict') or not self.synonym_dict:
            self.synonym_dict = self._load_synonyms()
        
        # Tokenize text into words
        words = re.findall(r'\b\w+\b', stego_text)
        
        # Extract bits from words with synonyms
        bits = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.synonym_dict and len(self.synonym_dict[word_lower]) > 1:
                synonyms = self.synonym_dict[word_lower]
                try:
                    synonym_idx = synonyms.index(word_lower)
                    bits.append(synonym_idx % 2)  # Extract bit based on synonym index
                except ValueError:
                    # Word not in synonym list, skip
                    continue
        
        return bits
    
    def _encode_capitalization(self, cover_text: str, message_bits: List[int]) -> str:
        """
        Hide message using capitalization patterns.
        
        Args:
            cover_text: Cover text
            message_bits: Message bits to encode
            
        Returns:
            str: Steganographic text
        """
        # Tokenize text into words
        words = re.findall(r'\b\w+\b', cover_text)
        
        # Filter for suitable words (alphabetic, not all caps, not proper nouns)
        eligible_words = []
        for i, word in enumerate(words):
            # Skip words that are too short
            if len(word) < 2:
                continue
            
            # Skip words that are all uppercase (likely acronyms)
            if word.isupper():
                continue
            
            # Skip words that are probably proper nouns (capitalized in middle of sentence)
            if i > 0 and word[0].isupper() and not words[i-1].endswith(('.', '?', '!')):
                continue
            
            eligible_words.append((i, word))
        
        if not eligible_words:
            raise ValueError("No suitable words found for capitalization")
        
        # Check if we have enough words to encode the message
        if len(eligible_words) < len(message_bits):
            raise ValueError(
                f"Not enough suitable words ({len(eligible_words)}) to encode message ({len(message_bits)} bits)"
            )
        
        # Create mapping of positions to bits
        bit_positions = {}
        for i, (pos, _) in enumerate(eligible_words):
            if i < len(message_bits):
                bit_positions[pos] = message_bits[i]
        
        # Create new text with modified capitalization
        new_words = words.copy()
        
        for pos, bit in bit_positions.items():
            word = new_words[pos]
            
            # Apply capitalization based on bit
            if bit == 0:
                # Lowercase first letter
                new_words[pos] = word[0].lower() + word[1:]
            else:
                # Capitalize first letter
                new_words[pos] = word[0].upper() + word[1:]
        
        # Reconstruct text with modified capitalization
        text_parts = []
        word_idx = 0
        char_idx = 0
        
        while char_idx < len(cover_text):
            # Check if we're at the start of a word
            if char_idx < len(cover_text) - 1 and not cover_text[char_idx-1:char_idx].isalnum() and cover_text[char_idx:char_idx+1].isalnum():
                # Add the word with modified capitalization
                if word_idx < len(new_words):
                    text_parts.append(new_words[word_idx])
                    # Skip the original word
                    orig_word_len = len(words[word_idx])
                    char_idx += orig_word_len
                    word_idx += 1
                continue
            
            # Add non-word characters
            text_parts.append(cover_text[char_idx:char_idx+1])
            char_idx += 1
        
        stego_text = ''.join(text_parts)
        
        return stego_text
    
    def _decode_capitalization(self, stego_text: str) -> List[int]:
        """
        Extract message from capitalization patterns.
        
        Args:
            stego_text: Steganographic text
            
        Returns:
            list: Extracted message bits
        """
        # Tokenize text into words
        words = re.findall(r'\b\w+\b', stego_text)
        
        # Extract bits from capitalization patterns
        bits = []
        
        for i, word in enumerate(words):
            # Skip words that are too short
            if len(word) < 2:
                continue
            
            # Skip words that are all uppercase (likely acronyms)
            if word.isupper():
                continue
            
            # Skip words that are probably proper nouns at start of sentences
            if i > 0 and words[i-1].endswith(('.', '?', '!')):
                continue
            
            # Extract bit based on capitalization
            if word[0].isupper():
                bits.append(1)
            else:
                bits.append(0)
        
        return bits
    
    def _encode_generative(self, cover_text: str, message_bits: List[int]) -> str:
        """
        Hide message using language model generation patterns.
        
        This is a simplified implementation that uses word choices to encode bits.
        A real implementation would use more sophisticated language model features.
        
        Args:
            cover_text: Cover text
            message_bits: Message bits to encode
            
        Returns:
            str: Steganographic text
        """
        # Check if we have a language model
        if not hasattr(self, 'language_model') or not self.language_model:
            # Use a simplified approach without a language model
            return self._encode_synonym(cover_text, message_bits)
        
        # For simplicity, we'll just insert a paragraph at the end with encoded data
        # A real implementation would integrate the message throughout the text
        
        # Placeholder for language model generation
        paragraph = "In summary, this document contains encoded information using natural language. "
        paragraph += "The following text is generated automatically. "
        
        # Encode bits in choice of words
        for i in range(0, len(message_bits), 2):
            if i + 1 < len(message_bits):
                # Get 2 bits
                bit_pair = (message_bits[i], message_bits[i+1])
                
                # Choose word based on bit pair
                if bit_pair == (0, 0):
                    paragraph += "always "
                elif bit_pair == (0, 1):
                    paragraph += "sometimes "
                elif bit_pair == (1, 0):
                    paragraph += "rarely "
                else:  # (1, 1)
                    paragraph += "never "
            else:
                # Last bit
                if message_bits[i] == 0:
                    paragraph += "finally. "
                else:
                    paragraph += "lastly. "
        
        # Combine original text with encoded paragraph
        stego_text = cover_text + "\n\n" + paragraph
        
        return stego_text
    
    def _decode_generative(self, stego_text: str) -> List[int]:
        """
        Extract message from language model generation patterns.
        
        Args:
            stego_text: Steganographic text
            
        Returns:
            list: Extracted message bits
        """
        # Extract the last paragraph
        paragraphs = stego_text.split("\n\n")
        
        if len(paragraphs) < 2:
            raise ValueError("No encoded paragraph found")
        
        paragraph = paragraphs[-1]
        
        # Look for indicator words
        words = paragraph.lower().split()
        
        # Extract bits from word choices
        bits = []
        
        for word in words:
            word = word.strip(".,!?;:")
            
            if word == "always":
                bits.extend([0, 0])
            elif word == "sometimes":
                bits.extend([0, 1])
            elif word == "rarely":
                bits.extend([1, 0])
            elif word == "never":
                bits.extend([1, 1])
            elif word == "finally":
                bits.append(0)
            elif word == "lastly":
                bits.append(1)
        
        return bits
    
    def _analyze_whitespace(self, text: str) -> Dict:
        """
        Analyze text for zero-width character steganography.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results
        """
        # Find zero-width characters
        zero_width_chars = find_zero_width_characters(text)
        
        # Calculate probability based on pattern of zero-width characters
        if not zero_width_chars:
            probability = 0.0
            assessment = "Likely clean"
            confidence = "High"
        else:
            # Check for start and end markers
            chars = ''.join(char for _, char in zero_width_chars)
            start_marker = ZERO_WIDTH_CHARS['start']
            end_marker = ZERO_WIDTH_CHARS['end']
            
            start_idx = chars.find(start_marker)
            end_idx = chars.find(end_marker, start_idx + 1) if start_idx != -1 else -1
            
            if start_idx != -1 and end_idx != -1:
                # Found start and end markers
                probability = 0.95
                assessment = "Likely contains hidden data"
                confidence = "High"
            elif len(zero_width_chars) > 10:
                # Many zero-width characters but no markers
                probability = 0.7
                assessment = "Likely contains hidden data"
                confidence = "Medium"
            else:
                # Some zero-width characters but they might be legitimate
                probability = 0.4
                assessment = "Possibly contains hidden data"
                confidence = "Low"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'zero_width_count': len(zero_width_chars),
        }
    
    def _analyze_synonym(self, text: str) -> Dict:
        """
        Analyze text for synonym-based steganography.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results
        """
        # This is a challenging analysis as synonym usage is common in normal text
        # We'll look for unusual patterns of synonym usage
        
        # Ensure we have synonyms loaded
        if not hasattr(self, 'synonym_dict') or not self.synonym_dict:
            self.synonym_dict = self._load_synonyms()
        
        # Tokenize text into words
        words = re.findall(r'\b\w+\b', text)
        
        # Count words that have synonyms
        synonym_words = 0
        for word in words:
            if word.lower() in self.synonym_dict and len(self.synonym_dict[word.lower()]) > 1:
                synonym_words += 1
        
        # Calculate ratio of words with synonyms
        synonym_ratio = synonym_words / len(words) if words else 0
        
        # Assess based on ratio
        if synonym_ratio > 0.3:  # Unusually high
            probability = 0.7
            assessment = "Likely contains hidden data"
            confidence = "Medium"
        elif synonym_ratio > 0.2:  # Somewhat high
            probability = 0.5
            assessment = "Possibly contains hidden data"
            confidence = "Low"
        else:  # Normal
            probability = 0.2
            assessment = "Likely clean"
            confidence = "Medium"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'synonym_ratio': synonym_ratio,
        }
    
    def _analyze_capitalization(self, text: str) -> Dict:
        """
        Analyze text for capitalization-based steganography.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results
        """
        # Tokenize text into words
        words = re.findall(r'\b\w+\b', text)
        
        # Count inconsistent capitalization
        inconsistent_caps = 0
        for i, word in enumerate(words):
            # Skip words that are too short or all caps
            if len(word) < 2 or word.isupper():
                continue
            
            # Check if capitalization is inconsistent with position
            is_start_of_sentence = i == 0 or (i > 0 and words[i-1].endswith(('.', '?', '!')))
            
            if (is_start_of_sentence and not word[0].isupper()) or (not is_start_of_sentence and word[0].isupper() and not word.istitle()):
                inconsistent_caps += 1
        
        # Calculate ratio of inconsistent capitalization
        cap_ratio = inconsistent_caps / len(words) if words else 0
        
        # Assess based on ratio
        if cap_ratio > 0.1:  # Unusually high
            probability = 0.8
            assessment = "Likely contains hidden data"
            confidence = "Medium"
        elif cap_ratio > 0.05:  # Somewhat high
            probability = 0.6
            assessment = "Possibly contains hidden data"
            confidence = "Low"
        else:  # Normal
            probability = 0.2
            assessment = "Likely clean"
            confidence = "Medium"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'capitalization_ratio': cap_ratio,
        }
    
    def _analyze_generative(self, text: str) -> Dict:
        """
        Analyze text for language model-based steganography.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results
        """
        # Extract the last paragraph
        paragraphs = text.split("\n\n")
        
        if len(paragraphs) < 2:
            return {
                'probability': 0.2,
                'assessment': "Likely clean",
                'confidence': "Medium",
            }
        
        paragraph = paragraphs[-1]
        
        # Look for indicator words
        indicator_words = ["always", "sometimes", "rarely", "never", "finally", "lastly"]
        word_count = 0
        
        for word in paragraph.lower().split():
            word = word.strip(".,!?;:")
            if word in indicator_words:
                word_count += 1
        
        # Calculate probability based on indicator word density
        word_density = word_count / len(paragraph.split()) if paragraph.split() else 0
        
        if word_density > 0.2:  # Very high
            probability = 0.9
            assessment = "Likely contains hidden data"
            confidence = "High"
        elif word_density > 0.1:  # High
            probability = 0.7
            assessment = "Likely contains hidden data"
            confidence = "Medium"
        elif word_density > 0.05:  # Somewhat high
            probability = 0.5
            assessment = "Possibly contains hidden data"
            confidence = "Low"
        else:  # Normal
            probability = 0.2
            assessment = "Likely clean"
            confidence = "Medium"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'indicator_word_density': word_density,
        }
    
    def fit(self, *args, **kwargs):
        """
        Train the model.
        
        Text steganography typically doesn't require training.
        """
        logger.info(f"No training needed for {self.method} method")
        return None
    
    @classmethod
    def load(cls, path=None, method='whitespace', model_path=None, cuda=False, verbose=False):
        """
        Load a model from disk or create a new one.
        
        Args:
            path: Path to saved model (optional)
            method: Steganography method (default: 'whitespace')
            model_path: Path to language model files (for generative method)
            cuda: Whether to use GPU if available
            verbose: Whether to print verbose output
            
        Returns:
            TextStegoNet: Loaded or new model
        """
        if path is not None:
            try:
                model = torch.load(path, map_location='cpu')
                model.verbose = verbose
                
                if verbose:
                    logger.info(f"Model loaded from {path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Creating new model...")
        
        # Create new model with specified parameters
        return cls(method=method, model_path=model_path, cuda=cuda, verbose=verbose)