# -*- coding: utf-8 -*-
"""
Text steganography decoders for Stego-AI.

This module implements various decoder architectures for extracting messages
from text using different steganographic techniques.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stegoai.models.base import BaseDecoder
from stegoai.utils.text_utils import bits_to_bytearray, bytearray_to_text

# Set up logging
logger = logging.getLogger(__name__)

# Zero-width characters for whitespace method
ZERO_WIDTH_CHARS = {
    '0': '\u200B',  # Zero width space
    '1': '\u200C',  # Zero width non-joiner
    'start': '\u200D',  # Zero width joiner (used as start marker)
    'end': '\u2060',  # Word joiner (used as end marker)
}


class TextWhitespaceDecoder(BaseDecoder):
    """
    Decoder for extracting messages hidden using zero-width characters.
    
    This decoder extracts bits encoded as invisible zero-width characters
    inserted at specific positions in the text.
    """
    
    def __init__(self, data_depth: int = 1):
        """
        Initialize the whitespace decoder.
        
        Args:
            data_depth: Number of bits hidden per element (not used in this decoder)
        """
        super().__init__()
        self.data_depth = data_depth
    
    def forward(self, stego_text: str) -> List[int]:
        """
        Extract a message hidden using zero-width characters.
        
        Args:
            stego_text: Steganographic text with hidden message
            
        Returns:
            list: Extracted bit sequence
        """
        # Find all zero-width characters in the text
        zero_width_chars = self._find_zero_width_characters(stego_text)
        
        if not zero_width_chars:
            logger.warning("No zero-width characters found in the text")
            return []
        
        # Extract characters
        chars = ''.join(char for _, char in zero_width_chars)
        
        # Find start and end markers
        start_marker = ZERO_WIDTH_CHARS['start']
        end_marker = ZERO_WIDTH_CHARS['end']
        
        start_idx = chars.find(start_marker)
        end_idx = chars.find(end_marker, start_idx + 1) if start_idx != -1 else -1
        
        if start_idx == -1 or end_idx == -1:
            logger.warning("Start or end marker not found in zero-width characters")
            return []
        
        # Extract bits between markers
        bit_chars = chars[start_idx + 1:end_idx]
        
        # Convert characters to bits
        bits = []
        char_to_bit = {v: k for k, v in ZERO_WIDTH_CHARS.items() if k in ['0', '1']}
        
        for char in bit_chars:
            if char in char_to_bit:
                bits.append(int(char_to_bit[char]))
        
        return bits
    
    def _find_zero_width_characters(self, text: str) -> List[Tuple[int, str]]:
        """
        Find all zero-width characters in text with their positions.
        
        Args:
            text: Text to scan for zero-width characters
            
        Returns:
            list: Tuples of (position, character)
        """
        zero_width_chars = []
        for i, char in enumerate(text):
            if char in ZERO_WIDTH_CHARS.values():
                zero_width_chars.append((i, char))
        return zero_width_chars


class TextSynonymDecoder(BaseDecoder):
    """
    Decoder for extracting messages hidden using synonym substitution.
    
    This decoder identifies words that have been replaced with synonyms
    and extracts the bits encoded by these replacements.
    """
    
    def __init__(self, data_depth: int = 1, synonym_dict: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the synonym decoder.
        
        Args:
            data_depth: Number of bits hidden per word replacement
            synonym_dict: Dictionary of words to their synonyms
        """
        super().__init__()
        self.data_depth = data_depth
        self.synonym_dict = synonym_dict or self._load_default_synonyms()
        self.reversed_dict = self._create_reversed_dict()
    
    def forward(self, stego_text: str) -> List[int]:
        """
        Extract a message hidden using synonym substitution.
        
        Args:
            stego_text: Steganographic text with hidden message
            
        Returns:
            list: Extracted bit sequence
        """
        # Extract words from text
        words = re.findall(r'\b\w+\b', stego_text)
        
        # Extract bits
        bits = []
        for word in words:
            # Get lowercase version for dictionary lookup
            word_lower = word.lower()
            
            # Check if word is in reversed dictionary
            if word_lower in self.reversed_dict:
                # Get canonical form and synonym list
                canonical, synonyms = self.reversed_dict[word_lower]
                
                # Find index in synonym list
                try:
                    idx = synonyms.index(word_lower)
                    
                    # Convert index to bits based on data_depth
                    for i in range(self.data_depth):
                        # Extract each bit from the index
                        bit = (idx >> i) & 1
                        bits.append(bit)
                
                except ValueError:
                    # Skip if not in synonym list (shouldn't happen)
                    continue
        
        return bits
    
    def _load_default_synonyms(self) -> Dict[str, List[str]]:
        """
        Load a default synonym dictionary.
        
        Returns:
            dict: Mapping of words to synonyms
        """
        # Include a basic set of synonyms for common words
        # This should match the encoder's dictionary
        return {
            # Common adjectives
            'big': ['large', 'huge', 'enormous', 'massive', 'substantial', 'great', 'sizable'],
            'small': ['tiny', 'little', 'miniature', 'petite', 'minute', 'microscopic', 'compact'],
            'good': ['great', 'excellent', 'fine', 'quality', 'superior', 'wonderful', 'superb', 'exceptional'],
            'bad': ['poor', 'terrible', 'awful', 'dreadful', 'horrible', 'inferior', 'substandard', 'unpleasant'],
            'happy': ['glad', 'joyful', 'cheerful', 'delighted', 'pleased', 'content', 'thrilled', 'elated'],
            'sad': ['unhappy', 'sorrowful', 'dejected', 'depressed', 'gloomy', 'miserable', 'downhearted', 'blue'],
            'beautiful': ['pretty', 'attractive', 'gorgeous', 'stunning', 'lovely', 'exquisite', 'handsome', 'elegant'],
            'ugly': ['unattractive', 'hideous', 'unsightly', 'homely', 'plain', 'grotesque', 'repulsive', 'unpleasant'],
            'smart': ['intelligent', 'clever', 'bright', 'brilliant', 'sharp', 'wise', 'astute', 'intellectual'],
            'stupid': ['foolish', 'dumb', 'idiotic', 'dim', 'dense', 'brainless', 'mindless', 'simpleminded'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty', 'expeditious', 'brisk', 'nimble'],
            'slow': ['sluggish', 'gradual', 'unhurried', 'leisurely', 'plodding', 'dawdling', 'lagging', 'crawling'],
            'hot': ['warm', 'heated', 'scorching', 'burning', 'fiery', 'sweltering', 'boiling', 'sizzling'],
            'cold': ['cool', 'chilly', 'freezing', 'icy', 'frosty', 'frigid', 'wintry', 'bitter'],
            
            # Common verbs
            'run': ['sprint', 'dash', 'jog', 'race', 'gallop', 'bolt', 'scurry', 'hurry'],
            'walk': ['stroll', 'stride', 'saunter', 'amble', 'wander', 'hike', 'trudge', 'tread'],
            'look': ['see', 'view', 'observe', 'watch', 'examine', 'inspect', 'glimpse', 'gaze'],
            'eat': ['consume', 'devour', 'ingest', 'dine', 'feast', 'chew', 'nibble', 'munch'],
            'say': ['tell', 'speak', 'utter', 'express', 'articulate', 'verbalize', 'mention', 'comment'],
            'think': ['believe', 'consider', 'contemplate', 'ponder', 'reflect', 'deliberate', 'meditate', 'muse'],
            'make': ['create', 'form', 'construct', 'build', 'fabricate', 'produce', 'manufacture', 'assemble'],
            'get': ['obtain', 'acquire', 'gain', 'procure', 'attain', 'secure', 'receive', 'fetch'],
            
            # Common nouns
            'house': ['home', 'residence', 'dwelling', 'abode', 'domicile', 'habitat', 'lodging', 'quarters'],
            'car': ['vehicle', 'automobile', 'motor', 'machine', 'ride', 'transport', 'sedan', 'coupe'],
            'job': ['work', 'occupation', 'profession', 'career', 'vocation', 'position', 'role', 'employment'],
            'money': ['cash', 'currency', 'funds', 'wealth', 'capital', 'finance', 'assets', 'resources'],
            'food': ['sustenance', 'nourishment', 'fare', 'cuisine', 'provisions', 'victuals', 'edibles', 'meals'],
            'friend': ['companion', 'pal', 'ally', 'associate', 'confidant', 'buddy', 'comrade', 'mate'],
            'time': ['period', 'duration', 'span', 'term', 'interval', 'era', 'epoch', 'age'],
            'way': ['method', 'approach', 'manner', 'means', 'technique', 'procedure', 'path', 'route'],
            
            # Common function words (using identity mapping to avoid changing these)
            'the': ['the'],
            'a': ['a'],
            'an': ['an'],
            'and': ['and'],
            'or': ['or'],
            'but': ['but'],
            'if': ['if'],
            'of': ['of'],
            'at': ['at'],
            'by': ['by'],
            'for': ['for'],
            'with': ['with'],
            'about': ['about'],
            'from': ['from'],
            'to': ['to'],
            'in': ['in'],
            'on': ['on'],
            'is': ['is'],
            'are': ['are'],
            'was': ['was'],
            'were': ['were'],
        }
    
    def _create_reversed_dict(self) -> Dict[str, Tuple[str, List[str]]]:
        """
        Create a reversed dictionary for decoding.
        
        The reversed dictionary maps each synonym to its canonical form
        and the full list of synonyms, for efficient decoding.
        
        Returns:
            dict: Mapping of synonyms to (canonical_form, full_synonym_list)
        """
        reversed_dict = {}
        
        for canonical, synonyms in self.synonym_dict.items():
            # Add the canonical form itself
            if canonical not in reversed_dict:
                reversed_dict[canonical] = (canonical, synonyms)
            
            # Add all synonyms
            for synonym in synonyms:
                if synonym not in reversed_dict:
                    reversed_dict[synonym] = (canonical, synonyms)
        
        return reversed_dict


class TextCapitalizationDecoder(BaseDecoder):
    """
    Decoder for extracting messages hidden using capitalization patterns.
    
    This decoder identifies words with modified capitalization and
    extracts the bits encoded by these modifications.
    """
    
    def __init__(self, data_depth: int = 1):
        """
        Initialize the capitalization decoder.
        
        Args:
            data_depth: Number of bits hidden per word (only 1 supported)
        """
        super().__init__()
        self.data_depth = 1  # Only 1 bit per word makes sense for capitalization
    
    def forward(self, stego_text: str) -> List[int]:
        """
        Extract a message hidden using capitalization patterns.
        
        Args:
            stego_text: Steganographic text with hidden message
            
        Returns:
            list: Extracted bit sequence
        """
        # Extract words and their context
        words_with_context = self._extract_words_with_context(stego_text)
        
        # Extract bits from eligible words
        bits = []
        for i, (word, is_sentence_start) in enumerate(words_with_context):
            # Skip short words
            if len(word) < 2:
                continue
            
            # Skip all-uppercase words (likely acronyms)
            if word.isupper():
                continue
            
            # Skip words starting sentences (normally capitalized)
            if is_sentence_start:
                continue
            
            # Skip likely proper nouns (common words are never capitalized mid-sentence)
            common_words = {'the', 'a', 'an', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 
                            'with', 'by', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
                            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                            'some', 'such', 'no', 'not', 'only', 'own', 'same', 'than', 'too',
                            'very', 'can', 'will', 'just', 'should', 'now'}
            
            if word.lower() in common_words and word[0].isupper() and not is_sentence_start:
                # This is likely an encoded bit rather than a proper noun
                pass
            elif word[0].isupper() and not is_sentence_start:
                # Skip likely proper nouns
                continue
            
            # Extract bit based on capitalization
            if word[0].isupper():
                bits.append(1)
            else:
                bits.append(0)
        
        return bits
    
    def _extract_words_with_context(self, text: str) -> List[Tuple[str, bool]]:
        """
        Extract words from text with context about sentence boundaries.
        
        Args:
            text: Text to analyze
            
        Returns:
            list: Tuples of (word, is_sentence_start)
        """
        words_with_context = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            
            for i, word in enumerate(words):
                is_sentence_start = (i == 0)
                words_with_context.append((word, is_sentence_start))
        
        return words_with_context


class TextGenerativeDecoder(BaseDecoder):
    """
    Decoder for extracting messages hidden using language model generation patterns.
    
    This decoder identifies specific words or patterns in generated text
    that encode hidden bits based on the encoder's scheme.
    """
    
    def __init__(self, data_depth: int = 2):
        """
        Initialize the generative decoder.
        
        Args:
            data_depth: Number of bits encoded per generated element
        """
        super().__init__()
        self.data_depth = data_depth
        
        # Special words for decoding data (must match encoder)
        self.decoding_words = {
            # 1-bit encoding
            1: {'always': 0, 'certainly': 0, 'definitely': 0,
                'sometimes': 1, 'perhaps': 1, 'maybe': 1},
            
            # 2-bit encoding
            2: {'always': 0, 'certainly': 0, 'definitely': 0,
                'sometimes': 1, 'occasionally': 1, 'periodically': 1,
                'rarely': 2, 'seldom': 2, 'hardly': 2,
                'never': 3, 'none': 3, 'nil': 3},
            
            # 3-bit encoding
            3: {'confidently': 0, 'certainly': 0, 'definitely': 0,
                'probably': 1, 'likely': 1, 'presumably': 1,
                'possibly': 2, 'perhaps': 2, 'maybe': 2,
                'unlikely': 3, 'doubtfully': 3, 'questionably': 3,
                'rarely': 4, 'seldom': 4, 'infrequently': 4,
                'hardly': 5, 'barely': 5, 'scarcely': 5,
                'almost': 6, 'nearly': 6, 'virtually': 6,
                'never': 7, 'nowhere': 7, 'nought': 7}
        }
    
    def forward(self, stego_text: str) -> List[int]:
        """
        Extract a message hidden in generatively created text.
        
        Args:
            stego_text: Steganographic text with hidden message
            
        Returns:
            list: Extracted bit sequence
        """
        # Check for summary/conclusion section as a marker
        markers = [
            "In summary", "In conclusion", "To summarize", "To conclude",
            "In the end", "Finally", "Lastly", "To sum up"
        ]
        
        # Find the marker section
        marker_pos = -1
        for marker in markers:
            pos = stego_text.find(marker)
            if pos != -1 and (marker_pos == -1 or pos < marker_pos):
                marker_pos = pos
        
        if marker_pos == -1:
            # No marker found, check the entire text
            text_to_check = stego_text
        else:
            # Extract the section after the marker
            text_to_check = stego_text[marker_pos:]
        
        # Identify encoding words in the text
        words = re.findall(r'\b\w+\b', text_to_check.lower())
        
        # Extract bits from encoding words
        all_bits = []
        
        # Try each encoding depth, starting with the highest
        for depth in range(3, 0, -1):
            if depth > self.data_depth:
                continue
                
            bits = []
            
            for word in words:
                if word in self.decoding_words.get(depth, {}):
                    # Get the value encoded by this word
                    value = self.decoding_words[depth][word]
                    
                    # Convert to bits
                    word_bits = []
                    for i in range(depth):
                        word_bits.append((value >> i) & 1)
                    
                    bits.extend(word_bits)
            
            # If we found a good number of bits, use this depth
            if len(bits) >= 8:  # At least one byte
                all_bits = bits
                break
        
        return all_bits


def get_text_decoder(method: str = 'whitespace', **kwargs) -> BaseDecoder:
    """
    Factory function to get the appropriate text decoder.
    
    Args:
        method: Steganography method ('whitespace', 'synonym', 'capitalization', 'generative')
        **kwargs: Additional arguments to pass to the decoder
        
    Returns:
        BaseDecoder: Appropriate decoder for the method
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'whitespace':
        return TextWhitespaceDecoder(**kwargs)
    elif method == 'synonym':
        return TextSynonymDecoder(**kwargs)
    elif method == 'capitalization':
        return TextCapitalizationDecoder(**kwargs)
    elif method == 'generative':
        return TextGenerativeDecoder(**kwargs)
    else:
        raise ValueError(f"Unsupported text steganography method: {method}")