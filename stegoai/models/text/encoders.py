# -*- coding: utf-8 -*-
"""
Text steganography encoders for Stego-AI.

This module implements various encoder architectures for hiding messages
in text using different steganographic techniques.
"""

import os
import re
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stegoai.models.base import BaseEncoder
from stegoai.utils.text_utils import text_to_bits, ensure_unicode_printable

# Set up logging
logger = logging.getLogger(__name__)

# Zero-width characters for whitespace method
ZERO_WIDTH_CHARS = {
    '0': '\u200B',  # Zero width space
    '1': '\u200C',  # Zero width non-joiner
    'start': '\u200D',  # Zero width joiner (used as start marker)
    'end': '\u2060',  # Word joiner (used as end marker)
}


class TextWhitespaceEncoder(BaseEncoder):
    """
    Encoder for hiding messages using zero-width characters.
    
    This encoder hides data by inserting invisible zero-width characters
    that represent bits at specific positions in the text.
    """
    
    def __init__(self, data_depth: int = 1):
        """
        Initialize the whitespace encoder.
        
        Args:
            data_depth: Number of bits to hide per element (not used in this encoder)
        """
        super().__init__()
        self.data_depth = data_depth
    
    def forward(self, cover_text: str, message: List[int]) -> str:
        """
        Hide a message in text using zero-width characters.
        
        Args:
            cover_text: Original text to hide message in
            message: Bit sequence to hide
            
        Returns:
            str: Steganographic text with hidden message
        """
        # Clean the text of any existing zero-width characters
        clean_text = self._strip_zero_width_characters(cover_text)
        
        # Convert bits to zero-width characters
        bit_chars = ''.join(ZERO_WIDTH_CHARS[str(bit)] for bit in message)
        
        # Add start and end markers
        stego_data = ZERO_WIDTH_CHARS['start'] + bit_chars + ZERO_WIDTH_CHARS['end']
        
        # Find suitable positions to insert the steganographic data
        positions = self._find_insertion_positions(clean_text)
        
        if not positions:
            # If no good positions found, place at the beginning
            return stego_data + clean_text
        
        # Choose a position (middle of the text is less suspicious)
        positions.sort()
        position_idx = len(positions) // 2
        position = positions[position_idx]
        
        # Insert the stego data
        stego_text = clean_text[:position] + stego_data + clean_text[position:]
        
        return stego_text
    
    def _strip_zero_width_characters(self, text: str) -> str:
        """
        Remove all zero-width characters from text.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        zero_width_chars = list(ZERO_WIDTH_CHARS.values())
        result = text
        
        for char in zero_width_chars:
            result = result.replace(char, '')
        
        return result
    
    def _find_insertion_positions(self, text: str) -> List[int]:
        """
        Find suitable positions to insert hidden data.
        
        Args:
            text: Text to analyze
            
        Returns:
            list: Character indices suitable for insertion
        """
        positions = []
        
        # Find paragraph breaks (highest priority)
        for match in re.finditer(r'\n\s*\n', text):
            positions.append(match.start() + 1)  # Insert after the first newline
        
        # Find sentence endings (medium priority)
        if not positions:
            for match in re.finditer(r'[.!?]\s+[A-Z]', text):
                positions.append(match.start() + 2)  # Insert after the punctuation and space
        
        # Find phrase breaks (lower priority)
        if not positions:
            for match in re.finditer(r'[,;:]\s+', text):
                positions.append(match.start() + 2)  # Insert after the punctuation and space
        
        # Find word breaks as a last resort
        if not positions:
            for match in re.finditer(r'\s+', text):
                positions.append(match.start() + 1)  # Insert after the space
        
        return positions


class TextSynonymEncoder(BaseEncoder):
    """
    Encoder for hiding messages using synonym substitution.
    
    This encoder hides data by replacing words in the text with
    carefully chosen synonyms based on the bits to encode.
    """
    
    def __init__(self, data_depth: int = 1, synonym_dict: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the synonym encoder.
        
        Args:
            data_depth: Number of bits to hide per word replacement
            synonym_dict: Dictionary of words to their synonyms
        """
        super().__init__()
        self.data_depth = data_depth
        self.synonym_dict = synonym_dict or self._load_default_synonyms()
    
    def forward(self, cover_text: str, message: List[int]) -> str:
        """
        Hide a message in text using synonym substitution.
        
        Args:
            cover_text: Original text to hide message in
            message: Bit sequence to hide
            
        Returns:
            str: Steganographic text with hidden message
        """
        # Parse text into tokens (simple tokenization)
        tokens = []
        spans = []  # (start, end) character positions
        
        current_pos = 0
        for match in re.finditer(r'\b\w+\b', cover_text):
            # Add non-word content before this match
            if match.start() > current_pos:
                tokens.append(cover_text[current_pos:match.start()])
                spans.append((current_pos, match.start()))
            
            # Add the word
            tokens.append(match.group())
            spans.append(match.span())
            
            current_pos = match.end()
        
        # Add any trailing non-word content
        if current_pos < len(cover_text):
            tokens.append(cover_text[current_pos:])
            spans.append((current_pos, len(cover_text)))
        
        # Find eligible words for substitution
        eligible_indices = []
        for i, token in enumerate(tokens):
            # Check if it's a word (not punctuation or whitespace)
            if token.strip() and token.lower() in self.synonym_dict:
                synonyms = self.synonym_dict[token.lower()]
                if len(synonyms) >= 2:  # Need at least 2 options to encode a bit
                    eligible_indices.append(i)
        
        # Check if we can encode the message
        if len(eligible_indices) < len(message) // self.data_depth:
            logger.warning(f"Text has only {len(eligible_indices)} usable words but needs {len(message)//self.data_depth}")
            # Continue anyway, encoding as much as possible
        
        # Encode message by replacing words
        bit_idx = 0
        for word_idx in eligible_indices:
            # Check if we've encoded the whole message
            if bit_idx >= len(message):
                break
            
            # Get the original word and its synonyms
            original_word = tokens[word_idx]
            synonyms = self.synonym_dict[original_word.lower()]
            
            # Select synonym based on next data_depth bits or remaining bits
            remaining_bits = min(self.data_depth, len(message) - bit_idx)
            
            # Calculate the index into the synonym list
            synonym_idx = 0
            for i in range(remaining_bits):
                synonym_idx |= message[bit_idx + i] << i
            
            # Wrap around if needed
            synonym_idx = synonym_idx % len(synonyms)
            
            # Get the synonym
            synonym = synonyms[synonym_idx]
            
            # Preserve capitalization and other case patterns
            synonym = self._match_case(original_word, synonym)
            
            # Replace the token
            tokens[word_idx] = synonym
            
            # Move to next bits
            bit_idx += remaining_bits
        
        # Reconstruct the text
        stego_text = ''.join(tokens)
        
        return stego_text
    
    def _load_default_synonyms(self) -> Dict[str, List[str]]:
        """
        Load a default synonym dictionary.
        
        Returns:
            dict: Mapping of words to synonyms
        """
        # Include a basic set of synonyms for common words
        # In a real implementation, this would use a much larger dataset
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
    
    def _match_case(self, original: str, replacement: str) -> str:
        """
        Match the case pattern of the original word in the replacement.
        
        Args:
            original: Original word
            replacement: Replacement word
            
        Returns:
            str: Replacement word with matched case
        """
        # Handle empty strings
        if not original or not replacement:
            return replacement
        
        # All uppercase
        if original.isupper():
            return replacement.upper()
        
        # All lowercase
        if original.islower():
            return replacement.lower()
        
        # Title case (first letter uppercase, rest lowercase)
        if original[0].isupper() and original[1:].islower():
            return replacement.capitalize()
        
        # Mixed case - try to match pattern
        result = list(replacement.lower())
        for i, char in enumerate(original[:min(len(original), len(replacement))]):
            if char.isupper():
                result[i] = result[i].upper()
        
        return ''.join(result)


class TextCapitalizationEncoder(BaseEncoder):
    """
    Encoder for hiding messages using capitalization patterns.
    
    This encoder hides data by modifying the capitalization of
    certain words in the text based on the bits to encode.
    """
    
    def __init__(self, data_depth: int = 1):
        """
        Initialize the capitalization encoder.
        
        Args:
            data_depth: Number of bits to hide per word (only 1 supported)
        """
        super().__init__()
        self.data_depth = 1  # Only 1 bit per word makes sense for capitalization
    
    def forward(self, cover_text: str, message: List[int]) -> str:
        """
        Hide a message in text using capitalization patterns.
        
        Args:
            cover_text: Original text to hide message in
            message: Bit sequence to hide
            
        Returns:
            str: Steganographic text with hidden message
        """
        # Parse text into words and non-words
        tokens = []
        spans = []  # (start, end) character positions
        
        current_pos = 0
        for match in re.finditer(r'\b\w+\b', cover_text):
            # Add non-word content before this match
            if match.start() > current_pos:
                tokens.append(cover_text[current_pos:match.start()])
                spans.append((current_pos, match.start()))
            
            # Add the word
            tokens.append(match.group())
            spans.append(match.span())
            
            current_pos = match.end()
        
        # Add any trailing non-word content
        if current_pos < len(cover_text):
            tokens.append(cover_text[current_pos:])
            spans.append((current_pos, len(cover_text)))
        
        # Find words eligible for capitalization changes
        eligible_indices = []
        for i, token in enumerate(tokens):
            # Skip non-word tokens
            if not re.match(r'\b\w+\b', token):
                continue
                
            # Skip short words
            if len(token) < 2:
                continue
            
            # Skip all-uppercase words (likely acronyms)
            if token.isupper():
                continue
            
            # Skip words starting sentences
            is_sentence_start = i == 0 or (i > 0 and re.search(r'[.!?]\s*$', tokens[i-1]))
            if is_sentence_start:
                continue
            
            # Skip proper nouns (heuristic: capitalized words not at sentence start)
            if token[0].isupper() and not is_sentence_start:
                continue
            
            eligible_indices.append(i)
        
        # Check if we can encode the message
        if len(eligible_indices) < len(message):
            logger.warning(f"Text has only {len(eligible_indices)} usable words but needs {len(message)}")
            # Continue anyway, encoding as much as possible
        
        # Encode message by modifying capitalization
        for bit_idx, word_idx in enumerate(eligible_indices):
            if bit_idx >= len(message):
                break
            
            # Get the word
            word = tokens[word_idx]
            
            # Modify capitalization based on bit
            if message[bit_idx] == 0:
                # Ensure first letter is lowercase
                if word[0].isupper():
                    tokens[word_idx] = word[0].lower() + word[1:]
            else:
                # Ensure first letter is uppercase
                if word[0].islower():
                    tokens[word_idx] = word[0].upper() + word[1:]
        
        # Reconstruct the text
        stego_text = ''.join(tokens)
        
        return stego_text


class TextGenerativeEncoder(BaseEncoder):
    """
    Encoder for hiding messages using language model generation patterns.
    
    This encoder hides data by generating text that encodes the hidden
    message in the choice of words or patterns, using a language model.
    """
    
    def __init__(self, data_depth: int = 2, lm_tokenizer=None, lm_model=None):
        """
        Initialize the generative encoder.
        
        Args:
            data_depth: Number of bits to encode per generated element
            lm_tokenizer: Language model tokenizer
            lm_model: Language model for text generation
        """
        super().__init__()
        self.data_depth = data_depth
        self.tokenizer = lm_tokenizer
        self.model = lm_model
        
        # Special words for encoding data
        self.encoding_words = {
            # 1-bit encoding
            1: {0: ['always', 'certainly', 'definitely'],
                1: ['sometimes', 'perhaps', 'maybe']},
            
            # 2-bit encoding
            2: {0: ['always', 'certainly', 'definitely'],
                1: ['sometimes', 'occasionally', 'periodically'],
                2: ['rarely', 'seldom', 'hardly'],
                3: ['never', 'none', 'nil']},
            
            # 3-bit encoding
            3: {0: ['confidently', 'certainly', 'definitely'],
                1: ['probably', 'likely', 'presumably'],
                2: ['possibly', 'perhaps', 'maybe'],
                3: ['unlikely', 'doubtfully', 'questionably'],
                4: ['rarely', 'seldom', 'infrequently'],
                5: ['hardly', 'barely', 'scarcely'],
                6: ['almost', 'nearly', 'virtually'],
                7: ['never', 'nowhere', 'nought']}
        }
    
    def forward(self, cover_text: str, message: List[int]) -> str:
        """
        Hide a message using language model generation patterns.
        
        Args:
            cover_text: Original text to hide message in
            message: Bit sequence to hide
            
        Returns:
            str: Steganographic text with hidden message
        """
        # If we have a language model, use it for generating text
        if self.model and self.tokenizer:
            return self._encode_with_lm(cover_text, message)
        else:
            # Fallback to simpler method if no language model
            return self._encode_simple(cover_text, message)
    
    def _encode_with_lm(self, cover_text: str, message: List[int]) -> str:
        """
        Hide a message using a language model.
        
        Args:
            cover_text: Original text to hide message in
            message: Bit sequence to hide
            
        Returns:
            str: Steganographic text with hidden message
        """
        # Convert message bits to tokens
        enc_depth = min(self.data_depth, 3)  # Use at most 3 bits per word (2^3 = 8 options)
        tokens = []
        
        for i in range(0, len(message), enc_depth):
            # Get the next enc_depth bits (or fewer if at the end)
            bits = message[i:i+enc_depth]
            actual_depth = len(bits)
            
            # Calculate the encoding value
            value = 0
            for j, bit in enumerate(bits):
                value |= bit << j
            
            # Get a random word for this value
            options = self.encoding_words.get(actual_depth, {}).get(value, [''])
            tokens.append(random.choice(options))
        
        # Generate text using the model and token constraints
        try:
            # Prepare the prompt with cover text
            prompt = cover_text
            
            # Add a natural transition sentence
            prompt += "\n\nIn summary, when considering these factors, we can say that "
            
            # Generate text with the model, encouraging it to use our tokens
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Move to GPU if available
            if hasattr(self, 'cuda') and self.cuda and torch.cuda.is_available():
                input_ids = input_ids.to('cuda')
                self.model = self.model.to('cuda')
            
            # Generate text
            outputs = []
            current_input = input_ids
            
            for token_word in tokens:
                # Generate next part with guidance toward our token
                output = self.model.generate(
                    current_input,
                    max_length=current_input.shape[1] + 10,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                )
                
                # Decode the output
                generated_text = self.tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                
                # Extract the new part
                new_part = generated_text[len(self.tokenizer.decode(current_input[0], skip_special_tokens=True)):]
                
                # Ensure our token word is included
                if token_word not in new_part:
                    if new_part.strip():  # If there's some text, append our word
                        new_part += f" {token_word}"
                    else:  # If empty, just use our word
                        new_part = f" {token_word}"
                
                # Add this part to outputs
                outputs.append(new_part)
                
                # Update input for next iteration
                current_input = self.tokenizer.encode(generated_text + " ", return_tensors="pt")
                if hasattr(self, 'cuda') and self.cuda and torch.cuda.is_available():
                    current_input = current_input.to('cuda')
            
            # Combine everything
            stego_text = prompt + ''.join(outputs)
            
            return stego_text
            
        except Exception as e:
            logger.error(f"Error generating text with language model: {e}")
            # Fall back to simple method
            return self._encode_simple(cover_text, message)
    
    def _encode_simple(self, cover_text: str, message: List[int]) -> str:
        """
        Hide a message using simple word substitution.
        
        Args:
            cover_text: Original text to hide message in
            message: Bit sequence to hide
            
        Returns:
            str: Steganographic text with hidden message
        """
        # Create a simple encoding paragraph
        enc_depth = min(self.data_depth, 3)  # Use at most 3 bits per word
        
        # Start with a natural transition
        paragraph = "\n\nIn conclusion, we can say that "
        
        # Add encoded words
        i = 0
        while i < len(message):
            # Get the next enc_depth bits (or fewer if at the end)
            remaining = min(enc_depth, len(message) - i)
            bits = message[i:i+remaining]
            
            # Calculate the value
            value = 0
            for j, bit in enumerate(bits):
                value |= bit << j
            
            # Get words for this value
            options = self.encoding_words.get(remaining, {}).get(value, [''])
            word = random.choice(options)
            
            # Add connecting words for natural flow
            if i > 0:
                connectors = ["and ", "while ", "as ", "although ", "since ", "because ", "", ""]
                paragraph += random.choice(connectors)
            
            paragraph += word + " "
            
            # Add filler words occasionally for more natural text
            if random.random() < 0.3:
                fillers = [
                    "it is ", "we find ", "the results are ", "the evidence suggests ",
                    "the data indicates ", "we observe ", "one can see that ", "it appears "
                ]
                paragraph += random.choice(fillers)
            
            i += remaining
        
        # Add a closing period
        paragraph += "."
        
        # Combine with original text
        stego_text = cover_text + paragraph
        
        return stego_text


def get_text_encoder(method: str = 'whitespace', **kwargs) -> BaseEncoder:
    """
    Factory function to get the appropriate text encoder.
    
    Args:
        method: Steganography method ('whitespace', 'synonym', 'capitalization', 'generative')
        **kwargs: Additional arguments to pass to the encoder
        
    Returns:
        BaseEncoder: Appropriate encoder for the method
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'whitespace':
        return TextWhitespaceEncoder(**kwargs)
    elif method == 'synonym':
        return TextSynonymEncoder(**kwargs)
    elif method == 'capitalization':
        return TextCapitalizationEncoder(**kwargs)
    elif method == 'generative':
        return TextGenerativeEncoder(**kwargs)
    else:
        raise ValueError(f"Unsupported text steganography method: {method}")