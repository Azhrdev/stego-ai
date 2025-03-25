# -*- coding: utf-8 -*-
"""
Network steganography models for Stego-AI.

This module contains the main NetworkStegoNet class for network steganography,
implementing various algorithms for hiding data in network communications.
"""

import os
import logging
import socket
import struct
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import numpy as np

from stegoai.models.base import BaseStegoNet
from stegoai.utils.text_utils import text_to_bits, bits_to_bytearray, bytearray_to_text

# Set up logging
logger = logging.getLogger(__name__)


class NetworkStegoNet(BaseStegoNet):
    """
    Network steganography model.
    
    This class implements various algorithms for hiding data in network traffic:
    - header: Hides data in unused or rarely checked protocol headers
    - timing: Encodes data through packet timing
    - size: Encodes data through packet size manipulation
    - sequence: Encodes data through sequence manipulation
    - covert_channel: Creates covert channels through agreed protocols
    
    Attributes:
        method (str): Steganography method to use
        protocol (str): Network protocol to use (tcp, udp, icmp, etc.)
        interface (str): Network interface to use
    """

    METHODS = {
        'header': {
            'capacity': 'medium',
            'robustness': 'medium',
            'visibility': 'medium',
        },
        'timing': {
            'capacity': 'low',
            'robustness': 'high',
            'visibility': 'low',
        },
        'size': {
            'capacity': 'medium',
            'robustness': 'medium',
            'visibility': 'medium',
        },
        'sequence': {
            'capacity': 'high',
            'robustness': 'low',
            'visibility': 'high',
        },
        'covert_channel': {
            'capacity': 'high',
            'robustness': 'medium',
            'visibility': 'medium',
        },
    }

    PROTOCOLS = ['tcp', 'udp', 'icmp', 'dns']

    def __init__(
        self, 
        method: str = 'header',
        protocol: str = 'tcp',
        interface: str = 'eth0',
        port: int = 8000,
        data_depth: int = 1,
        cuda: bool = False,  # Not typically needed for network stego
        verbose: bool = False,
    ):
        """
        Initialize NetworkStegoNet.
        
        Args:
            method: Steganography method (default: 'header')
            protocol: Network protocol (default: 'tcp')
            interface: Network interface (default: 'eth0')
            port: Network port for applicable protocols (default: 8000)
            data_depth: Bits per element to hide (default: 1)
            cuda: Whether to use GPU (not typically needed)
            verbose: Whether to print verbose output (default: False)
        """
        super().__init__(
            data_depth=data_depth,
            cuda=cuda,
            verbose=verbose,
        )
        
        self.method = method
        self.protocol = protocol.lower()
        self.interface = interface
        self.port = port
        
        # Validate protocol
        if self.protocol not in self.PROTOCOLS:
            raise ValueError(f"Unsupported protocol: {protocol}. Supported: {self.PROTOCOLS}")
            
        # Validate method
        if self.method not in self.METHODS:
            raise ValueError(f"Unsupported method: {method}. Supported: {list(self.METHODS.keys())}")
            
        # For some methods, we need extra configuration
        if self.method == 'covert_channel':
            # Define a secret pattern or protocol
            self.secret_pattern = b'STEGOAI'
            
        # Initialize required libraries based on method
        self._init_network_libs()
    
    def _init_network_libs(self):
        """Initialize required network libraries."""
        try:
            # Try to import required modules
            # Note: These may require installation or admin/root privileges
            # Only import what we need based on the method and protocol
            
            if self.protocol in ['tcp', 'udp']:
                import socket
                self.socket = socket
            
            if self.protocol == 'icmp' or self.method in ['header', 'sequence']:
                try:
                    import scapy.all as scapy
                    self.scapy = scapy
                except ImportError:
                    logger.warning("Scapy library not available. Some methods may not work.")
            
            if self.method == 'timing':
                import time
                self.time = time
                
        except ImportError as e:
            logger.warning(f"Could not import all required libraries: {e}")
            logger.warning("Some methods may not be available.")
    
    def encode(self, target: str, message: str, port: Optional[int] = None) -> bool:
        """
        Hide a message in network traffic to a target.
        
        Args:
            target: Target hostname or IP address
            message: Message to hide
            port: Optional port override (default: use instance port)
            
        Returns:
            bool: Success status
        """
        if port is None:
            port = self.port
            
        # Convert message to bits
        message_bits = text_to_bits(message)
        
        # Add a terminator sequence (32 zeros)
        message_bits = message_bits + [0] * 32
        
        # Call the appropriate encoding method
        if self.method == 'header':
            return self._encode_header(target, message_bits, port)
        elif self.method == 'timing':
            return self._encode_timing(target, message_bits, port)
        elif self.method == 'size':
            return self._encode_size(target, message_bits, port)
        elif self.method == 'sequence':
            return self._encode_sequence(target, message_bits, port)
        elif self.method == 'covert_channel':
            return self._encode_covert_channel(target, message_bits, port)
        else:
            raise ValueError(f"Unknown steganography method: {self.method}")
    
    def decode(self, listen_time: int = 60, callback: Optional[Callable] = None) -> str:
        """
        Listen for and extract a hidden message from network traffic.
        
        Args:
            listen_time: How long to listen for messages (in seconds)
            callback: Optional progress callback function
            
        Returns:
            str: Extracted message
        """
        # Call the appropriate decoding method
        if self.method == 'header':
            bits = self._decode_header(listen_time, callback)
        elif self.method == 'timing':
            bits = self._decode_timing(listen_time, callback)
        elif self.method == 'size':
            bits = self._decode_size(listen_time, callback)
        elif self.method == 'sequence':
            bits = self._decode_sequence(listen_time, callback)
        elif self.method == 'covert_channel':
            bits = self._decode_covert_channel(listen_time, callback)
        else:
            raise ValueError(f"Unknown steganography method: {self.method}")
            
        # Find terminator sequence
        terminator_pos = -1
        for i in range(len(bits) - 32):
            if all(bit == 0 for bit in bits[i:i+32]):
                terminator_pos = i
                break
        
        # Extract message bits (up to terminator if found)
        if terminator_pos != -1:
            message_bits = bits[:terminator_pos]
        else:
            message_bits = bits
        
        # Convert to text
        try:
            byte_data = bits_to_bytearray(message_bits)
            message = bytearray_to_text(byte_data)
            return message
        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            raise ValueError("Could not extract a valid message from the traffic")
    
    def estimate_capacity(self, protocol: Optional[str] = None, method: Optional[str] = None) -> Dict:
        """
        Estimate the steganographic capacity of a network method.
        
        Args:
            protocol: Network protocol (default: instance protocol)
            method: Steganography method (default: instance method)
            
        Returns:
            dict: Capacity information
        """
        if protocol is None:
            protocol = self.protocol
            
        if method is None:
            method = self.method
            
        # Baseline capacity in bits per packet
        if method == 'header':
            if protocol == 'tcp':
                # TCP headers have several fields:
                # - Sequence and ACK numbers: 64 bits
                # - Reserved bits: 6 bits
                # - Window size: 16 bits
                # - Urgent pointer: 16 bits
                # But we can't use all of them without breaking the protocol
                capacity_per_packet = 16  # Conservative estimate
            elif protocol == 'udp':
                # Less space in UDP headers
                capacity_per_packet = 8
            elif protocol == 'icmp':
                # ICMP headers have more unused fields
                capacity_per_packet = 32
            elif protocol == 'dns':
                # DNS queries have various fields and can hide data in case variations
                capacity_per_packet = 40
            else:
                capacity_per_packet = 8
        elif method == 'timing':
            # Very low capacity, typically 1 bit per packet timing
            capacity_per_packet = 1
        elif method == 'size':
            # Can vary packet size to encode more data
            capacity_per_packet = 8
        elif method == 'sequence':
            # Sequence numbers and ordering can encode more data
            capacity_per_packet = 12 if protocol == 'tcp' else 4
        elif method == 'covert_channel':
            # Custom protocols can hide more data
            capacity_per_packet = 64
        else:
            capacity_per_packet = 0
            
        # Adjust for data_depth (if applicable)
        capacity_per_packet = capacity_per_packet * self.data_depth
        
        # Estimate typical packets per second for selected protocol
        if protocol == 'tcp':
            packets_per_second = 100  # Typical HTTP/S traffic
        elif protocol == 'udp':
            packets_per_second = 30   # Typical UDP traffic
        elif protocol == 'icmp':
            packets_per_second = 1    # ICMP is typically more limited
        elif protocol == 'dns':
            packets_per_second = 5    # DNS queries
        else:
            packets_per_second = 10
            
        # Calculate bits per second
        bits_per_second = capacity_per_packet * packets_per_second
        
        # Estimate text capacity (characters per second)
        chars_per_second = bits_per_second // 8
        
        return {
            'protocol': protocol,
            'method': method,
            'capacity_per_packet_bits': capacity_per_packet,
            'estimated_packets_per_second': packets_per_second,
            'estimated_bits_per_second': bits_per_second,
            'estimated_chars_per_second': chars_per_second,
            'estimated_capacity_1min_chars': chars_per_second * 60,
            'data_depth': self.data_depth,
        }
    
    def fit(self, *args, **kwargs):
        """
        Train the model on network data.
        
        Network steganography typically doesn't require training.
        """
        logger.info(f"No training needed for {self.method} method")
        return None
    
    @classmethod
    def load(cls, path=None, method='header', protocol='tcp', cuda=False, verbose=False):
        """
        Load a model from disk or create a new one.
        
        Args:
            path: Path to saved model (optional)
            method: Steganography method (default: 'header')
            protocol: Network protocol (default: 'tcp')
            cuda: Whether to use GPU if available
            verbose: Whether to print verbose output
            
        Returns:
            NetworkStegoNet: Loaded or new model
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
        return cls(method=method, protocol=protocol, cuda=cuda, verbose=verbose)
    
    def _encode_header(self, target: str, message_bits: List[int], port: int) -> bool:
        """
        Encode message in protocol headers.
        
        Args:
            target: Target hostname or IP address
            message_bits: Bits to encode
            port: Target port
            
        Returns:
            bool: Success status
        """
        try:
            if not hasattr(self, 'scapy'):
                raise ImportError("Scapy library is required for header steganography")
            
            # Resolve hostname to IP if needed
            try:
                target_ip = socket.gethostbyname(target)
            except socket.gaierror:
                raise ValueError(f"Could not resolve hostname: {target}")
            
            # Determine how to split bits across packets
            if self.protocol == 'tcp':
                # For TCP, we can use sequence numbers, window size, etc.
                bits_per_packet = 16
            elif self.protocol == 'udp':
                # For UDP, we can use port numbers and length
                bits_per_packet = 8
            elif self.protocol == 'icmp':
                # For ICMP, we can use ID, sequence numbers, etc.
                bits_per_packet = 16
            else:
                raise ValueError(f"Header steganography not implemented for {self.protocol}")
            
            # Split message bits into chunks
            chunks = [message_bits[i:i+bits_per_packet] 
                      for i in range(0, len(message_bits), bits_per_packet)]
            
            # Prepare and send packets
            for i, chunk in enumerate(chunks):
                if self.verbose:
                    logger.info(f"Sending packet {i+1}/{len(chunks)} with {len(chunk)} bits")
                
                # Pad chunk if needed
                padded_chunk = chunk + [0] * (bits_per_packet - len(chunk))
                
                # Create packet based on protocol
                if self.protocol == 'tcp':
                    # Create TCP packet with hidden data
                    # Use the first 16 bits to modify window size
                    win_size = 0
                    for j, bit in enumerate(padded_chunk[:16]):
                        win_size |= (bit << j)
                    
                    # Basic TCP packet
                    packet = self.scapy.IP(dst=target_ip) / \
                             self.scapy.TCP(dport=port, window=win_size)
                
                elif self.protocol == 'udp':
                    # Create UDP packet with hidden data
                    # Use the bits to modify src port
                    src_port = 12345  # Base port
                    for j, bit in enumerate(padded_chunk[:8]):
                        src_port |= (bit << j)
                    
                    packet = self.scapy.IP(dst=target_ip) / \
                             self.scapy.UDP(sport=src_port, dport=port)
                
                elif self.protocol == 'icmp':
                    # Create ICMP packet with hidden data
                    # Use the bits to modify ID and sequence
                    icmp_id = 1000  # Base ID
                    icmp_seq = 0
                    
                    # First 8 bits go to ID
                    for j, bit in enumerate(padded_chunk[:8]):
                        icmp_id |= (bit << j)
                    
                    # Next 8 bits go to sequence
                    for j, bit in enumerate(padded_chunk[8:16]):
                        icmp_seq |= (bit << j)
                    
                    packet = self.scapy.IP(dst=target_ip) / \
                             self.scapy.ICMP(id=icmp_id, seq=icmp_seq)
                
                # Send the packet
                self.scapy.send(packet, verbose=0)
                
                # Small delay to avoid flooding
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in header encoding: {e}")
            return False
    
    def _decode_header(self, listen_time: int, callback: Optional[Callable]) -> List[int]:
        """
        Decode message from protocol headers.
        
        Args:
            listen_time: How long to listen (in seconds)
            callback: Optional progress callback
            
        Returns:
            list: Extracted message bits
        """
        try:
            if not hasattr(self, 'scapy'):
                raise ImportError("Scapy library is required for header steganography")
            
            extracted_bits = []
            start_time = time.time()
            
            # Craft filter based on protocol
            if self.protocol == 'tcp':
                filter_str = f"tcp and port {self.port}"
            elif self.protocol == 'udp':
                filter_str = f"udp and port {self.port}"
            elif self.protocol == 'icmp':
                filter_str = "icmp"
            else:
                raise ValueError(f"Header steganography not implemented for {self.protocol}")
            
            # Define packet handler
            def packet_handler(packet):
                nonlocal extracted_bits
                
                if self.protocol == 'tcp' and packet.haslayer(self.scapy.TCP):
                    # Extract window size
                    win_size = packet[self.scapy.TCP].window
                    
                    # Convert to bits
                    for j in range(16):
                        extracted_bits.append((win_size >> j) & 1)
                
                elif self.protocol == 'udp' and packet.haslayer(self.scapy.UDP):
                    # Extract source port
                    src_port = packet[self.scapy.UDP].sport
                    
                    # Convert to bits
                    for j in range(8):
                        extracted_bits.append((src_port >> j) & 1)
                
                elif self.protocol == 'icmp' and packet.haslayer(self.scapy.ICMP):
                    # Extract ID and sequence
                    icmp_id = packet[self.scapy.ICMP].id
                    icmp_seq = packet[self.scapy.ICMP].seq
                    
                    # Convert to bits
                    for j in range(8):
                        extracted_bits.append((icmp_id >> j) & 1)
                    
                    for j in range(8):
                        extracted_bits.append((icmp_seq >> j) & 1)
                        
                # Update progress if callback provided
                if callback:
                    elapsed = time.time() - start_time
                    progress = min(1.0, elapsed / listen_time)
                    callback(progress, len(extracted_bits) // 8)
                
                # Check for terminator
                if len(extracted_bits) >= 32:
                    last_32 = extracted_bits[-32:]
                    if all(bit == 0 for bit in last_32):
                        # Stop sniffing when terminator found
                        return True
            
            # Start sniffing
            if self.verbose:
                logger.info(f"Listening for {listen_time}s with filter: {filter_str}")
                
            self.scapy.sniff(
                filter=filter_str,
                prn=packet_handler,
                timeout=listen_time,
                store=False,
                stop_filter=lambda p: time.time() - start_time > listen_time
            )
            
            return extracted_bits
            
        except Exception as e:
            logger.error(f"Error in header decoding: {e}")
            return []
    
    def _encode_timing(self, target: str, message_bits: List[int], port: int) -> bool:
        """
        Encode message through packet timing.
        
        Args:
            target: Target hostname or IP address
            message_bits: Bits to encode
            port: Target port
            
        Returns:
            bool: Success status
        """
        try:
            # Determine timing scheme
            # 0 = short delay, 1 = long delay
            short_delay = 0.1  # seconds
            long_delay = 0.3   # seconds
            
            # Create socket
            if self.protocol == 'tcp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                # Try to connect
                try:
                    sock.connect((target, port))
                except ConnectionRefusedError:
                    logger.error(f"Connection refused to {target}:{port}")
                    return False
                
            elif self.protocol == 'udp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            else:
                raise ValueError(f"Timing steganography not implemented for {self.protocol}")
            
            # Send packets with timing
            for i, bit in enumerate(message_bits):
                if self.verbose and i % 8 == 0:
                    logger.info(f"Sending bit {i+1}/{len(message_bits)}")
                
                # Send a packet
                if self.protocol == 'tcp':
                    sock.send(b"X")
                elif self.protocol == 'udp':
                    sock.sendto(b"X", (target, port))
                
                # Delay based on bit value
                if bit == 0:
                    time.sleep(short_delay)
                else:
                    time.sleep(long_delay)
            
            # Send termination signal (a series of packets with consistent timing)
            for _ in range(5):
                if self.protocol == 'tcp':
                    sock.send(b"X")
                elif self.protocol == 'udp':
                    sock.sendto(b"X", (target, port))
                time.sleep(0.2)
            
            # Close socket
            sock.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in timing encoding: {e}")
            return False
    
    def _decode_timing(self, listen_time: int, callback: Optional[Callable]) -> List[int]:
        """
        Decode message from packet timing.
        
        Args:
            listen_time: How long to listen (in seconds)
            callback: Optional progress callback
            
        Returns:
            list: Extracted message bits
        """
        try:
            # Create listening socket
            if self.protocol == 'tcp':
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(('0.0.0.0', self.port))
                listener.listen(1)
                listener.settimeout(1.0)  # 1 second timeout for accept
            elif self.protocol == 'udp':
                listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                listener.bind(('0.0.0.0', self.port))
                listener.settimeout(1.0)
            else:
                raise ValueError(f"Timing steganography not implemented for {self.protocol}")
            
            # Timing threshold to differentiate 0s and 1s
            threshold = 0.2  # seconds
            
            # Variables for tracking
            packet_times = []
            extracted_bits = []
            start_time = time.time()
            
            # For TCP, we need to accept a connection
            if self.protocol == 'tcp':
                try:
                    client_socket, _ = listener.accept()
                    client_socket.settimeout(1.0)
                except socket.timeout:
                    logger.warning("No TCP connection within timeout")
                    return []
            
            # Main listening loop
            last_packet_time = time.time()
            
            while time.time() - start_time < listen_time:
                try:
                    if self.protocol == 'tcp':
                        # Receive from TCP socket
                        data = client_socket.recv(1024)
                        if not data:
                            # Connection closed
                            break
                    elif self.protocol == 'udp':
                        # Receive from UDP socket
                        data, _ = listener.recvfrom(1024)
                    
                    # Record packet arrival time
                    current_time = time.time()
                    interval = current_time - last_packet_time
                    packet_times.append(interval)
                    last_packet_time = current_time
                    
                    if len(packet_times) > 1:  # Skip the first interval
                        # Determine bit value based on timing
                        if packet_times[-1] < threshold:
                            extracted_bits.append(0)  # Short delay = 0
                        else:
                            extracted_bits.append(1)  # Long delay = 1
                    
                    # Update progress if callback provided
                    if callback:
                        elapsed = time.time() - start_time
                        progress = min(1.0, elapsed / listen_time)
                        callback(progress, len(extracted_bits) // 8)
                    
                    # Check for termination pattern
                    # A series of consistent timings indicates end of transmission
                    if len(packet_times) >= 5:
                        last_5 = packet_times[-5:]
                        avg = sum(last_5) / 5
                        if all(abs(t - avg) < 0.05 for t in last_5):
                            if self.verbose:
                                logger.info("Detected termination pattern")
                            break
                
                except socket.timeout:
                    # No packet received within timeout
                    continue
                except Exception as e:
                    logger.error(f"Error receiving packet: {e}")
                    break
            
            # Close sockets
            if self.protocol == 'tcp':
                try:
                    client_socket.close()
                except:
                    pass
            listener.close()
            
            return extracted_bits
            
        except Exception as e:
            logger.error(f"Error in timing decoding: {e}")
            return []
    
    def _encode_size(self, target: str, message_bits: List[int], port: int) -> bool:
        """
        Encode message through packet size manipulation.
        
        Args:
            target: Target hostname or IP address
            message_bits: Bits to encode
            port: Target port
            
        Returns:
            bool: Success status
        """
        try:
            # Define packet sizes for 0 and 1
            # The sizes should be different enough to be distinguishable
            # but not so different as to be easily detected
            size_0 = 40   # bytes (minimum packet size)
            size_1 = 100  # bytes
            
            # Create socket
            if self.protocol == 'tcp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                # Try to connect
                try:
                    sock.connect((target, port))
                except ConnectionRefusedError:
                    logger.error(f"Connection refused to {target}:{port}")
                    return False
                
            elif self.protocol == 'udp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            else:
                raise ValueError(f"Size steganography not implemented for {self.protocol}")
            
            # Send packets with varying sizes
            for i, bit in enumerate(message_bits):
                if self.verbose and i % 8 == 0:
                    logger.info(f"Sending bit {i+1}/{len(message_bits)}")
                
                # Determine packet size based on bit
                if bit == 0:
                    data = b"X" * size_0
                else:
                    data = b"X" * size_1
                
                # Send the packet
                if self.protocol == 'tcp':
                    sock.send(data)
                elif self.protocol == 'udp':
                    sock.sendto(data, (target, port))
                
                # Small delay between packets
                time.sleep(0.1)
            
            # Send termination signal (alternating sizes)
            for _ in range(6):
                term_data = b"X" * (size_0 if _ % 2 == 0 else size_1)
                if self.protocol == 'tcp':
                    sock.send(term_data)
                elif self.protocol == 'udp':
                    sock.sendto(term_data, (target, port))
                time.sleep(0.1)
            
            # Close socket
            sock.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in size encoding: {e}")
            return False
    
    def _decode_size(self, listen_time: int, callback: Optional[Callable]) -> List[int]:
        """
        Decode message from packet size variations.
        
        Args:
            listen_time: How long to listen (in seconds)
            callback: Optional progress callback
            
        Returns:
            list: Extracted message bits
        """
        try:
            # Create listening socket
            if self.protocol == 'tcp':
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(('0.0.0.0', self.port))
                listener.listen(1)
                listener.settimeout(1.0)  # 1 second timeout for accept
            elif self.protocol == 'udp':
                listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                listener.bind(('0.0.0.0', self.port))
                listener.settimeout(1.0)
            else:
                raise ValueError(f"Size steganography not implemented for {self.protocol}")
            
            # Define threshold for size differentiation
            size_threshold = 70  # bytes (between size_0 and size_1)
            
            # Variables for tracking
            packet_sizes = []
            extracted_bits = []
            start_time = time.time()
            
            # For TCP, we need to accept a connection
            if self.protocol == 'tcp':
                try:
                    client_socket, _ = listener.accept()
                    client_socket.settimeout(1.0)
                except socket.timeout:
                    logger.warning("No TCP connection within timeout")
                    return []
            
            # Main listening loop
            while time.time() - start_time < listen_time:
                try:
                    if self.protocol == 'tcp':
                        # Receive from TCP socket
                        data = client_socket.recv(1024)
                        if not data:
                            # Connection closed
                            break
                    elif self.protocol == 'udp':
                        # Receive from UDP socket
                        data, _ = listener.recvfrom(1024)
                    
                    # Record packet size
                    packet_size = len(data)
                    packet_sizes.append(packet_size)
                    
                    # Determine bit value based on size
                    if packet_size < size_threshold:
                        extracted_bits.append(0)  # Small packet = 0
                    else:
                        extracted_bits.append(1)  # Large packet = 1
                    
                    # Update progress if callback provided
                    if callback:
                        elapsed = time.time() - start_time
                        progress = min(1.0, elapsed / listen_time)
                        callback(progress, len(extracted_bits) // 8)
                    
                    # Check for termination pattern
                    # 6 alternating sizes indicates end of transmission
                    if len(packet_sizes) >= 6:
                        last_6 = packet_sizes[-6:]
                        if (all(last_6[i] < size_threshold and last_6[i+1] >= size_threshold for i in range(0, 5, 2)) or
                            all(last_6[i] >= size_threshold and last_6[i+1] < size_threshold for i in range(0, 5, 2))):
                            if self.verbose:
                                logger.info("Detected termination pattern")
                            # Remove the termination pattern from extracted bits
                            extracted_bits = extracted_bits[:-6]
                            break
                
                except socket.timeout:
                    # No packet received within timeout
                    continue
                except Exception as e:
                    logger.error(f"Error receiving packet: {e}")
                    break
            
            # Close sockets
            if self.protocol == 'tcp':
                try:
                    client_socket.close()
                except:
                    pass
            listener.close()
            
            return extracted_bits
            
        except Exception as e:
            logger.error(f"Error in size decoding: {e}")
            return []
    
    def _encode_sequence(self, target: str, message_bits: List[int], port: int) -> bool:
        """
        Encode message through packet sequence manipulation.
        
        Args:
            target: Target hostname or IP address
            message_bits: Bits to encode
            port: Target port
            
        Returns:
            bool: Success status
        """
        try:
            if not hasattr(self, 'scapy'):
                raise ImportError("Scapy library is required for sequence steganography")
            
            # Resolve hostname to IP if needed
            try:
                target_ip = socket.gethostbyname(target)
            except socket.gaierror:
                raise ValueError(f"Could not resolve hostname: {target}")
            
            # For sequence steganography, we'll use source port ordering
            # Even-indexed packets use low source ports for 0, high for 1
            # Odd-indexed packets use high source ports for 0, low for 1
            low_port_start = 10000
            high_port_start = 50000
            
            # Create and send packets
            for i, bit in enumerate(message_bits):
                if self.verbose and i % 8 == 0:
                    logger.info(f"Sending bit {i+1}/{len(message_bits)}")
                
                # Determine source port based on bit and index
                if (i % 2 == 0 and bit == 0) or (i % 2 == 1 and bit == 1):
                    # Use low port
                    src_port = low_port_start + (i % 1000)
                else:
                    # Use high port
                    src_port = high_port_start + (i % 1000)
                
                # Create packet based on protocol
                if self.protocol == 'tcp':
                    packet = self.scapy.IP(dst=target_ip) / \
                             self.scapy.TCP(sport=src_port, dport=port)
                elif self.protocol == 'udp':
                    packet = self.scapy.IP(dst=target_ip) / \
                             self.scapy.UDP(sport=src_port, dport=port)
                else:
                    raise ValueError(f"Sequence steganography not implemented for {self.protocol}")
                
                # Send the packet
                self.scapy.send(packet, verbose=0)
                
                # Small delay between packets
                time.sleep(0.05)
            
            # Send termination signal (a sequence of packets with incrementing ports)
            for i in range(8):
                term_port = 30000 + i
                if self.protocol == 'tcp':
                    packet = self.scapy.IP(dst=target_ip) / \
                             self.scapy.TCP(sport=term_port, dport=port)
                elif self.protocol == 'udp':
                    packet = self.scapy.IP(dst=target_ip) / \
                             self.scapy.UDP(sport=term_port, dport=port)
                
                self.scapy.send(packet, verbose=0)
                time.sleep(0.05)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in sequence encoding: {e}")
            return False
    
    def _decode_sequence(self, listen_time: int, callback: Optional[Callable]) -> List[int]:
        """
        Decode message from packet sequence patterns.
        
        Args:
            listen_time: How long to listen (in seconds)
            callback: Optional progress callback
            
        Returns:
            list: Extracted message bits
        """
        try:
            if not hasattr(self, 'scapy'):
                raise ImportError("Scapy library is required for sequence steganography")
            
            extracted_bits = []
            source_ports = []
            start_time = time.time()
            
            # Port thresholds
            port_threshold = 30000
            
            # Craft filter based on protocol
            if self.protocol == 'tcp':
                filter_str = f"tcp and dst port {self.port}"
            elif self.protocol == 'udp':
                filter_str = f"udp and dst port {self.port}"
            else:
                raise ValueError(f"Sequence steganography not implemented for {self.protocol}")
            
            # Define packet handler
            def packet_handler(packet):
                nonlocal extracted_bits, source_ports
                
                if self.protocol == 'tcp' and packet.haslayer(self.scapy.TCP):
                    src_port = packet[self.scapy.TCP].sport
                elif self.protocol == 'udp' and packet.haslayer(self.scapy.UDP):
                    src_port = packet[self.scapy.UDP].sport
                else:
                    return
                
                # Store the source port
                source_ports.append(src_port)
                
                # Process based on port pattern
                if len(source_ports) > 1:
                    # Check for termination pattern (8 sequential ports)
                    if len(source_ports) >= 8:
                        last_8 = source_ports[-8:]
                        if all(abs(last_8[i] - (30000 + i)) < 10 for i in range(8)):
                            if self.verbose:
                                logger.info("Detected termination pattern")
                            return True
                    
                    # Determine bit based on port and position
                    i = len(source_ports) - 1
                    port = source_ports[i]
                    
                    if (i % 2 == 0 and port < port_threshold) or (i % 2 == 1 and port >= port_threshold):
                        extracted_bits.append(0)
                    else:
                        extracted_bits.append(1)
                
                # Update progress if callback provided
                if callback:
                    elapsed = time.time() - start_time
                    progress = min(1.0, elapsed / listen_time)
                    callback(progress, len(extracted_bits) // 8)
            
            # Start sniffing
            if self.verbose:
                logger.info(f"Listening for {listen_time}s with filter: {filter_str}")
                
            self.scapy.sniff(
                filter=filter_str,
                prn=packet_handler,
                timeout=listen_time,
                store=False,
                stop_filter=lambda p: time.time() - start_time > listen_time
            )
            
            return extracted_bits
            
        except Exception as e:
            logger.error(f"Error in sequence decoding: {e}")
            return []
    
    def _encode_covert_channel(self, target: str, message_bits: List[int], port: int) -> bool:
        """
        Encode message using a covert channel protocol.
        
        Args:
            target: Target hostname or IP address
            message_bits: Bits to encode
            port: Target port
            
        Returns:
            bool: Success status
        """
        try:
            # For covert channel, we'll implement a custom protocol
            # First byte is the secret pattern, followed by the data
            
            # Chunk size (bytes per packet)
            chunk_size = 8
            
            # Convert bits to bytes
            message_bytes = bits_to_bytearray(message_bits)
            
            # Create socket
            if self.protocol == 'tcp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                # Try to connect
                try:
                    sock.connect((target, port))
                except ConnectionRefusedError:
                    logger.error(f"Connection refused to {target}:{port}")
                    return False
                
            elif self.protocol == 'udp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            else:
                raise ValueError(f"Covert channel not implemented for {self.protocol}")
            
            # Split message into chunks and send
            for i in range(0, len(message_bytes), chunk_size):
                chunk = message_bytes[i:i+chunk_size]
                
                if self.verbose and i % chunk_size == 0:
                    logger.info(f"Sending chunk {i//chunk_size + 1}/{(len(message_bytes) + chunk_size - 1)//chunk_size}")
                
                # Add secret pattern and sequence number to chunk
                seq_num = (i // chunk_size).to_bytes(4, byteorder='big')
                packet = self.secret_pattern + seq_num + chunk
                
                # Send the packet
                if self.protocol == 'tcp':
                    sock.send(packet)
                elif self.protocol == 'udp':
                    sock.sendto(packet, (target, port))
                
                # Small delay between packets
                time.sleep(0.05)
            
            # Send termination packet
            term_packet = self.secret_pattern + b'\xff\xff\xff\xff'
            
            if self.protocol == 'tcp':
                sock.send(term_packet)
            elif self.protocol == 'udp':
                sock.sendto(term_packet, (target, port))
            
            # Close socket
            sock.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in covert channel encoding: {e}")
            return False
    
    def _decode_covert_channel(self, listen_time: int, callback: Optional[Callable]) -> List[int]:
        """
        Decode message from covert channel communication.
        
        Args:
            listen_time: How long to listen (in seconds)
            callback: Optional progress callback
            
        Returns:
            list: Extracted message bits
        """
        try:
            # Create listening socket
            if self.protocol == 'tcp':
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(('0.0.0.0', self.port))
                listener.listen(1)
                listener.settimeout(1.0)  # 1 second timeout for accept
            elif self.protocol == 'udp':
                listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                listener.bind(('0.0.0.0', self.port))
                listener.settimeout(1.0)
            else:
                raise ValueError(f"Covert channel not implemented for {self.protocol}")
            
            # Variables for tracking
            chunks = {}  # Using dict to handle out-of-order packets
            start_time = time.time()
            secret_pattern_len = len(self.secret_pattern)
            
            # For TCP, we need to accept a connection
            if self.protocol == 'tcp':
                try:
                    client_socket, _ = listener.accept()
                    client_socket.settimeout(1.0)
                except socket.timeout:
                    logger.warning("No TCP connection within timeout")
                    return []
            
            # Main listening loop
            while time.time() - start_time < listen_time:
                try:
                    if self.protocol == 'tcp':
                        # Receive from TCP socket
                        data = client_socket.recv(1024)
                        if not data:
                            # Connection closed
                            break
                    elif self.protocol == 'udp':
                        # Receive from UDP socket
                        data, _ = listener.recvfrom(1024)
                    
                    # Check if packet starts with secret pattern
                    if data[:secret_pattern_len] == self.secret_pattern:
                        # Extract sequence number
                        seq_bytes = data[secret_pattern_len:secret_pattern_len+4]
                        seq_num = int.from_bytes(seq_bytes, byteorder='big')
                        
                        # Check for termination packet
                        if seq_num == 0xFFFFFFFF:
                            if self.verbose:
                                logger.info("Received termination packet")
                            break
                        
                        # Extract chunk
                        chunk = data[secret_pattern_len+4:]
                        chunks[seq_num] = chunk
                        
                        # Update progress if callback provided
                        if callback:
                            elapsed = time.time() - start_time
                            progress = min(1.0, elapsed / listen_time)
                            callback(progress, sum(len(c) for c in chunks.values()))
                
                except socket.timeout:
                    # No packet received within timeout
                    continue
                except Exception as e:
                    logger.error(f"Error receiving packet: {e}")
                    break
            
            # Close sockets
            if self.protocol == 'tcp':
                try:
                    client_socket.close()
                except:
                    pass
            listener.close()
            
            # Combine chunks in order
            ordered_chunks = [chunks[i] for i in range(len(chunks))]
            message_bytes = b''.join(ordered_chunks)
            
            # Convert to bits
            message_bits = []
            for byte in message_bytes:
                for i in range(7, -1, -1):
                    message_bits.append((byte >> i) & 1)
            
            return message_bits
            
        except Exception as e:
            logger.error(f"Error in covert channel decoding: {e}")
            return []