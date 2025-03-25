# -*- coding: utf-8 -*-
"""
Network protocol helpers for Stego-AI.

This module provides protocol-specific implementations and utilities
for the network steganography methods in Stego-AI.
"""

import os
import logging
import socket
import struct
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

# Set up logging
logger = logging.getLogger(__name__)


class ProtocolHandler:
    """Base class for protocol-specific handlers."""
    
    def __init__(self, interface: str = 'eth0', port: int = 8000, verbose: bool = False):
        """
        Initialize the protocol handler.
        
        Args:
            interface: Network interface to use
            port: Network port for applicable protocols
            verbose: Whether to print verbose output
        """
        self.interface = interface
        self.port = port
        self.verbose = verbose
    
    def create_packet(self, *args, **kwargs) -> Any:
        """Create a packet for the specific protocol."""
        raise NotImplementedError("Subclasses must implement create_packet")
    
    def send_packet(self, *args, **kwargs) -> bool:
        """Send a packet using the specific protocol."""
        raise NotImplementedError("Subclasses must implement send_packet")
    
    def receive_packet(self, *args, **kwargs) -> Any:
        """Receive a packet using the specific protocol."""
        raise NotImplementedError("Subclasses must implement receive_packet")
    
    def extract_data(self, *args, **kwargs) -> List[int]:
        """Extract steganographic data from a packet."""
        raise NotImplementedError("Subclasses must implement extract_data")


class TCPHandler(ProtocolHandler):
    """Handler for TCP protocol steganography."""
    
    def __init__(self, interface: str = 'eth0', port: int = 8000, verbose: bool = False):
        """Initialize the TCP handler."""
        super().__init__(interface, port, verbose)
        self.scapy_available = self._check_scapy()
    
    def _check_scapy(self) -> bool:
        """Check if Scapy is available."""
        try:
            import scapy.all as scapy
            self.scapy = scapy
            return True
        except ImportError:
            if self.verbose:
                logger.warning("Scapy not available, falling back to socket implementation")
            return False
    
    def create_packet(self, target: str, data: Union[bytes, List[int]], method: str) -> Any:
        """
        Create a TCP packet with hidden data.
        
        Args:
            target: Target IP or hostname
            data: Data to hide (bytes or list of bits)
            method: Steganography method to use
            
        Returns:
            Packet object or bytes
        """
        if self.scapy_available:
            return self._create_packet_scapy(target, data, method)
        else:
            return self._create_packet_socket(target, data, method)
    
    def _create_packet_scapy(self, target: str, data: Union[bytes, List[int]], method: str) -> Any:
        """Create a TCP packet using Scapy."""
        try:
            # Resolve hostname to IP if needed
            target_ip = socket.gethostbyname(target)
            
            if method == 'header':
                # Hide data in TCP header fields
                if isinstance(data, list):
                    # Convert bit list to a number for window size
                    window_size = 0
                    for i, bit in enumerate(data[:16]):
                        window_size |= (bit << i)
                else:
                    # Use first 2 bytes of data for window size
                    window_size = int.from_bytes(data[:2], byteorder='little')
                
                # Create packet with modified window size
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.TCP(dport=self.port, window=window_size)
                
                # Set sequence number for additional data if needed
                if isinstance(data, list) and len(data) > 16:
                    seq_num = 0
                    for i, bit in enumerate(data[16:32]):
                        seq_num |= (bit << i)
                    packet[self.scapy.TCP].seq = seq_num
                
                return packet
            
            elif method == 'sequence':
                # Use a specific source port based on data
                src_port = 1024 + (int.from_bytes(data[:2], byteorder='little') % 64511)
                
                # Create basic packet
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.TCP(sport=src_port, dport=self.port)
                
                return packet
            
            else:
                # For other methods, create a basic packet
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.TCP(dport=self.port) / data
                
                return packet
        
        except Exception as e:
            logger.error(f"Error creating TCP packet with Scapy: {e}")
            return None
    
    def _create_packet_socket(self, target: str, data: Union[bytes, List[int]], method: str) -> bytes:
        """Create a TCP packet using sockets (limited functionality)."""
        # This is a simplified implementation since raw packet creation 
        # without Scapy is complex and often platform-dependent
        if isinstance(data, list):
            # Convert bit list to bytes
            byte_data = bytearray()
            for i in range(0, len(data), 8):
                byte = 0
                for j in range(min(8, len(data) - i)):
                    byte |= (data[i + j] << j)
                byte_data.append(byte)
            data = bytes(byte_data)
        
        return data  # Just return the data, actual packet creation happens in send_packet
    
    def send_packet(self, target: str, packet: Any) -> bool:
        """
        Send a TCP packet.
        
        Args:
            target: Target IP or hostname
            packet: Packet to send (Scapy packet or bytes)
            
        Returns:
            bool: Success status
        """
        try:
            if self.scapy_available and isinstance(packet, self.scapy.Packet):
                # Send packet using Scapy
                self.scapy.send(packet, verbose=0)
                return True
            else:
                # Send data using sockets
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((target, self.port))
                
                # If packet is already bytes, send directly
                if isinstance(packet, bytes):
                    sock.send(packet)
                else:
                    # Convert to bytes if needed
                    sock.send(str(packet).encode())
                
                sock.close()
                return True
        
        except Exception as e:
            logger.error(f"Error sending TCP packet: {e}")
            return False
    
    def receive_packet(self, timeout: float = 1.0) -> Any:
        """
        Receive a TCP packet.
        
        Args:
            timeout: Receive timeout in seconds
            
        Returns:
            Received packet or None
        """
        try:
            if self.scapy_available:
                # Use Scapy to sniff packets
                packets = self.scapy.sniff(
                    filter=f"tcp and port {self.port}",
                    count=1,
                    timeout=timeout
                )
                
                if packets:
                    return packets[0]
                return None
            
            else:
                # Use sockets to receive data
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', self.port))
                sock.listen(1)
                sock.settimeout(timeout)
                
                try:
                    conn, addr = sock.accept()
                    conn.settimeout(timeout)
                    data = conn.recv(1024)
                    conn.close()
                    return data
                except socket.timeout:
                    return None
                finally:
                    sock.close()
        
        except Exception as e:
            logger.error(f"Error receiving TCP packet: {e}")
            return None
    
    def extract_data(self, packet: Any, method: str) -> List[int]:
        """
        Extract steganographic data from a TCP packet.
        
        Args:
            packet: Received packet
            method: Steganography method used
            
        Returns:
            list: Extracted data bits
        """
        try:
            if self.scapy_available and isinstance(packet, self.scapy.Packet):
                # Extract data using Scapy
                if method == 'header':
                    bits = []
                    
                    # Extract bits from window size
                    if self.scapy.TCP in packet:
                        window = packet[self.scapy.TCP].window
                        for i in range(16):
                            bits.append((window >> i) & 1)
                        
                        # Extract bits from sequence number
                        seq = packet[self.scapy.TCP].seq
                        for i in range(16):
                            bits.append((seq >> i) & 1)
                    
                    return bits
                
                elif method == 'sequence':
                    bits = []
                    
                    # Extract data from source port
                    if self.scapy.TCP in packet:
                        sport = packet[self.scapy.TCP].sport
                        # Convert source port to bits
                        for i in range(16):
                            bits.append((sport >> i) & 1)
                    
                    return bits
                
                else:
                    # Get payload
                    if self.scapy.TCP in packet and packet[self.scapy.TCP].payload:
                        # Convert bytes to bits
                        payload = bytes(packet[self.scapy.TCP].payload)
                        bits = []
                        for byte in payload:
                            for i in range(8):
                                bits.append((byte >> i) & 1)
                        return bits
                    
                    return []
            
            else:
                # Handle socket data (bytes)
                if isinstance(packet, bytes):
                    bits = []
                    for byte in packet:
                        for i in range(8):
                            bits.append((byte >> i) & 1)
                    return bits
                
                return []
        
        except Exception as e:
            logger.error(f"Error extracting data from TCP packet: {e}")
            return []


class UDPHandler(ProtocolHandler):
    """Handler for UDP protocol steganography."""
    
    def __init__(self, interface: str = 'eth0', port: int = 8000, verbose: bool = False):
        """Initialize the UDP handler."""
        super().__init__(interface, port, verbose)
        self.scapy_available = self._check_scapy()
    
    def _check_scapy(self) -> bool:
        """Check if Scapy is available."""
        try:
            import scapy.all as scapy
            self.scapy = scapy
            return True
        except ImportError:
            if self.verbose:
                logger.warning("Scapy not available, falling back to socket implementation")
            return False
    
    def create_packet(self, target: str, data: Union[bytes, List[int]], method: str) -> Any:
        """
        Create a UDP packet with hidden data.
        
        Args:
            target: Target IP or hostname
            data: Data to hide (bytes or list of bits)
            method: Steganography method to use
            
        Returns:
            Packet object or bytes
        """
        if self.scapy_available:
            return self._create_packet_scapy(target, data, method)
        else:
            return self._create_packet_socket(target, data, method)
    
    def _create_packet_scapy(self, target: str, data: Union[bytes, List[int]], method: str) -> Any:
        """Create a UDP packet using Scapy."""
        try:
            # Resolve hostname to IP if needed
            target_ip = socket.gethostbyname(target)
            
            if method == 'header':
                # Hide data in UDP header fields (limited options)
                if isinstance(data, list):
                    # Convert bit list to a number for source port
                    sport = 1024 + (sum(bit << i for i, bit in enumerate(data[:12])) % 64511)
                else:
                    # Use first 2 bytes of data for source port
                    sport = 1024 + (int.from_bytes(data[:2], byteorder='little') % 64511)
                
                # Create packet with modified source port
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.UDP(sport=sport, dport=self.port)
                
                return packet
            
            elif method == 'size':
                # Vary packet size based on data
                padding_size = 0
                if isinstance(data, list):
                    # Use first 8 bits to determine padding
                    padding_size = sum(bit << i for i, bit in enumerate(data[:8]))
                else:
                    # Use first byte to determine padding
                    padding_size = data[0] if data else 0
                
                # Create packet with padding
                padding = b'\x00' * padding_size
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.UDP(dport=self.port) / (data if isinstance(data, bytes) else b'') / padding
                
                return packet
            
            else:
                # For other methods, create a basic packet
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.UDP(dport=self.port) / (data if isinstance(data, bytes) else b'')
                
                return packet
        
        except Exception as e:
            logger.error(f"Error creating UDP packet with Scapy: {e}")
            return None
    
    def _create_packet_socket(self, target: str, data: Union[bytes, List[int]], method: str) -> bytes:
        """Create a UDP packet using sockets (limited functionality)."""
        if isinstance(data, list):
            # Convert bit list to bytes
            byte_data = bytearray()
            for i in range(0, len(data), 8):
                byte = 0
                for j in range(min(8, len(data) - i)):
                    byte |= (data[i + j] << j)
                byte_data.append(byte)
            data = bytes(byte_data)
        
        if method == 'size':
            # Add padding for size method
            padding_size = data[0] if data else 0
            data = data + b'\x00' * padding_size
        
        return data
    
    def send_packet(self, target: str, packet: Any) -> bool:
        """
        Send a UDP packet.
        
        Args:
            target: Target IP or hostname
            packet: Packet to send (Scapy packet or bytes)
            
        Returns:
            bool: Success status
        """
        try:
            if self.scapy_available and isinstance(packet, self.scapy.Packet):
                # Send packet using Scapy
                self.scapy.send(packet, verbose=0)
                return True
            else:
                # Send data using sockets
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                # If packet is already bytes, send directly
                if isinstance(packet, bytes):
                    sock.sendto(packet, (target, self.port))
                else:
                    # Convert to bytes if needed
                    sock.sendto(str(packet).encode(), (target, self.port))
                
                sock.close()
                return True
        
        except Exception as e:
            logger.error(f"Error sending UDP packet: {e}")
            return False
    
    def receive_packet(self, timeout: float = 1.0) -> Any:
        """
        Receive a UDP packet.
        
        Args:
            timeout: Receive timeout in seconds
            
        Returns:
            Received packet or None
        """
        try:
            if self.scapy_available:
                # Use Scapy to sniff packets
                packets = self.scapy.sniff(
                    filter=f"udp and port {self.port}",
                    count=1,
                    timeout=timeout
                )
                
                if packets:
                    return packets[0]
                return None
            
            else:
                # Use sockets to receive data
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(('0.0.0.0', self.port))
                sock.settimeout(timeout)
                
                try:
                    data, addr = sock.recvfrom(1024)
                    return data
                except socket.timeout:
                    return None
                finally:
                    sock.close()
        
        except Exception as e:
            logger.error(f"Error receiving UDP packet: {e}")
            return None
    
    def extract_data(self, packet: Any, method: str) -> List[int]:
        """
        Extract steganographic data from a UDP packet.
        
        Args:
            packet: Received packet
            method: Steganography method used
            
        Returns:
            list: Extracted data bits
        """
        try:
            if self.scapy_available and isinstance(packet, self.scapy.Packet):
                # Extract data using Scapy
                if method == 'header':
                    bits = []
                    
                    # Extract bits from source port
                    if self.scapy.UDP in packet:
                        sport = packet[self.scapy.UDP].sport
                        # Convert source port to bits (12 bits since range is limited)
                        for i in range(12):
                            bits.append((sport >> i) & 1)
                    
                    return bits
                
                elif method == 'size':
                    bits = []
                    
                    # Extract data from packet size
                    if self.scapy.UDP in packet:
                        # Get payload length
                        payload_len = len(bytes(packet[self.scapy.UDP].payload))
                        # Convert length to bits
                        for i in range(8):
                            bits.append((payload_len >> i) & 1)
                    
                    return bits
                
                else:
                    # Get payload
                    if self.scapy.UDP in packet and packet[self.scapy.UDP].payload:
                        # Convert bytes to bits
                        payload = bytes(packet[self.scapy.UDP].payload)
                        bits = []
                        for byte in payload:
                            for i in range(8):
                                bits.append((byte >> i) & 1)
                        return bits
                    
                    return []
            
            else:
                # Handle socket data (bytes)
                if isinstance(packet, bytes):
                    bits = []
                    for byte in packet:
                        for i in range(8):
                            bits.append((byte >> i) & 1)
                    return bits
                
                return []
        
        except Exception as e:
            logger.error(f"Error extracting data from UDP packet: {e}")
            return []


class ICMPHandler(ProtocolHandler):
    """Handler for ICMP protocol steganography."""
    
    def __init__(self, interface: str = 'eth0', verbose: bool = False):
        """Initialize the ICMP handler."""
        super().__init__(interface, 0, verbose)  # ICMP doesn't use ports
        self.scapy_available = self._check_scapy()
    
    def _check_scapy(self) -> bool:
        """Check if Scapy is available."""
        try:
            import scapy.all as scapy
            self.scapy = scapy
            return True
        except ImportError:
            if self.verbose:
                logger.warning("Scapy not available. ICMP steganography requires Scapy")
            return False
    
    def create_packet(self, target: str, data: Union[bytes, List[int]], method: str) -> Any:
        """
        Create an ICMP packet with hidden data.
        
        Args:
            target: Target IP or hostname
            data: Data to hide (bytes or list of bits)
            method: Steganography method to use
            
        Returns:
            Packet object or None
        """
        if not self.scapy_available:
            logger.error("Scapy is required for ICMP steganography")
            return None
        
        try:
            # Resolve hostname to IP if needed
            target_ip = socket.gethostbyname(target)
            
            if method == 'header':
                # Hide data in ICMP header fields
                icmp_id = 1000
                icmp_seq = 0
                
                if isinstance(data, list):
                    # Use first 16 bits to set ID and sequence
                    if len(data) >= 8:
                        icmp_id = 1000 + sum(bit << i for i, bit in enumerate(data[:8]))
                    if len(data) >= 16:
                        icmp_seq = sum(bit << i for i, bit in enumerate(data[8:16]))
                else:
                    # Use first 4 bytes to set ID and sequence
                    if len(data) >= 2:
                        icmp_id = 1000 + int.from_bytes(data[:2], byteorder='little')
                    if len(data) >= 4:
                        icmp_seq = int.from_bytes(data[2:4], byteorder='little')
                
                # Create packet with modified ID and sequence
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.ICMP(id=icmp_id, seq=icmp_seq, type=8)  # Echo request
                
                return packet
            
            elif method == 'size':
                # Vary packet size based on data
                padding_size = 0
                if isinstance(data, list):
                    # Use first 8 bits to determine padding
                    padding_size = sum(bit << i for i, bit in enumerate(data[:8]))
                else:
                    # Use first byte to determine padding
                    padding_size = data[0] if data else 0
                
                # Create packet with padding
                padding = b'\x00' * padding_size
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.ICMP(type=8) / (data if isinstance(data, bytes) else b'') / padding
                
                return packet
            
            else:
                # For other methods, hide data in payload
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.ICMP(type=8) / (data if isinstance(data, bytes) else b'')
                
                return packet
        
        except Exception as e:
            logger.error(f"Error creating ICMP packet: {e}")
            return None
    
    def send_packet(self, target: str, packet: Any) -> bool:
        """
        Send an ICMP packet.
        
        Args:
            target: Target IP or hostname
            packet: Packet to send (Scapy packet)
            
        Returns:
            bool: Success status
        """
        if not self.scapy_available:
            logger.error("Scapy is required for ICMP steganography")
            return False
        
        try:
            # Send packet using Scapy
            self.scapy.send(packet, verbose=0)
            return True
        
        except Exception as e:
            logger.error(f"Error sending ICMP packet: {e}")
            return False
    
    def receive_packet(self, timeout: float = 1.0) -> Any:
        """
        Receive an ICMP packet.
        
        Args:
            timeout: Receive timeout in seconds
            
        Returns:
            Received packet or None
        """
        if not self.scapy_available:
            logger.error("Scapy is required for ICMP steganography")
            return None
        
        try:
            # Use Scapy to sniff packets
            packets = self.scapy.sniff(
                filter="icmp",
                count=1,
                timeout=timeout
            )
            
            if packets:
                return packets[0]
            return None
        
        except Exception as e:
            logger.error(f"Error receiving ICMP packet: {e}")
            return None
    
    def extract_data(self, packet: Any, method: str) -> List[int]:
        """
        Extract steganographic data from an ICMP packet.
        
        Args:
            packet: Received packet
            method: Steganography method used
            
        Returns:
            list: Extracted data bits
        """
        if not self.scapy_available:
            logger.error("Scapy is required for ICMP steganography")
            return []
        
        try:
            # Extract data using Scapy
            if method == 'header':
                bits = []
                
                # Extract bits from ID and sequence
                if self.scapy.ICMP in packet:
                    icmp_id = packet[self.scapy.ICMP].id
                    icmp_seq = packet[self.scapy.ICMP].seq
                    
                    # Convert ID to bits (8 bits)
                    for i in range(8):
                        bits.append((icmp_id >> i) & 1)
                    
                    # Convert sequence to bits (8 bits)
                    for i in range(8):
                        bits.append((icmp_seq >> i) & 1)
                
                return bits
            
            elif method == 'size':
                bits = []
                
                # Extract data from packet size
                if self.scapy.ICMP in packet:
                    # Get payload length
                    payload_len = len(bytes(packet[self.scapy.ICMP].payload))
                    # Convert length to bits
                    for i in range(8):
                        bits.append((payload_len >> i) & 1)
                
                return bits
            
            else:
                # Get payload
                if self.scapy.ICMP in packet and packet[self.scapy.ICMP].payload:
                    # Convert bytes to bits
                    payload = bytes(packet[self.scapy.ICMP].payload)
                    bits = []
                    for byte in payload:
                        for i in range(8):
                            bits.append((byte >> i) & 1)
                    return bits
                
                return []
        
        except Exception as e:
            logger.error(f"Error extracting data from ICMP packet: {e}")
            return []


class DNSHandler(ProtocolHandler):
    """Handler for DNS protocol steganography."""
    
    def __init__(self, interface: str = 'eth0', port: int = 53, verbose: bool = False):
        """Initialize the DNS handler."""
        super().__init__(interface, port, verbose)
        self.scapy_available = self._check_scapy()
    
    def _check_scapy(self) -> bool:
        """Check if Scapy is available."""
        try:
            import scapy.all as scapy
            self.scapy = scapy
            return True
        except ImportError:
            if self.verbose:
                logger.warning("Scapy not available. DNS steganography requires Scapy")
            return False
    
    def create_packet(self, target: str, data: Union[bytes, List[int]], method: str) -> Any:
        """
        Create a DNS packet with hidden data.
        
        Args:
            target: Target IP or hostname
            data: Data to hide (bytes or list of bits)
            method: Steganography method to use
            
        Returns:
            Packet object or None
        """
        if not self.scapy_available:
            logger.error("Scapy is required for DNS steganography")
            return None
        
        try:
            # Resolve hostname to IP if needed
            target_ip = socket.gethostbyname(target)
            
            if method == 'domain':
                # Hide data in the queried domain
                domain = "stegoai.com"
                
                if isinstance(data, list):
                    # Convert bits to hex for a subdomain
                    hex_str = ""
                    for i in range(0, len(data), 4):
                        nybble = 0
                        for j in range(min(4, len(data) - i)):
                            nybble |= (data[i + j] << j)
                        hex_str += format(nybble, 'x')
                    
                    domain = f"{hex_str}.{domain}"
                
                elif isinstance(data, bytes):
                    # Convert bytes to hex for a subdomain
                    hex_str = data.hex()[:30]  # Limit length for DNS constraint
                    domain = f"{hex_str}.{domain}"
                
                # Create DNS query
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.UDP(dport=53) / \
                         self.scapy.DNS(rd=1, qd=self.scapy.DNSQR(qname=domain))
                
                return packet
            
            elif method == 'txid':
                # Hide data in the DNS transaction ID
                txid = 0
                
                if isinstance(data, list):
                    # Use first 16 bits for txid
                    for i in range(min(16, len(data))):
                        txid |= (data[i] << i)
                
                elif isinstance(data, bytes) and len(data) >= 2:
                    # Use first 2 bytes for txid
                    txid = int.from_bytes(data[:2], byteorder='little')
                
                # Create DNS query with specific txid
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.UDP(dport=53) / \
                         self.scapy.DNS(id=txid, rd=1, qd=self.scapy.DNSQR(qname="stegoai.com"))
                
                return packet
            
            else:
                # For other methods, create a basic DNS query
                packet = self.scapy.IP(dst=target_ip) / \
                         self.scapy.UDP(dport=53) / \
                         self.scapy.DNS(rd=1, qd=self.scapy.DNSQR(qname="stegoai.com"))
                
                return packet
        
        except Exception as e:
            logger.error(f"Error creating DNS packet: {e}")
            return None
    
    def send_packet(self, target: str, packet: Any) -> bool:
        """
        Send a DNS packet.
        
        Args:
            target: Target IP or hostname
            packet: Packet to send (Scapy packet)
            
        Returns:
            bool: Success status
        """
        if not self.scapy_available:
            logger.error("Scapy is required for DNS steganography")
            return False
        
        try:
            # Send packet using Scapy
            self.scapy.send(packet, verbose=0)
            return True
        
        except Exception as e:
            logger.error(f"Error sending DNS packet: {e}")
            return False
    
    def receive_packet(self, timeout: float = 1.0) -> Any:
        """
        Receive a DNS packet.
        
        Args:
            timeout: Receive timeout in seconds
            
        Returns:
            Received packet or None
        """
        if not self.scapy_available:
            logger.error("Scapy is required for DNS steganography")
            return None
        
        try:
            # Use Scapy to sniff packets
            packets = self.scapy.sniff(
                filter=f"udp and port {self.port}",
                count=1,
                timeout=timeout
            )
            
            # Filter DNS packets
            dns_packets = [p for p in packets if self.scapy.DNS in p]
            
            if dns_packets:
                return dns_packets[0]
            return None
        
        except Exception as e:
            logger.error(f"Error receiving DNS packet: {e}")
            return None
    
    def extract_data(self, packet: Any, method: str) -> List[int]:
        """
        Extract steganographic data from a DNS packet.
        
        Args:
            packet: Received packet
            method: Steganography method used
            
        Returns:
            list: Extracted data bits
        """
        if not self.scapy_available:
            logger.error("Scapy is required for DNS steganography")
            return []
        
        try:
            # Extract data using Scapy
            if method == 'domain':
                bits = []
                
                # Extract domain from query
                if self.scapy.DNS in packet and packet[self.scapy.DNS].qd:
                    domain = packet[self.scapy.DNS].qd.qname.decode('utf-8')
                    
                    # Extract hex subdomain
                    parts = domain.split('.')
                    if len(parts) > 1:
                        hex_str = parts[0]
                        
                        # Convert hex to bits
                        for char in hex_str:
                            try:
                                value = int(char, 16)
                                for i in range(4):
                                    bits.append((value >> i) & 1)
                            except ValueError:
                                pass
                
                return bits
            
            elif method == 'txid':
                bits = []
                
                # Extract txid
                if self.scapy.DNS in packet:
                    txid = packet[self.scapy.DNS].id
                    
                    # Convert txid to bits
                    for i in range(16):
                        bits.append((txid >> i) & 1)
                
                return bits
            
            else:
                return []
        
        except Exception as e:
            logger.error(f"Error extracting data from DNS packet: {e}")
            return []


def get_handler(protocol: str, interface: str = 'eth0', port: int = 8000, verbose: bool = False) -> ProtocolHandler:
    """
    Get an appropriate protocol handler.
    
    Args:
        protocol: Protocol name ('tcp', 'udp', 'icmp', 'dns')
        interface: Network interface to use
        port: Network port for applicable protocols
        verbose: Whether to print verbose output
        
    Returns:
        ProtocolHandler: Appropriate protocol handler
        
    Raises:
        ValueError: If protocol is not supported
    """
    protocol = protocol.lower()
    
    if protocol == 'tcp':
        return TCPHandler(interface, port, verbose)
    elif protocol == 'udp':
        return UDPHandler(interface, port, verbose)
    elif protocol == 'icmp':
        return ICMPHandler(interface, verbose)
    elif protocol == 'dns':
        return DNSHandler(interface, port if port != 8000 else 53, verbose)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


# Protocol-specific constants
MAX_PACKET_SIZE = {
    'tcp': 1460,  # TCP MSS (typical value)
    'udp': 508,   # UDP practical limit without fragmentation
    'icmp': 1472, # ICMP max data without fragmentation
    'dns': 512,   # DNS traditional limit
}

# Available steganography methods per protocol
PROTOCOL_METHODS = {
    'tcp': ['header', 'timing', 'size', 'sequence'],
    'udp': ['header', 'timing', 'size', 'sequence'],
    'icmp': ['header', 'timing', 'size'],
    'dns': ['domain', 'txid', 'sequence'],
}

# Capacity estimation (bits per packet)
CAPACITY_BITS = {
    'tcp': {
        'header': 32,     # Window size (16) + sequence (16)
        'timing': 1,      # One bit per timing interval
        'size': 8,        # Packet size variations
        'sequence': 16,   # Source port variations
    },
    'udp': {
        'header': 16,     # Source port (16 bits)
        'timing': 1,      # One bit per timing interval
        'size': 8,        # Packet size variations
        'sequence': 16,   # Source port variations
    },
    'icmp': {
        'header': 24,     # ID (8) + sequence (8) + type/code (8)
        'timing': 1,      # One bit per timing interval
        'size': 8,        # Packet size variations
    },
    'dns': {
        'domain': 40,     # ~10 hex chars in subdomain
        'txid': 16,       # Transaction ID (16 bits)
        'sequence': 16,   # Query sequence variations
    },
}