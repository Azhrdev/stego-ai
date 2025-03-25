# -*- coding: utf-8 -*-
"""
Network utilities for Stego-AI.

This module provides functions for network steganography operations,
including packet manipulation, header analysis, and network protocols.
"""

import os
import re
import time
import random
import socket
import struct
import logging
import ipaddress
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Set up logging
logger = logging.getLogger(__name__)

# Try to import specialized libraries
try:
    import scapy.all as scapy
    has_scapy = True
except ImportError:
    logger.warning("Scapy not available. Some network features will be limited.")
    has_scapy = False


def check_dependencies() -> Dict[str, bool]:
    """
    Check for required and optional dependencies.
    
    Returns:
        dict: Available dependencies
    """
    dependencies = {
        'scapy': has_scapy,
        'socket': True,  # Always available in standard library
    }
    
    return dependencies


def resolve_hostname(hostname: str) -> str:
    """
    Resolve hostname to IP address.
    
    Args:
        hostname: Hostname to resolve
        
    Returns:
        str: IP address
    """
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname {hostname}: {e}")


def is_local_ip(ip: str) -> bool:
    """
    Check if IP address is local.
    
    Args:
        ip: IP address to check
        
    Returns:
        bool: True if IP is local
    """
    try:
        # Convert to ipaddress object
        ip_obj = ipaddress.ip_address(ip)
        
        # Check if private
        if ip_obj.is_private:
            return True
        
        # Check if loopback
        if ip_obj.is_loopback:
            return True
        
        # Check specific ranges
        if ip_obj.is_link_local:
            return True
        
        return False
    
    except ValueError:
        return False


def get_local_ip() -> str:
    """
    Get local IP address.
    
    Returns:
        str: Local IP address
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to arbitrary address to get local IP
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # Fallback
        return socket.gethostbyname(socket.gethostname())


def get_network_interfaces() -> List[Dict[str, Any]]:
    """
    Get list of network interfaces.
    
    Returns:
        list: Network interfaces with name, address, netmask
    """
    if has_scapy:
        # Use Scapy to get interfaces
        interfaces = []
        for iface in scapy.get_if_list():
            try:
                ip = scapy.get_if_addr(iface)
                if ip:
                    interfaces.append({
                        'name': iface,
                        'address': ip,
                        'mac': scapy.get_if_hwaddr(iface),
                    })
            except Exception:
                pass
        return interfaces
    else:
        # Fallback to socket
        # This is a simplified approach, not as detailed as Scapy
        import netifaces
        
        interfaces = []
        for iface in netifaces.interfaces():
            try:
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        interfaces.append({
                            'name': iface,
                            'address': addr['addr'],
                            'netmask': addr.get('netmask', ''),
                        })
                # Get MAC address if available
                if netifaces.AF_LINK in addrs:
                    for link in addrs[netifaces.AF_LINK]:
                        if 'addr' in link:
                            interfaces[-1]['mac'] = link['addr']
            except Exception:
                pass
        
        return interfaces


def ping(host: str, count: int = 4, timeout: float = 1.0) -> Dict[str, Any]:
    """
    Ping a host.
    
    Args:
        host: Hostname or IP address
        count: Number of pings
        timeout: Timeout in seconds
        
    Returns:
        dict: Ping results (min/avg/max RTT, packets sent/received)
    """
    if has_scapy:
        # Use Scapy for more control
        try:
            # Resolve hostname
            ip = resolve_hostname(host)
            
            # Prepare packets
            packets = []
            for i in range(count):
                # Create ICMP Echo Request
                pkt = scapy.IP(dst=ip) / scapy.ICMP(type=8, code=0, seq=i, id=os.getpid() & 0xFFFF)
                packets.append(pkt)
            
            # Send packets and measure RTT
            rtts = []
            sent = 0
            received = 0
            
            for pkt in packets:
                sent += 1
                start_time = time.time()
                
                # Send packet and wait for reply
                reply = scapy.sr1(pkt, timeout=timeout, verbose=0)
                
                if reply:
                    received += 1
                    rtt = (time.time() - start_time) * 1000  # Convert to ms
                    rtts.append(rtt)
                
                # Delay between pings
                time.sleep(0.2)
            
            # Calculate statistics
            if rtts:
                min_rtt = min(rtts)
                avg_rtt = sum(rtts) / len(rtts)
                max_rtt = max(rtts)
            else:
                min_rtt = avg_rtt = max_rtt = 0
            
            return {
                'host': host,
                'ip': ip,
                'sent': sent,
                'received': received,
                'loss': (sent - received) / sent * 100 if sent > 0 else 0,
                'min_rtt': min_rtt,
                'avg_rtt': avg_rtt,
                'max_rtt': max_rtt,
            }
            
        except Exception as e:
            logger.error(f"Error pinging host {host}: {e}")
            return {
                'host': host,
                'error': str(e),
                'sent': 0,
                'received': 0,
                'loss': 100,
                'min_rtt': 0,
                'avg_rtt': 0,
                'max_rtt': 0,
            }
    else:
        # Fallback to system ping
        import subprocess
        import platform
        
        try:
            # Determine ping command based on platform
            if platform.system().lower() == 'windows':
                cmd = ['ping', '-n', str(count), '-w', str(int(timeout * 1000)), host]
            else:
                cmd = ['ping', '-c', str(count), '-W', str(int(timeout)), host]
            
            # Run ping command
            output = subprocess.check_output(cmd, universal_newlines=True)
            
            # Parse output
            if platform.system().lower() == 'windows':
                # Parse Windows ping output
                match = re.search(r'Sent = (\d+), Received = (\d+), Lost = (\d+)', output)
                if match:
                    sent, received, lost = map(int, match.groups())
                else:
                    sent, received, lost = 0, 0, 0
                
                match = re.search(r'Minimum = (\d+)ms, Maximum = (\d+)ms, Average = (\d+)ms', output)
                if match:
                    min_rtt, max_rtt, avg_rtt = map(float, match.groups())
                else:
                    min_rtt = avg_rtt = max_rtt = 0
            else:
                # Parse Unix ping output
                match = re.search(r'(\d+) packets transmitted, (\d+) (packets )?received', output)
                if match:
                    sent, received = map(int, match.groups()[:2])
                    lost = sent - received
                else:
                    sent, received, lost = 0, 0, 0
                
                match = re.search(r'min/avg/max/(mdev)?.*?= ([\d.]+)/([\d.]+)/([\d.]+)', output)
                if match:
                    min_rtt, avg_rtt, max_rtt = map(float, match.groups()[1:])
                else:
                    min_rtt = avg_rtt = max_rtt = 0
            
            # Extract IP
            match = re.search(r'PING.*?(\d+\.\d+\.\d+\.\d+)', output)
            ip = match.group(1) if match else host
            
            return {
                'host': host,
                'ip': ip,
                'sent': sent,
                'received': received,
                'loss': (sent - received) / sent * 100 if sent > 0 else 0,
                'min_rtt': min_rtt,
                'avg_rtt': avg_rtt,
                'max_rtt': max_rtt,
            }
            
        except Exception as e:
            logger.error(f"Error pinging host {host}: {e}")
            return {
                'host': host,
                'error': str(e),
                'sent': 0,
                'received': 0,
                'loss': 100,
                'min_rtt': 0,
                'avg_rtt': 0,
                'max_rtt': 0,
            }


def traceroute(host: str, max_hops: int = 30, timeout: float = 1.0) -> List[Dict[str, Any]]:
    """
    Trace route to host.
    
    Args:
        host: Hostname or IP address
        max_hops: Maximum number of hops
        timeout: Timeout in seconds
        
    Returns:
        list: Traceroute results (hop, IP, hostname, RTT)
    """
    if has_scapy:
        # Use Scapy for traceroute
        try:
            # Resolve hostname
            ip = resolve_hostname(host)
            
            # Create traceroute
            results, _ = scapy.traceroute(ip, maxttl=max_hops, timeout=timeout, verbose=0)
            
            # Parse results
            hops = []
            for i in range(1, max_hops + 1):
                if i in results:
                    hop = results[i][0]
                    hop_ip = hop.src
                    try:
                        hop_host = socket.gethostbyaddr(hop_ip)[0]
                    except socket.herror:
                        hop_host = hop_ip
                    
                    # Calculate RTT
                    rtt = results[i][1].time * 1000 if hasattr(results[i][1], 'time') else 0
                    
                    hops.append({
                        'hop': i,
                        'ip': hop_ip,
                        'host': hop_host,
                        'rtt': rtt,
                    })
                else:
                    # No response for this hop
                    hops.append({
                        'hop': i,
                        'ip': None,
                        'host': None,
                        'rtt': None,
                    })
                    
                # Stop if we reached the destination
                if hop_ip == ip:
                    break
            
            return hops
            
        except Exception as e:
            logger.error(f"Error tracing route to {host}: {e}")
            return [{'hop': 1, 'ip': None, 'host': None, 'rtt': None, 'error': str(e)}]
    else:
        # Fallback to system traceroute
        import subprocess
        import platform
        
        try:
            # Determine traceroute command based on platform
            if platform.system().lower() == 'windows':
                cmd = ['tracert', '-d', '-h', str(max_hops), '-w', str(int(timeout * 1000)), host]
            else:
                cmd = ['traceroute', '-n', '-m', str(max_hops), '-w', str(int(timeout)), host]
            
            # Run traceroute command
            output = subprocess.check_output(cmd, universal_newlines=True)
            
            # Parse output (simplified)
            hops = []
            for line in output.splitlines():
                # Skip header lines
                if not re.match(r'^\s*\d+', line):
                    continue
                
                # Parse hop data
                parts = line.strip().split()
                hop_num = int(parts[0])
                
                # Extract IP (may have multiple entries per hop)
                ip_matches = re.findall(r'\d+\.\d+\.\d+\.\d+', line)
                hop_ip = ip_matches[0] if ip_matches else None
                
                # Extract RTT (simplified)
                rtt_matches = re.findall(r'(\d+) ms', line)
                hop_rtt = float(rtt_matches[0]) if rtt_matches else None
                
                hops.append({
                    'hop': hop_num,
                    'ip': hop_ip,
                    'host': hop_ip,  # Using IP as hostname for simplicity
                    'rtt': hop_rtt,
                })
            
            return hops
            
        except Exception as e:
            logger.error(f"Error tracing route to {host}: {e}")
            return [{'hop': 1, 'ip': None, 'host': None, 'rtt': None, 'error': str(e)}]


def send_tcp_packet(target: str, port: int, data: Optional[bytes] = None,
                   flags: str = 'S', ttl: int = 64, seq: int = None,
                   source_port: int = None) -> Dict[str, Any]:
    """
    Send a TCP packet with custom parameters.
    
    Args:
        target: Target hostname or IP
        port: Target port
        data: Packet payload
        flags: TCP flags ('S' for SYN, 'A' for ACK, etc.)
        ttl: Time to live
        seq: Sequence number (random if None)
        source_port: Source port (random if None)
        
    Returns:
        dict: Result including response if any
    """
    if not has_scapy:
        raise ImportError("Scapy is required for packet manipulation")
    
    try:
        # Resolve hostname
        target_ip = resolve_hostname(target)
        
        # Generate random sequence number if not provided
        if seq is None:
            seq = random.randint(1000, 1000000)
        
        # Generate random source port if not provided
        if source_port is None:
            source_port = random.randint(10000, 60000)
        
        # Parse TCP flags
        tcp_flags = 0
        flag_map = {'F': 0x01, 'S': 0x02, 'R': 0x04, 'P': 0x08, 'A': 0x10, 'U': 0x20}
        for flag in flags:
            tcp_flags |= flag_map.get(flag.upper(), 0)
        
        # Create packet
        ip = scapy.IP(dst=target_ip, ttl=ttl)
        tcp = scapy.TCP(sport=source_port, dport=port, flags=tcp_flags, seq=seq)
        
        # Add payload if provided
        if data:
            packet = ip / tcp / data
        else:
            packet = ip / tcp
        
        # Send packet and get response
        response = scapy.sr1(packet, timeout=2, verbose=0)
        
        result = {
            'success': True,
            'target': target,
            'port': port,
            'flags': flags,
            'seq': seq,
            'source_port': source_port,
            'response': None,
        }
        
        # Parse response if any
        if response:
            result['response'] = {
                'time': time.time(),
                'src': response[scapy.IP].src,
                'ttl': response[scapy.IP].ttl,
                'proto': response[scapy.IP].proto,
            }
            
            if scapy.TCP in response:
                result['response']['tcp'] = {
                    'sport': response[scapy.TCP].sport,
                    'dport': response[scapy.TCP].dport,
                    'flags': response[scapy.TCP].flags,
                    'seq': response[scapy.TCP].seq,
                    'ack': response[scapy.TCP].ack,
                }
        
        return result
    
    except Exception as e:
        logger.error(f"Error sending TCP packet to {target}:{port}: {e}")
        return {
            'success': False,
            'target': target,
            'port': port,
            'error': str(e),
        }


def send_udp_packet(target: str, port: int, data: Optional[bytes] = None,
                  ttl: int = 64, source_port: int = None) -> Dict[str, Any]:
    """
    Send a UDP packet with custom parameters.
    
    Args:
        target: Target hostname or IP
        port: Target port
        data: Packet payload
        ttl: Time to live
        source_port: Source port (random if None)
        
    Returns:
        dict: Result including response if any
    """
    if not has_scapy:
        # Fallback to socket (limited features)
        return send_udp_socket(target, port, data, source_port)
    
    try:
        # Resolve hostname
        target_ip = resolve_hostname(target)
        
        # Generate random source port if not provided
        if source_port is None:
            source_port = random.randint(10000, 60000)
        
        # Create packet
        ip = scapy.IP(dst=target_ip, ttl=ttl)
        udp = scapy.UDP(sport=source_port, dport=port)
        
        # Add payload if provided
        if data:
            packet = ip / udp / data
        else:
            packet = ip / udp
        
        # Send packet and get response
        response = scapy.sr1(packet, timeout=2, verbose=0)
        
        result = {
            'success': True,
            'target': target,
            'port': port,
            'source_port': source_port,
            'response': None,
        }
        
        # Parse response if any
        if response:
            result['response'] = {
                'time': time.time(),
                'src': response[scapy.IP].src,
                'ttl': response[scapy.IP].ttl,
                'proto': response[scapy.IP].proto,
            }
            
            if scapy.ICMP in response:
                result['response']['icmp'] = {
                    'type': response[scapy.ICMP].type,
                    'code': response[scapy.ICMP].code,
                }
        
        return result
    
    except Exception as e:
        logger.error(f"Error sending UDP packet to {target}:{port}: {e}")
        return {
            'success': False,
            'target': target,
            'port': port,
            'error': str(e),
        }


def send_udp_socket(target: str, port: int, data: Optional[bytes] = None,
                   source_port: int = None) -> Dict[str, Any]:
    """
    Send a UDP packet using socket API (fallback).
    
    Args:
        target: Target hostname or IP
        port: Target port
        data: Packet payload
        source_port: Source port (ignored in socket API)
        
    Returns:
        dict: Result
    """
    try:
        # Resolve hostname
        target_ip = resolve_hostname(target)
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Set timeout
        sock.settimeout(2)
        
        # Prepare data
        if data is None:
            data = b'\x00' * 8
        
        # Send data
        sock.sendto(data, (target_ip, port))
        
        result = {
            'success': True,
            'target': target,
            'port': port,
            'response': None,
        }
        
        # Try to receive response
        try:
            response, addr = sock.recvfrom(1024)
            result['response'] = {
                'time': time.time(),
                'src': addr[0],
                'port': addr[1],
                'data': response,
            }
        except socket.timeout:
            # No response (normal for UDP)
            pass
        finally:
            sock.close()
        
        return result
    
    except Exception as e:
        logger.error(f"Error sending UDP data to {target}:{port}: {e}")
        return {
            'success': False,
            'target': target,
            'port': port,
            'error': str(e),
        }


def packet_sniffer(interface: Optional[str] = None, 
                  filter_str: Optional[str] = None,
                  count: Optional[int] = None,
                  timeout: Optional[float] = None,
                  callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Capture network packets.
    
    Args:
        interface: Network interface
        filter_str: BPF filter string
        count: Number of packets to capture
        timeout: Capture timeout in seconds
        callback: Callback function for each packet
        
    Returns:
        list: Captured packets
    """
    if not has_scapy:
        raise ImportError("Scapy is required for packet sniffing")
    
    captured_packets = []
    
    def packet_handler(packet):
        """Process each captured packet."""
        packet_dict = packet_to_dict(packet)
        
        # Add to captured packets
        captured_packets.append(packet_dict)
        
        # Call callback if provided
        if callback:
            callback(packet_dict)
        
        # Stop if we have enough packets
        if count and len(captured_packets) >= count:
            return True
    
    try:
        # Start packet capture
        scapy.sniff(
            iface=interface,
            filter=filter_str,
            prn=packet_handler,
            count=count,
            timeout=timeout,
            store=0,  # Don't store packets in memory
        )
        
        return captured_packets
    
    except Exception as e:
        logger.error(f"Error in packet capture: {e}")
        return []


def packet_to_dict(packet) -> Dict[str, Any]:
    """
    Convert Scapy packet to dictionary.
    
    Args:
        packet: Scapy packet
        
    Returns:
        dict: Packet data
    """
    packet_dict = {
        'time': time.time(),
        'summary': packet.summary(),
        'layers': {},
    }
    
    # Process each layer
    if scapy.Ether in packet:
        packet_dict['layers']['ether'] = {
            'src': packet[scapy.Ether].src,
            'dst': packet[scapy.Ether].dst,
            'type': packet[scapy.Ether].type,
        }
    
    if scapy.IP in packet:
        packet_dict['layers']['ip'] = {
            'version': packet[scapy.IP].version,
            'ihl': packet[scapy.IP].ihl,
            'tos': packet[scapy.IP].tos,
            'len': packet[scapy.IP].len,
            'id': packet[scapy.IP].id,
            'flags': packet[scapy.IP].flags,
            'frag': packet[scapy.IP].frag,
            'ttl': packet[scapy.IP].ttl,
            'proto': packet[scapy.IP].proto,
            'chksum': packet[scapy.IP].chksum,
            'src': packet[scapy.IP].src,
            'dst': packet[scapy.IP].dst,
            'options': [],  # IP options would be extracted here
        }
    
    if scapy.IPv6 in packet:
        packet_dict['layers']['ipv6'] = {
            'version': packet[scapy.IPv6].version,
            'tc': packet[scapy.IPv6].tc,
            'fl': packet[scapy.IPv6].fl,
            'plen': packet[scapy.IPv6].plen,
            'nh': packet[scapy.IPv6].nh,
            'hlim': packet[scapy.IPv6].hlim,
            'src': packet[scapy.IPv6].src,
            'dst': packet[scapy.IPv6].dst,
        }
    
    if scapy.TCP in packet:
        packet_dict['layers']['tcp'] = {
            'sport': packet[scapy.TCP].sport,
            'dport': packet[scapy.TCP].dport,
            'seq': packet[scapy.TCP].seq,
            'ack': packet[scapy.TCP].ack,
            'dataofs': packet[scapy.TCP].dataofs,
            'reserved': packet[scapy.TCP].reserved,
            'flags': packet[scapy.TCP].flags,
            'window': packet[scapy.TCP].window,
            'chksum': packet[scapy.TCP].chksum,
            'urgptr': packet[scapy.TCP].urgptr,
            'options': [],  # TCP options would be extracted here
        }
    
    if scapy.UDP in packet:
        packet_dict['layers']['udp'] = {
            'sport': packet[scapy.UDP].sport,
            'dport': packet[scapy.UDP].dport,
            'len': packet[scapy.UDP].len,
            'chksum': packet[scapy.UDP].chksum,
        }
    
    if scapy.ICMP in packet:
        packet_dict['layers']['icmp'] = {
            'type': packet[scapy.ICMP].type,
            'code': packet[scapy.ICMP].code,
            'chksum': packet[scapy.ICMP].chksum,
            'id': getattr(packet[scapy.ICMP], 'id', None),
            'seq': getattr(packet[scapy.ICMP], 'seq', None),
        }
    
    if scapy.DNS in packet:
        packet_dict['layers']['dns'] = {
            'id': packet[scapy.DNS].id,
            'qr': packet[scapy.DNS].qr,
            'opcode': packet[scapy.DNS].opcode,
            'aa': packet[scapy.DNS].aa,
            'tc': packet[scapy.DNS].tc,
            'rd': packet[scapy.DNS].rd,
            'ra': packet[scapy.DNS].ra,
            'z': packet[scapy.DNS].z,
            'rcode': packet[scapy.DNS].rcode,
            'qdcount': packet[scapy.DNS].qdcount,
            'ancount': packet[scapy.DNS].ancount,
            'nscount': packet[scapy.DNS].nscount,
            'arcount': packet[scapy.DNS].arcount,
            'qd': [],  # DNS queries would be extracted here
            'an': [],  # DNS answers would be extracted here
        }
    
    # Extract payload if available
    if hasattr(packet, 'load'):
        packet_dict['payload'] = {
            'hex': packet.load.hex(),
            'len': len(packet.load),
        }
    
    return packet_dict


def create_tcp_header(source_port: int, dest_port: int, seq_num: int, ack_num: int,
                     flags: int = 0, window: int = 8192) -> bytes:
    """
    Create TCP header.
    
    Args:
        source_port: Source port
        dest_port: Destination port
        seq_num: Sequence number
        ack_num: Acknowledgement number
        flags: TCP flags (URG, ACK, PSH, RST, SYN, FIN)
        window: Window size
        
    Returns:
        bytes: TCP header
    """
    # Set data offset (5 words, no options)
    offset = 5 << 4
    
    # Combine offset and flags
    offset_flags = offset | flags
    
    # Pack TCP header (without checksum)
    tcp_header = struct.pack(
        '!HHIIBBHHH',
        source_port,          # Source port
        dest_port,            # Destination port
        seq_num,              # Sequence number
        ack_num,              # Acknowledgement number
        offset_flags,         # Data offset and flags
        window,               # Window size
        0,                    # Checksum (0 for now)
        0,                    # Urgent pointer
    )
    
    return tcp_header


def create_udp_header(source_port: int, dest_port: int, length: int) -> bytes:
    """
    Create UDP header.
    
    Args:
        source_port: Source port
        dest_port: Destination port
        length: Length of UDP header + data
        
    Returns:
        bytes: UDP header
    """
    # Pack UDP header (without checksum)
    udp_header = struct.pack(
        '!HHHH',
        source_port,          # Source port
        dest_port,            # Destination port
        length,               # Length
        0,                    # Checksum (0 for now)
    )
    
    return udp_header


def create_ip_header(source_ip: str, dest_ip: str, protocol: int,
                   id: int = None, ttl: int = 64, df: bool = True) -> bytes:
    """
    Create IP header.
    
    Args:
        source_ip: Source IP address
        dest_ip: Destination IP address
        protocol: Protocol number (6 for TCP, 17 for UDP)
        id: IP ID (random if None)
        ttl: Time to live
        df: Don't Fragment flag
        
    Returns:
        bytes: IP header
    """
    # Generate random ID if not provided
    if id is None:
        id = random.randint(0, 65535)
    
    # Set IP version (4) and header length (5 words, no options)
    version_ihl = (4 << 4) | 5
    
    # Set flags and fragment offset
    flags_offset = (2 if df else 0) << 13
    
    # Convert IP addresses to integers
    source_addr = int(ipaddress.IPv4Address(source_ip))
    dest_addr = int(ipaddress.IPv4Address(dest_ip))
    
    # Pack IP header (without checksum)
    ip_header = struct.pack(
        '!BBHHHBBHII',
        version_ihl,          # Version and header length
        0,                    # Type of service
        0,                    # Total length (will be filled later)
        id,                   # Identification
        flags_offset,         # Flags and fragment offset
        ttl,                  # Time to live
        protocol,             # Protocol
        0,                    # Header checksum (0 for now)
        source_addr,          # Source address
        dest_addr,            # Destination address
    )
    
    return ip_header


def calculate_checksum(data: bytes) -> int:
    """
    Calculate IP/TCP/UDP checksum.
    
    Args:
        data: Data to checksum
        
    Returns:
        int: Checksum value
    """
    # Ensure data length is even
    if len(data) % 2:
        data += b'\x00'
    
    # Sum 16-bit words
    checksum = 0
    for i in range(0, len(data), 2):
        word = (data[i] << 8) + data[i + 1]
        checksum += word
    
    # Add any carry
    checksum = (checksum >> 16) + (checksum & 0xFFFF)
    checksum += checksum >> 16
    
    # Take one's complement
    return ~checksum & 0xFFFF


def hide_data_in_packet(packet, data: bytes, method: str = 'header') -> bytes:
    """
    Hide data in network packet.
    
    Args:
        packet: Scapy packet
        data: Data to hide
        method: Hiding method ('header', 'id', 'seq', 'options')
        
    Returns:
        bytes: Modified packet
    """
    if not has_scapy:
        raise ImportError("Scapy is required for packet manipulation")
    
    try:
        # Convert data to bits
        data_bits = []
        for byte in data:
            for i in range(7, -1, -1):
                data_bits.append((byte >> i) & 1)
        
        # Apply hiding method
        if method == 'header':
            # Hide in IP ID and fragmentation fields
            if scapy.IP in packet:
                # Use 16 bits of IP ID
                packet[scapy.IP].id = int(''.join(map(str, data_bits[:16])), 2) if len(data_bits) >= 16 else 0
                
                # Use 3 bits of IP flags (careful, only 1 bit is really usable)
                # Usually only the "Don't Fragment" flag can be safely modified
                if len(data_bits) > 16:
                    df_bit = data_bits[16] if len(data_bits) > 16 else 1  # Default to Don't Fragment
                    packet[scapy.IP].flags = df_bit * 2  # DF is bit 1
                
                # Update length field
                # (handled automatically by Scapy)
                
                # Clear checksum for recalculation
                packet[scapy.IP].chksum = None
        
        elif method == 'id':
            # Hide in IP ID field
            if scapy.IP in packet:
                # Use 16 bits of IP ID
                value = 0
                for i, bit in enumerate(data_bits[:16]):
                    value |= bit << (15 - i)
                packet[scapy.IP].id = value
                
                # Clear checksum for recalculation
                packet[scapy.IP].chksum = None
        
        elif method == 'seq':
            # Hide in TCP sequence number
            if scapy.TCP in packet:
                # Use 32 bits of TCP sequence number
                value = 0
                for i, bit in enumerate(data_bits[:32]):
                    value |= bit << (31 - i)
                packet[scapy.TCP].seq = value
                
                # Clear checksum for recalculation
                packet[scapy.TCP].chksum = None
        
        elif method == 'options':
            # Hide in TCP options
            if scapy.TCP in packet:
                # Create a custom option (kind 253, experimental)
                option_data = bytearray()
                for i in range(0, len(data_bits), 8):
                    byte = 0
                    for j in range(min(8, len(data_bits) - i)):
                        byte |= data_bits[i + j] << (7 - j)
                    option_data.append(byte)
                
                # Add option to packet
                packet[scapy.TCP].options = [(253, bytes(option_data))]
                
                # Clear checksum for recalculation
                packet[scapy.TCP].chksum = None
        
        else:
            raise ValueError(f"Unknown hiding method: {method}")
        
        # Convert packet to bytes
        return bytes(packet)
    
    except Exception as e:
        logger.error(f"Error hiding data in packet: {e}")
        raise


def extract_data_from_packet(packet, method: str = 'header', data_length: int = 16) -> bytes:
    """
    Extract hidden data from network packet.
    
    Args:
        packet: Scapy packet or raw bytes
        method: Hiding method ('header', 'id', 'seq', 'options')
        data_length: Length of hidden data in bits
        
    Returns:
        bytes: Extracted data
    """
    if not has_scapy:
        raise ImportError("Scapy is required for packet analysis")
    
    try:
        # Convert raw bytes to Scapy packet if needed
        if isinstance(packet, bytes):
            packet = scapy.Ether(packet)
        
        # Extract bits based on method
        extracted_bits = []
        
        if method == 'header':
            # Extract from IP ID and fragmentation fields
            if scapy.IP in packet:
                # Extract 16 bits from IP ID
                id_value = packet[scapy.IP].id
                for i in range(15, -1, -1):
                    extracted_bits.append((id_value >> i) & 1)
                
                # Extract 1 bit from DF flag
                df_bit = (packet[scapy.IP].flags & 2) >> 1
                extracted_bits.append(df_bit)
        
        elif method == 'id':
            # Extract from IP ID field
            if scapy.IP in packet:
                # Extract 16 bits from IP ID
                id_value = packet[scapy.IP].id
                for i in range(15, -1, -1):
                    extracted_bits.append((id_value >> i) & 1)
        
        elif method == 'seq':
            # Extract from TCP sequence number
            if scapy.TCP in packet:
                # Extract 32 bits from TCP sequence number
                seq_value = packet[scapy.TCP].seq
                for i in range(31, -1, -1):
                    extracted_bits.append((seq_value >> i) & 1)
        
        elif method == 'options':
            # Extract from TCP options
            if scapy.TCP in packet:
                # Look for custom option (kind 253)
                for option in packet[scapy.TCP].options:
                    if option[0] == 253:
                        option_data = option[1]
                        for byte in option_data:
                            for i in range(7, -1, -1):
                                extracted_bits.append((byte >> i) & 1)
        
        else:
            raise ValueError(f"Unknown hiding method: {method}")
        
        # Limit to requested data length
        extracted_bits = extracted_bits[:data_length]
        
        # Convert bits to bytes
        result = bytearray()
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                byte = 0
                for j in range(8):
                    byte |= extracted_bits[i + j] << (7 - j)
                result.append(byte)
        
        return bytes(result)
    
    except Exception as e:
        logger.error(f"Error extracting data from packet: {e}")
        return b''


def scan_port(target: str, port: int, timeout: float = 1.0) -> Dict[str, Any]:
    """
    Scan a single port.
    
    Args:
        target: Target hostname or IP
        port: Port to scan
        timeout: Timeout in seconds
        
    Returns:
        dict: Port scan result
    """
    try:
        # Resolve hostname
        target_ip = resolve_hostname(target)
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Attempt connection
        start_time = time.time()
        result = sock.connect_ex((target_ip, port))
        response_time = time.time() - start_time
        
        # Get service name if possible
        service = ''
        try:
            service = socket.getservbyport(port)
        except:
            pass
        
        # Close socket
        sock.close()
        
        return {
            'target': target,
            'ip': target_ip,
            'port': port,
            'state': 'open' if result == 0 else 'closed',
            'service': service,
            'response_time': response_time,
        }
    
    except Exception as e:
        logger.error(f"Error scanning {target}:{port}: {e}")
        return {
            'target': target,
            'port': port,
            'state': 'error',
            'error': str(e),
        }


def scan_ports(target: str, ports: List[int], timeout: float = 1.0,
             callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Scan multiple ports.
    
    Args:
        target: Target hostname or IP
        ports: List of ports to scan
        timeout: Timeout in seconds
        callback: Progress callback function
        
    Returns:
        list: Port scan results
    """
    results = []
    
    # Resolve hostname once
    try:
        target_ip = resolve_hostname(target)
    except Exception as e:
        logger.error(f"Error resolving {target}: {e}")
        return [{
            'target': target,
            'port': port,
            'state': 'error',
            'error': f"Could not resolve hostname: {e}",
        } for port in ports]
    
    # Scan each port
    for i, port in enumerate(ports):
        try:
            # Update progress if callback provided
            if callback:
                progress = (i + 1) / len(ports)
                callback(progress, i + 1, len(ports))
            
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Attempt connection
            start_time = time.time()
            result = sock.connect_ex((target_ip, port))
            response_time = time.time() - start_time
            
            # Get service name if possible
            service = ''
            try:
                service = socket.getservbyport(port)
            except:
                pass
            
            # Close socket
            sock.close()
            
            # Add result
            results.append({
                'target': target,
                'ip': target_ip,
                'port': port,
                'state': 'open' if result == 0 else 'closed',
                'service': service,
                'response_time': response_time,
            })
            
        except Exception as e:
            logger.error(f"Error scanning {target}:{port}: {e}")
            results.append({
                'target': target,
                'ip': target_ip,
                'port': port,
                'state': 'error',
                'error': str(e),
            })
    
    return results