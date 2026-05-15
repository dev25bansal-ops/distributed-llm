"""TLS credential management for gRPC secure channels."""

import os
import grpc
from loguru import logger
from typing import List, Tuple, Optional


def generate_self_signed_certs(cert_dir: str = "certs", extra_hosts: Optional[List[str]] = None) -> Tuple[str, str, str]:
    """Generate self-signed TLS certificates for development.

    Security: Includes proper SANs for common deployment scenarios.
    WARNING: Self-signed certs should only be used for development.
    """
    import subprocess
    import shutil

    os.makedirs(cert_dir, exist_ok=True)

    cert_file = os.path.join(cert_dir, "server.crt")
    key_file = os.path.join(cert_dir, "server.key")
    ca_cert_file = os.path.join(cert_dir, "ca.crt")

    # Security: Strong SAN configuration
    san_entries = [
        "DNS:localhost",
        "IP:127.0.0.1",
        "IP:::1",
        "DNS:*.local",  # Internal DNS
        "DNS:distllm",
        "DNS:distllm.local",
    ]

    # Add custom hosts from configuration
    if extra_hosts:
        for host in extra_hosts:
            if host.startswith("http://") or host.startswith("https://"):
                host = host.split("://")[1].split(":")[0]
            if ":" in host:  # IP address
                san_entries.append(f"IP:{host}")
            else:
                san_entries.append(f"DNS:{host}")

    san_value = ",".join(san_entries)

    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", key_file, "-out", cert_file,
        "-days", "365", "-nodes",
        "-subj", "/CN=distributed-llm/O=distributed-llm/CN=localhost",
        "-addext", f"subjectAltName={san_value}",
    ], check=True, capture_output=True)

    shutil.copy2(cert_file, ca_cert_file)
    logger.info(f"Generated self-signed certificates in {cert_dir} with SANs: {san_entries}")
    return cert_file, key_file, ca_cert_file


def load_tls_credentials(cert_file: str, key_file: str) -> grpc.ServerCredentials:
    """Load TLS server credentials from certificate files."""
    with open(cert_file, "rb") as f:
        cert_chain = f.read()
    with open(key_file, "rb") as f:
        private_key = f.read()
    return grpc.ssl_server_credentials([(private_key, cert_chain)])


def load_tls_channel_credentials(ca_cert_file: str, server_hostname: str = "localhost") -> grpc.ChannelCredentials:
    """Load TLS channel credentials for client connections."""
    with open(ca_cert_file, "rb") as f:
        ca_cert = f.read()
    return grpc.ssl_channel_credentials(root_certificates=ca_cert)


def create_secure_server(port: int, servicer, cert_file: str, key_file: str, max_workers: int = 10):
    """Create a TLS-secured gRPC server."""
    from concurrent import futures
    from distllm.communication.node_pb2_grpc import (
        add_NodeServiceServicer_to_server,
        add_CoordinatorServiceServicer_to_server,
    )

    credentials = load_tls_credentials(cert_file, key_file)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    if hasattr(servicer, 'ForwardPass'):
        add_NodeServiceServicer_to_server(servicer, server)
    else:
        add_CoordinatorServiceServicer_to_server(servicer, server)

    server.add_secure_port(f"[::]:{port}", credentials)
    return server


def create_secure_channel(host: str, port: int, ca_cert_file: str) -> grpc.SecureChannel:
    """Create a TLS-secured gRPC channel."""
    credentials = load_tls_channel_credentials(ca_cert_file, host)
    return grpc.secure_channel(f"{host}:{port}", credentials)
