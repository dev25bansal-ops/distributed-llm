"""TLS credential management for gRPC secure channels."""

import os
import datetime
import ipaddress
import grpc
from loguru import logger


def generate_self_signed_certs(cert_dir: str = "certs", extra_hosts: list[str] | None = None) -> tuple[str, str, str]:
    """Generate self-signed TLS certificates for development.

    Security: Includes proper SANs for common deployment scenarios.
    WARNING: Self-signed certs should only be used for development.

    Uses the 'cryptography' library for cross-platform compatibility (Windows, Linux, macOS).
    Falls back to subprocess openssl if cryptography is not available.
    """
    os.makedirs(cert_dir, exist_ok=True)

    cert_file = os.path.join(cert_dir, "server.crt")
    key_file = os.path.join(cert_dir, "server.key")
    ca_cert_file = os.path.join(cert_dir, "ca.crt")

    # Security: Strong SAN configuration
    san_entries = [
        "DNS:localhost",
        "IP:127.0.0.1",
        "IP:::1",
        "DNS:*.local",
        "DNS:distllm",
        "DNS:distllm.local",
    ]

    if extra_hosts:
        for host in extra_hosts:
            if host.startswith("http://") or host.startswith("https://"):
                host = host.split("://")[1].split(":")[0]
            if ":" in host:
                san_entries.append(f"IP:{host}")
            else:
                san_entries.append(f"DNS:{host}")

    # Try cryptography library first (cross-platform, no subprocess needed)
    try:
        return _generate_certs_cryptography(
            cert_file, key_file, ca_cert_file, san_entries
        )
    except ImportError:
        logger.warning(
            "'cryptography' library not available, falling back to openssl subprocess. "
            "Install with: pip install cryptography"
        )

    # Fallback: subprocess openssl
    return _generate_certs_openssl(cert_file, key_file, ca_cert_file, san_entries)


def _generate_certs_cryptography(
    cert_file: str, key_file: str, ca_cert_file: str, san_entries: list[str]
) -> tuple[str, str, str]:
    """Generate self-signed certs using the cryptography library."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )

    # Build subject and issuer
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "distributed-llm"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    # Build SANs
    dns_names = []
    ip_addresses = []
    for entry in san_entries:
        if entry.startswith("DNS:"):
            dns_names.append(entry[4:])
        elif entry.startswith("IP:"):
            addr = entry[3:]
            try:
                ip_addresses.append(ipaddress.ip_address(addr))
            except ValueError:
                pass

    san_extension = x509.SubjectAlternativeName(
        [x509.DNSName(d) for d in dns_names]
        + [x509.IPAddress(ip) for ip in ip_addresses]
    )

    # Build certificate
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(san_extension, critical=False)
        .sign(private_key, hashes.SHA256())
    )

    # Write private key
    with open(key_file, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Write certificate
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # CA cert is the same as self-signed cert
    with open(ca_cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"Generated self-signed certificates using cryptography library with SANs: {san_entries}")
    return cert_file, key_file, ca_cert_file


def _generate_certs_openssl(
    cert_file: str, key_file: str, ca_cert_file: str, san_entries: list[str]
) -> tuple[str, str, str]:
    """Generate self-signed certs using openssl subprocess (fallback)."""
    import subprocess
    import shutil

    san_value = ",".join(san_entries)

    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", key_file, "-out", cert_file,
        "-days", "365", "-nodes",
        "-subj", "/CN=distributed-llm/O=distributed-llm/CN=localhost",
        "-addext", f"subjectAltName={san_value}",
    ], check=True, capture_output=True)

    shutil.copy2(cert_file, ca_cert_file)
    logger.info(f"Generated self-signed certificates using openssl with SANs: {san_entries}")
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
