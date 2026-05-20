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

    ca_key_file = os.path.join(cert_dir, "ca.key")
    ca_cert_file = os.path.join(cert_dir, "ca.crt")
    cert_file = os.path.join(cert_dir, "server.crt")
    key_file = os.path.join(cert_dir, "server.key")
    client_cert_file = os.path.join(cert_dir, "client.crt")
    client_key_file = os.path.join(cert_dir, "client.key")

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

    try:
        return _generate_certs_cryptography(
            ca_key_file, ca_cert_file, cert_file, key_file,
            client_cert_file, client_key_file, san_entries,
        )
    except ImportError:
        logger.warning(
            "'cryptography' library not available, falling back to openssl subprocess. "
            "Install with: pip install cryptography"
        )

    return _generate_certs_openssl(
        ca_key_file, ca_cert_file, cert_file, key_file,
        client_cert_file, client_key_file, san_entries,
    )


def _generate_certs_cryptography(
    ca_key_file: str, ca_cert_file: str,
    cert_file: str, key_file: str,
    client_cert_file: str, client_key_file: str,
    san_entries: list[str],
) -> tuple[str, str, str]:
    """Generate CA + server certificate chain using the cryptography library."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    # Generate CA key pair
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

    ca_subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "distributed-llm"),
        x509.NameAttribute(NameOID.COMMON_NAME, "distributed-llm CA"),
    ])

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365 * 10))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False, data_encipherment=False,
                key_agreement=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Generate server key pair
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

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

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "distributed-llm"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]))
        .issuer_name(ca_subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(san_extension, critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Generate client key pair for mutual TLS
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "distributed-llm"),
            x509.NameAttribute(NameOID.COMMON_NAME, "distributed-llm-client"),
        ]))
        .issuer_name(ca_subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Write CA
    with open(ca_key_file, "wb") as f:
        f.write(ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(ca_cert_file, "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    # Write server
    with open(key_file, "wb") as f:
        f.write(server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_file, "wb") as f:
        f.write(server_cert.public_bytes(serialization.Encoding.PEM))

    # Write client
    with open(client_key_file, "wb") as f:
        f.write(client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(client_cert_file, "wb") as f:
        f.write(client_cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"Generated CA + server + client certificates with SANs: {san_entries}")
    return cert_file, key_file, ca_cert_file


def _generate_certs_openssl(
    ca_key_file: str, ca_cert_file: str,
    cert_file: str, key_file: str,
    client_cert_file: str, client_key_file: str,
    san_entries: list[str],
) -> tuple[str, str, str]:
    """Generate CA + server certificate chain using openssl subprocess."""
    import subprocess

    san_value = ",".join(san_entries)

    # Generate CA
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", ca_key_file, "-out", ca_cert_file,
        "-days", "3650", "-nodes",
        "-subj", "/CN=distributed-llm CA/O=distributed-llm",
    ], check=True, capture_output=True)

    # Generate server CSR and sign with CA
    subprocess.run([
        "openssl", "req", "-newkey", "rsa:4096",
        "-keyout", key_file, "-out", os.path.join(os.path.dirname(cert_file), "server.csr"),
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost/O=distributed-llm",
        "-addext", f"subjectAltName={san_value}",
    ], check=True, capture_output=True)

    # Generate client CSR and sign with CA
    subprocess.run([
        "openssl", "req", "-newkey", "rsa:4096",
        "-keyout", client_key_file, "-out", os.path.join(os.path.dirname(cert_file), "client.csr"),
        "-days", "365", "-nodes",
        "-subj", "/CN=distributed-llm-client/O=distributed-llm",
    ], check=True, capture_output=True)

    # Sign server cert with CA
    with open(os.path.join(os.path.dirname(cert_file), "server.ext"), "w") as f:
        f.write(f"subjectAltName={san_value}\nextendedKeyUsage=serverAuth\n")

    subprocess.run([
        "openssl", "x509", "-req",
        "-in", os.path.join(os.path.dirname(cert_file), "server.csr"),
        "-CA", ca_cert_file, "-CAkey", ca_key_file,
        "-CAcreateserial", "-out", cert_file,
        "-days", "365", "-extfile", os.path.join(os.path.dirname(cert_file), "server.ext"),
    ], check=True, capture_output=True)

    # Sign client cert with CA
    with open(os.path.join(os.path.dirname(cert_file), "client.ext"), "w") as f:
        f.write("extendedKeyUsage=clientAuth\n")

    subprocess.run([
        "openssl", "x509", "-req",
        "-in", os.path.join(os.path.dirname(cert_file), "client.csr"),
        "-CA", ca_cert_file, "-CAkey", ca_key_file,
        "-CAcreateserial", "-out", client_cert_file,
        "-days", "365", "-extfile", os.path.join(os.path.dirname(cert_file), "client.ext"),
    ], check=True, capture_output=True)

    # Clean up temp files
    for f in ["server.csr", "client.csr", "server.ext", "client.ext", "ca.srl"]:
        p = os.path.join(os.path.dirname(cert_file), f)
        if os.path.exists(p):
            os.remove(p)

    logger.info(f"Generated CA + server + client certificates with SANs: {san_entries}")
    return cert_file, key_file, ca_cert_file


def load_tls_credentials(
    cert_file: str, key_file: str,
    ca_cert_file: str | None = None,
) -> grpc.ServerCredentials:
    """Load TLS server credentials from certificate files.

    When ca_cert_file is provided, require client certificates (mutual TLS).
    """
    with open(cert_file, "rb") as f:
        cert_chain = f.read()
    with open(key_file, "rb") as f:
        private_key = f.read()
    if ca_cert_file:
        with open(ca_cert_file, "rb") as f:
            ca_cert = f.read()
        return grpc.ssl_server_credentials(
            [(private_key, cert_chain)],
            root_certificates=ca_cert,
            require_client_auth=True,
        )
    return grpc.ssl_server_credentials([(private_key, cert_chain)])


def load_tls_channel_credentials(
    ca_cert_file: str,
    server_hostname: str = "localhost",
    client_cert_file: str | None = None,
    client_key_file: str | None = None,
) -> grpc.ChannelCredentials:
    """Load TLS channel credentials for client connections.

    When client_cert_file and client_key_file are provided, the client
    presents its certificate to the server (mutual TLS).
    The server_hostname is verified against the server cert's SANs.
    """
    with open(ca_cert_file, "rb") as f:
        ca_cert = f.read()

    if client_cert_file and client_key_file:
        with open(client_cert_file, "rb") as f:
            client_cert = f.read()
        with open(client_key_file, "rb") as f:
            client_key = f.read()
        return grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )
    return grpc.ssl_channel_credentials(root_certificates=ca_cert)


def create_secure_server(
    port: int, servicer,
    cert_file: str, key_file: str,
    ca_cert_file: str | None = None,
    max_workers: int = 10,
):
    """Create a TLS-secured gRPC server with optional mutual TLS."""
    from concurrent import futures
    from distllm.communication.node_pb2_grpc import (
        add_NodeServiceServicer_to_server,
        add_CoordinatorServiceServicer_to_server,
    )

    credentials = load_tls_credentials(cert_file, key_file, ca_cert_file)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    if hasattr(servicer, 'ForwardPass'):
        add_NodeServiceServicer_to_server(servicer, server)
    else:
        add_CoordinatorServiceServicer_to_server(servicer, server)

    server.add_secure_port(f"[::]:{port}", credentials)
    return server


def create_secure_channel(
    host: str, port: int, ca_cert_file: str,
    client_cert_file: str | None = None,
    client_key_file: str | None = None,
) -> grpc.SecureChannel:
    """Create a TLS-secured gRPC channel with optional mutual TLS."""
    credentials = load_tls_channel_credentials(
        ca_cert_file, host,
        client_cert_file=client_cert_file,
        client_key_file=client_key_file,
    )
    return grpc.secure_channel(f"{host}:{port}", credentials)
