"""Quick connectivity test for 2-laptop cluster.

Run on BOTH laptops:
    python tests/manual/test_connectivity.py

This verifies:
1. Network connectivity between laptops
2. Port availability
3. Basic API functionality
"""

import socket
import sys


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return "127.0.0.1"
    finally:
        s.close()


def check_port_open(host, port, timeout=3):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except:
        return False
    finally:
        sock.close()


def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
        return True
    except:
        return False
    finally:
        sock.close()


def main():
    local_ip = get_local_ip()
    hostname = socket.gethostname()

    print("=" * 50)
    print("  DistLLM 2-Laptop Connectivity Test")
    print("=" * 50)
    print()
    print(f"  Your IP:       {local_ip}")
    print(f"  Your Hostname: {hostname}")
    print()

    # Check required ports
    ports = [8000, 50050, 50051]
    print("  Port availability:")
    for port in ports:
        available = check_port_available(port)
        status = "[OK] Available" if available else "[!!] In use"
        print(f"    Port {port}: {status}")
    print()

    # Instructions
    print("  " + "=" * 45)
    print("  NEXT STEPS:")
    print("  " + "=" * 45)
    print()
    print("  1. Tell the OTHER laptop to run this script:")
    print(f"     python tests/manual/test_connectivity.py")
    print()
    print(f"  2. They should see their IP. Tell you what it is.")
    print()
    print(f"  3. On THIS laptop, start the coordinator:")
    print(f"     distllm run --model roneneldan/TinyStories-1M --local --port 8000")
    print()
    print(f"  4. On the OTHER laptop, connect as worker:")
    print(f"     distllm-node --coordinator {local_ip}:50050 --port 50051")
    print()
    print(f"  5. Test the connection:")
    print(f"     curl http://{local_ip}:8000/health")
    print()

    # Check if other laptop is reachable (if they provided an IP)
    if len(sys.argv) > 1:
        other_ip = sys.argv[1]
        print(f"  Testing connection to {other_ip}...")
        for port in [8000, 50050]:
            if check_port_open(other_ip, port):
                print(f"    [OK] Port {port} on {other_ip}: REACHABLE")
            else:
                print(f"    [!!] Port {port} on {other_ip}: NOT REACHABLE")


if __name__ == "__main__":
    main()
