"""Generate app icons for Tauri project."""
import struct, zlib, os

def make_png(width, height, r, g, b):
    """Create a minimal solid-color PNG."""
    def chunk(ctype, data):
        c = ctype + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    sig = b'\x89PNG\r\n\x1a\n'
    # color type 6 = RGBA
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
    raw = b''
    for _ in range(height):
        raw += b'\x00'
        raw += bytes([r, g, b, 255]) * width
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return sig + ihdr + idat + iend

def make_ico(png_data):
    """Wrap PNG in ICO container."""
    size = 32
    header = struct.pack('<HHH', 0, 1, 1)
    entry = struct.pack('<BBBBHHII', size, size, 0, 0, 1, 32, len(png_data), 22)
    return header + entry + png_data

def make_icns(png_data):
    """Wrap PNG in ICNS container."""
    icon = b'ic07' + struct.pack('>I', len(png_data) + 8) + png_data
    size = 8 + len(icon)
    return b'icns' + struct.pack('>I', size) + icon

def generate_all(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # App icon colors (dark blue/purple gradient simplified as solid)
    colors = {
        '32x32.png': (32, 32, 0x6C, 0x5C, 0xE7),
        '128x128.png': (128, 128, 0x6C, 0x5C, 0xE7),
        '128x128@2x.png': (256, 256, 0x6C, 0x5C, 0xE7),
    }

    # Generate app icons
    png_data = {}
    for name, (w, h, r, g, b) in colors.items():
        data = make_png(w, h, r, g, b)
        path = os.path.join(output_dir, name)
        with open(path, 'wb') as f:
            f.write(data)
        png_data[name] = data
        print(f"  Created {path} ({w}x{h})")

    # 32x32 for ICO
    small_png = make_png(32, 32, 0x6C, 0x5C, 0xE7)
    ico_path = os.path.join(output_dir, 'icon.ico')
    with open(ico_path, 'wb') as f:
        f.write(make_ico(small_png))
    print(f"  Created {ico_path}")

    # ICNS (simplified - just one size)
    icns_path = os.path.join(output_dir, 'icon.icns')
    with open(icns_path, 'wb') as f:
        f.write(make_icns(small_png))
    print(f"  Created {icns_path}")

    # Also create a copy as icon.png for the default window icon
    icon_png_path = os.path.join(output_dir, 'icon.png')
    with open(icon_png_path, 'wb') as f:
        f.write(make_png(128, 128, 0x6C, 0x5C, 0xE7))
    print(f"  Created {icon_png_path}")

if __name__ == '__main__':
    output = os.path.join(os.path.dirname(__file__), '..', 'src-tauri', 'icons')
    print(f"Generating icons in {os.path.abspath(output)}")
    generate_all(output)
    print("Done!")
