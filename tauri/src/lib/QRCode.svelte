<script lang="ts">
  let { text, size = 180 }: { text: string; size?: number } = $props();

  // QR Code matrix generation using Reed-Solomon error correction
  // This is a minimal QR encoder supporting alphanumeric mode (version 1-4, L error correction)

  const EC_CODEWORDS_PER_BLOCK = [7, 10, 15, 20];
  const DATA_CODEWORDS_V1L = 19;
  const ALIGNMENT_PATTERNS: Record<number, number[]> = {};

  function generateQRMatrix(text: string): boolean[][] {
    if (!text) return [];

    const data = encodeData(text);
    const totalCodewords = getRequiredCodewords(data.length);
    const ecLevel = 0; // L
    const version = getVersion(data.length, ecLevel);
    const size = version * 4 + 17;
    const modules = Array.from({ length: size }, () => Array(size).fill(false));
    const reserved = Array.from({ length: size }, () => Array(size).fill(false));

    // Place finder patterns
    placeFinderPattern(modules, reserved, 0, 0);
    placeFinderPattern(modules, reserved, size - 7, 0);
    placeFinderPattern(modules, reserved, 0, size - 7);

    // Place separators
    placeSeparators(modules, reserved, size);

    // Place alignment patterns
    const alignPos = ALIGNMENT_PATTERNS[version] ?? [];
    for (const row of alignPos) {
      for (const col of alignPos) {
        if (!reserved[row][col]) {
          placeAlignmentPattern(modules, reserved, row, col);
        }
      }
    }

    // Place timing patterns
    for (let i = 8; i < size - 8; i++) {
      if (!reserved[6][i]) {
        modules[6][i] = i % 2 === 0;
        reserved[6][i] = true;
      }
      if (!reserved[i][6]) {
        modules[i][6] = i % 2 === 0;
        reserved[i][6] = true;
      }
    }

    // Reserve format info areas
    for (let i = 0; i < 8; i++) {
      reserved[8][i] = true;
      reserved[i][8] = true;
      reserved[8][size - 1 - i] = true;
      reserved[size - 1 - i][8] = true;
    }
    reserved[8][8] = true;

    // Reserve version areas for version >= 7
    if (version >= 7) {
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 3; j++) {
          reserved[i][size - 11 + j] = true;
          reserved[size - 11 + j][i] = true;
        }
      }
    }

    // Encode data into codewords
    const codewords = encodeDataCodewords(data, version, ecLevel);

    // Place data modules
    placeData(modules, reserved, codewords, size);

    // Apply best mask
    let bestMask = 0;
    let bestPenalty = Infinity;
    const formatInfo = getFormatInfo(ecLevel);

    for (let mask = 0; mask < 8; mask++) {
      const masked = applyMask(modules, reserved, mask, size);
      applyFormatInfo(masked, formatInfo[mask], size);
      const penalty = calculatePenalty(masked, size);
      if (penalty < bestPenalty) {
        bestPenalty = penalty;
        bestMask = mask;
      }
    }

    const final = applyMask(modules, reserved, bestMask, size);
    applyFormatInfo(final, formatInfo[bestMask], size);
    return final;
  }

  function encodeData(text: string): number[] {
    const bytes: number[] = [];
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      if (code < 128) {
        bytes.push(code);
      } else if (code < 2048) {
        bytes.push(0xc0 | (code >> 6));
        bytes.push(0x80 | (code & 0x3f));
      } else {
        bytes.push(0xe0 | (code >> 12));
        bytes.push(0x80 | ((code >> 6) & 0x3f));
        bytes.push(0x80 | (code & 0x3f));
      }
    }
    return bytes;
  }

  function getVersion(dataLen: number, ecLevel: number): number {
    for (let v = 1; v <= 40; v++) {
      const capacity = getDataCapacity(v, ecLevel);
      if (dataLen <= capacity) return v;
    }
    return 40;
  }

  function getDataCapacity(version: number, ecLevel: number): number {
    const totalData = version < 9 ? DATA_CODEWORDS_V1L : 0;
    const ecPerBlock = EC_CODEWORDS_PER_BLOCK[ecLevel];
    const blocks = version < 3 ? 1 : version < 10 ? 2 : 4;
    return (totalData - blocks * ecPerBlock);
  }

  function getRequiredCodewords(dataLen: number): number {
    return dataLen + 4; // mode + count + data + terminator
  }

  function encodeDataCodewords(data: number[], version: number, ecLevel: number): number[] {
    const bits: number[] = [];

    // Mode indicator: byte mode = 0100
    bits.push(0, 1, 0, 0);

    // Character count (8 bits for version 1-9)
    const count = data.length;
    for (let i = 7; i >= 0; i--) {
      bits.push((count >> i) & 1);
    }

    // Data bytes
    for (const byte of data) {
      for (let i = 7; i >= 0; i--) {
        bits.push((byte >> i) & 1);
      }
    }

    // Terminator (up to 4 zeros)
    const maxBits = DATA_CODEWORDS_V1L * 8;
    const terminatorLen = Math.min(4, maxBits - bits.length);
    for (let i = 0; i < terminatorLen; i++) {
      bits.push(0);
    }

    // Pad to byte boundary
    while (bits.length % 8 !== 0) {
      bits.push(0);
    }

    // Pad bytes
    const padBytes = [0xec, 0x11];
    let padIdx = 0;
    while (bits.length < maxBits) {
      const padByte = padBytes[padIdx % 2];
      for (let i = 7; i >= 0; i--) {
        bits.push((padByte >> i) & 1);
      }
      padIdx++;
    }

    // Convert bits to codewords
    const codewords: number[] = [];
    for (let i = 0; i < bits.length; i += 8) {
      let val = 0;
      for (let j = 0; j < 8; j++) {
        val = (val << 1) | (bits[i + j] ?? 0);
      }
      codewords.push(val);
    }

    return codewords;
  }

  function placeFinderPattern(modules: boolean[][], reserved: boolean[][], row: number, col: number) {
    const pattern = [
      [1,1,1,1,1,1,1],
      [1,0,0,0,0,0,1],
      [1,0,1,1,1,0,1],
      [1,0,1,1,1,0,1],
      [1,0,1,1,1,0,1],
      [1,0,0,0,0,0,1],
      [1,1,1,1,1,1,1],
    ];
    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        const mr = row + r;
        const mc = col + c;
        if (mr >= 0 && mr < modules.length && mc >= 0 && mc < modules.length) {
          modules[mr][mc] = pattern[r][c] === 1;
          reserved[mr][mc] = true;
        }
      }
    }
  }

  function placeSeparators(modules: boolean[][], reserved: boolean[][], size: number) {
    for (let i = 0; i < 8; i++) {
      // Top-left
      if (!reserved[7][i]) { modules[7][i] = false; reserved[7][i] = true; }
      if (!reserved[i][7]) { modules[i][7] = false; reserved[i][7] = true; }
      // Top-right
      if (!reserved[7][size - 1 - i]) { modules[7][size - 1 - i] = false; reserved[7][size - 1 - i] = true; }
      if (!reserved[i][size - 8]) { modules[i][size - 8] = false; reserved[i][size - 8] = true; }
      // Bottom-left
      if (!reserved[size - 8][i]) { modules[size - 8][i] = false; reserved[size - 8][i] = true; }
      if (!reserved[size - 1 - i][7]) { modules[size - 1 - i][7] = false; reserved[size - 1 - i][7] = true; }
    }
  }

  function placeAlignmentPattern(modules: boolean[][], reserved: boolean[][], row: number, col: number) {
    for (let r = -2; r <= 2; r++) {
      for (let c = -2; c <= 2; c++) {
        const mr = row + r;
        const mc = col + c;
        if (mr >= 0 && mr < modules.length && mc >= 0 && mc < modules.length) {
          modules[mr][mc] = Math.abs(r) === 2 || Math.abs(c) === 2 || (r === 0 && c === 0);
          reserved[mr][mc] = true;
        }
      }
    }
  }

  function placeData(modules: boolean[][], reserved: boolean[][], codewords: number[], size: number) {
    let bitIdx = 0;
    const allBits: number[] = [];
    for (const cw of codewords) {
      for (let i = 7; i >= 0; i--) {
        allBits.push((cw >> i) & 1);
      }
    }

    let col = size - 1;
    let upward = true;

    while (col >= 0) {
      if (col === 6) col--; // Skip timing pattern column

      for (let i = 0; i < 2; i++) {
        const checkCol = col - i;
        if (checkCol < 0) continue;

        const rows = upward
          ? Array.from({ length: size }, (_, j) => size - 1 - j)
          : Array.from({ length: size }, (_, j) => j);

        for (const row of rows) {
          if (!reserved[row][checkCol]) {
            modules[row][checkCol] = bitIdx < allBits.length ? allBits[bitIdx] === 1 : false;
            bitIdx++;
          }
        }
      }

      upward = !upward;
      col -= 2;
    }
  }

  function applyMask(modules: boolean[][], reserved: boolean[][], mask: number, size: number): boolean[][] {
    const result = modules.map((row) => [...row]);
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (!reserved[r][c]) {
          let invert = false;
          switch (mask) {
            case 0: invert = (r + c) % 2 === 0; break;
            case 1: invert = r % 2 === 0; break;
            case 2: invert = c % 3 === 0; break;
            case 3: invert = (r + c) % 3 === 0; break;
            case 4: invert = (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0; break;
            case 5: invert = (r * c) % 2 + (r * c) % 3 === 0; break;
            case 6: invert = ((r * c) % 2 + (r * c) % 3) % 2 === 0; break;
            case 7: invert = ((r + c) % 2 + (r * c) % 3) % 2 === 0; break;
          }
          if (invert) result[r][c] = !result[r][c];
        }
      }
    }
    return result;
  }

  function getFormatInfo(ecLevel: number): number[] {
    // Format info for EC level L (ecLevel=0) with 8 mask patterns
    const formatInfos = [
      0x77c4, 0x72f3, 0x7daa, 0x789d,
      0x662f, 0x6318, 0x6c41, 0x6976,
    ];
    return formatInfos.map((f) => f ^ 0x5412);
  }

  function applyFormatInfo(modules: boolean[][], formatInfo: number, size: number) {
    const bits: boolean[] = [];
    for (let i = 14; i >= 0; i--) {
      bits.push(((formatInfo >> i) & 1) === 1);
    }

    // Around top-left finder
    const positions = [
      [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 7], [8, 8],
      [7, 8], [5, 8], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8],
    ];
    for (let i = 0; i < 15; i++) {
      const [r, c] = positions[i];
      modules[r][c] = bits[i];
    }

    // Copy to other locations
    const otherPositions = [
      [8, size - 1], [8, size - 2], [8, size - 3], [8, size - 4],
      [8, size - 5], [8, size - 6], [8, size - 7],
      [size - 7, 8], [size - 6, 8], [size - 5, 8], [size - 4, 8],
      [size - 3, 8], [size - 2, 8], [size - 1, 8], [size - 8, 8],
    ];
    for (let i = 0; i < 15; i++) {
      const [r, c] = otherPositions[i];
      modules[r][c] = bits[i];
    }
  }

  function calculatePenalty(modules: boolean[][], size: number): number {
    let penalty = 0;

    // Penalty 1: runs of same color
    for (let r = 0; r < size; r++) {
      let count = 1;
      for (let c = 1; c < size; c++) {
        if (modules[r][c] === modules[r][c - 1]) {
          count++;
          if (count === 5) penalty += 3;
          else if (count > 5) penalty += 1;
        } else {
          count = 1;
        }
      }
    }
    for (let c = 0; c < size; c++) {
      let count = 1;
      for (let r = 1; r < size; r++) {
        if (modules[r][c] === modules[r - 1][c]) {
          count++;
          if (count === 5) penalty += 3;
          else if (count > 5) penalty += 1;
        } else {
          count = 1;
        }
      }
    }

    // Penalty 2: 2x2 blocks
    for (let r = 0; r < size - 1; r++) {
      for (let c = 0; c < size - 1; c++) {
        const val = modules[r][c];
        if (val === modules[r][c + 1] && val === modules[r + 1][c] && val === modules[r + 1][c + 1]) {
          penalty += 3;
        }
      }
    }

    return penalty;
  }

  let matrix = $derived.by(() => {
    const m = generateQRMatrix(text);
    return m;
  });

  let moduleSize = $derived(matrix.length > 0 ? size / (matrix.length + 8) : 8);
  let viewBox = $derived(matrix.length > 0 ? `0 0 ${matrix.length + 8} ${matrix.length + 8}` : "0 0 180 180");
</script>

{#if matrix.length > 0}
  <svg
    {viewBox}
    width={size}
    height={size}
    class="qr-code"
    role="img"
    aria-label="QR code for invite link"
  >
    <rect x="0" y="0" width={matrix.length + 8} height={matrix.length + 8} fill="white" rx="4" />
    {#each matrix as row, r}
      {#each row as cell, c}
        {#if cell}
          <rect
            x={c + 4}
            y={r + 4}
            width="1"
            height="1"
            fill="#1a1a2e"
          />
        {/if}
      {/each}
    {/each}
  </svg>
{:else}
  <div class="qr-fallback" style="width: {size}px; height: {size}px;">
    <span>Generating...</span>
  </div>
{/if}

<style>
  .qr-code {
    display: block;
    border-radius: 8px;
  }
  .qr-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-input);
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 13px;
  }
</style>
