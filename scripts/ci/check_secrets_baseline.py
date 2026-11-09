import json
import sys


def load_hashes(path: str) -> set[str]:
    # detect-secrets baselines are JSON, but the file can land on disk as
    # UTF-16 (PowerShell `>` redirection writes UTF-16LE with a BOM) rather
    # than UTF-8. Accept either so the gate does not crash on the baseline's
    # encoding alone.
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            data = json.loads(raw.decode(encoding))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    else:
        raise SystemExit(f"ERROR: cannot parse {path} as UTF-8 or UTF-16 JSON")
    return {s["hashed_secret"] for r in data.get("results", {}).values() for s in r}


def main() -> None:
    old = load_hashes(".secrets.baseline")
    new = load_hashes(".secrets.new.baseline")

    added = new - old
    if added:
        print(f"ERROR: {len(added)} new secret(s) detected")
        sys.exit(1)

    print("No new secrets detected")


if __name__ == "__main__":
    main()
