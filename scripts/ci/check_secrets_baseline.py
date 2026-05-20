import json
import sys


def load_hashes(path: str) -> set[str]:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
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
