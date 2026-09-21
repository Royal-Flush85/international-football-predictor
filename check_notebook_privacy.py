"""Check notebook JSON before publishing; --clean removes unsafe text outputs.

Run from anywhere: python check_notebook_privacy.py [--clean]
Source cells are reported, never automatically rewritten. Charts and clean
outputs are retained. This is a text-path/error check, not a full secret scanner
or an inspection of text embedded in images.
"""
import argparse
import json
import re
from pathlib import Path

PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]+"
    r"|(?<![A-Za-z0-9:/])/(?:Users|home)/[^/\s]+/"
)


def output_text(output):
    # Image data is intentionally excluded; visually review published figures.
    parts = []
    for key in ("text", "traceback", "ename", "evalue"):
        value = output.get(key, "")
        parts.append("".join(value) if isinstance(value, list) else str(value))
    for key, value in output.get("data", {}).items():
        if key.startswith("text/"):
            parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    issues = 0
    for path in sorted(root.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for number, cell in enumerate(notebook["cells"], 1):
            if PATH_PATTERN.search("".join(cell.get("source", []))):
                print(f"SOURCE PATH: {path.name}, cell {number}; edit this source manually.")
                issues += 1
            if "outputs" not in cell:
                continue
            kept = []
            removed = False
            for output in cell["outputs"]:
                unsafe = output.get("output_type") == "error" or PATH_PATTERN.search(output_text(output))
                if unsafe:
                    if args.clean:
                        removed = changed = True
                        print(f"CLEARED OUTPUT: {path.name}, cell {number}")
                    else:
                        issues += 1
                        print(f"OUTPUT PATH/ERROR: {path.name}, cell {number}")
                        kept.append(output)
                else:
                    kept.append(output)
            if args.clean and removed:
                cell["outputs"] = kept
                cell["execution_count"] = None
        if changed:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if issues:
        raise SystemExit(f"Found {issues} item(s). Clear flagged outputs with --clean; edit flagged source paths.")
    print("Notebook text-path and error-output check passed.")


if __name__ == "__main__":
    main()
