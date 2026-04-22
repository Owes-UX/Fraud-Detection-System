"""IO: writes the ASCII output file."""
import os

def write_output(flagged_ids, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="ascii") as f:
        for tid in flagged_ids:
            f.write(str(tid) + "\n")
    print(f"[IO] Output written: {output_path} ({len(flagged_ids)} transactions flagged)")
