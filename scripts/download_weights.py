import socket
import sys
import urllib.request
from pathlib import Path
from urllib.error import URLError

_BASE_URL = "https://github.com/Rejnhard/vrl_framework/releases/download/v0.1.0-alpha"
CHKPT_URI = f"{_BASE_URL}/checkpoint_gen_8000.pt"
LORA_URI = f"{_BASE_URL}/lora_skill_gen_8000.pt"

_workspace = Path(__file__).resolve().parents[1]
STORAGE_PATH = _workspace / "test_data"

socket.setdefaulttimeout(30.0)


def _cli_tracker(b_count, b_size, t_size):
    if t_size < 0:
        return

    downloaded = b_count * b_size
    ratio = downloaded / t_size

    if ratio > 1.0:
        ratio = 1.0

    pct = int(ratio * 100)
    ticks = int(40 * ratio)
    visual = "#" * ticks + "." * (40 - ticks)

    sys.stdout.write(f"\rFetching |{visual}| {pct}% ")
    sys.stdout.flush()


def _pull_payload(src, dst):
    name = dst.name
    print(f"\nInit transfer: {name}")

    try:
        urllib.request.urlretrieve(src, str(dst), reporthook=_cli_tracker)
        print(f"\nSaved -> {dst}")
    except URLError as err:
        print(f"\nNetwork failure while pulling {name}. Trace: {err}")
        sys.exit(1)


def main():
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    main_chkpt = STORAGE_PATH / "checkpoint_gen_8000.pt"
    lora_chkpt = STORAGE_PATH / "lora_skill_gen_8000.pt"

    print("--- VRL Weights Fetcher ---")

    if main_chkpt.exists():
        print(f"\nSkipping (found local copy): {main_chkpt}")
    else:
        _pull_payload(CHKPT_URI, main_chkpt)

    if lora_chkpt.exists():
        print(f"\nSkipping (found local copy): {lora_chkpt}")
    else:
        _pull_payload(LORA_URI, lora_chkpt)

    print("\nAll weights ready.")


if __name__ == "__main__":
    main()
