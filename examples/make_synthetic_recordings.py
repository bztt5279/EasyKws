"""Generate tones for interface tests, not speech or a recognition benchmark."""
import argparse
import json
import math
from pathlib import Path
import struct
import wave


def create(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    audio = output / "audio"
    audio.mkdir()
    rows = []
    for person in range(5):
        for label in (0, -1):
            sample = f"person_{person}_{'target' if label == 0 else 'negative'}"
            frequency = 220 + person * 31 + (0 if label == 0 else 430)
            samples = [int(4000 * math.sin(2 * math.pi * frequency * n / 16000))
                       for n in range(3200)]
            with wave.open(str(audio / (sample + ".wav")), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
            rows.append({"sample_id": sample, "speaker_id": f"person_{person}",
                         "source_recording_id": f"session_{person}", "audio_path": sample + ".wav",
                         "label": label, "group": "target" if label == 0 else "background",
                         "expected_phrase": "你好小A" if label == 0 else "",
                         "review_status": "pass", "origin": "synthetic"})
    manifest = output / "recordings.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(create(args.output))
