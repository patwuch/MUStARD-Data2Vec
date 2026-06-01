"""Keyframe extraction using Katna.

Step 1 of the pipeline:
  1. Extract one keyframe per video from the raw video folders.
  2. Identify videos for which Katna failed to extract a keyframe.
  3. Move those failed videos to the fallback folders so feature_extraction.py
     can process them frame-by-frame instead.

Usage:
    python src/keyframe_extraction.py --config config.yaml
"""

import argparse
import os
import shutil

import yaml


def extract_keyframes(video_dir: str, output_dir: str, n_frames: int = 1) -> None:
    """Run Katna on every .mp4 in video_dir and write keyframes to output_dir."""
    from Katna.video import Video
    from Katna.writer import KeyFrameDiskWriter

    os.makedirs(output_dir, exist_ok=True)
    vd = Video()
    writer = KeyFrameDiskWriter(location=output_dir)

    for fname in sorted(os.listdir(video_dir)):
        if not fname.endswith(".mp4"):
            continue
        video_path = os.path.join(video_dir, fname)
        try:
            vd.extract_video_keyframes(
                no_of_frames=n_frames,
                file_path=video_path,
                writer=writer,
            )
        except Exception as e:
            print(f"[WARN] Katna failed for {fname}: {e}")


def find_missing_keyframes(video_dir: str, keyframe_dir: str) -> list:
    """Return list of video filenames whose keyframe is absent from keyframe_dir."""
    keyframe_files = set(os.listdir(keyframe_dir)) if os.path.isdir(keyframe_dir) else set()
    missing = []
    for fname in os.listdir(video_dir):
        if not fname.endswith(".mp4"):
            continue
        expected = fname.replace(".mp4", "_0.jpeg")
        if expected not in keyframe_files:
            missing.append(fname)
    return missing


def isolate_fallback_videos(
    missing_videos: list, source_dir: str, dest_dir: str
) -> None:
    """Move videos that lack a keyframe to the fallback directory."""
    os.makedirs(dest_dir, exist_ok=True)
    for fname in missing_videos:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_dir, fname)
        shutil.move(src, dst)
        print(f"Moved {fname} → fallback")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    p = cfg["paths"]

    for split in ("utterance", "context"):
        raw_dir = p[f"{split}_videos_raw"]
        keyframe_dir = p[f"{split}_keyframes"]
        fallback_dir = p[f"{split}_videos_fallback"]

        print(f"\n=== Extracting keyframes for {split} videos ===")
        extract_keyframes(raw_dir, keyframe_dir)

        missing = find_missing_keyframes(raw_dir, keyframe_dir)
        print(f"Videos without a keyframe: {len(missing)}")

        if missing:
            isolate_fallback_videos(missing, raw_dir, fallback_dir)

    print("\nKeyframe extraction complete.")


if __name__ == "__main__":
    main()
