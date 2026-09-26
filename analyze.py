#!/usr/bin/env python3
"""Scan a folder of videos and save the frames that contain specific characters.

Characters are defined by reference images you supply. Faces found in sampled
frames are matched against those references using InsightFace (ArcFace
embeddings), and matching frames are written to <out>/<character>/.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".wmv", ".flv", ".ts", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FIELDS = ["video", "timestamp", "seconds", "frame", "character", "similarity", "x1", "y1", "x2", "y2", "image"]


# ---------------------------------------------------------------- image io ---
# cv2.imread/imwrite can't handle non-ASCII paths on Windows; go through numpy.

def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: Path, img, quality: int) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())


# ------------------------------------------------------------------- model ---

def load_face_model(gpu: int, det_size: int):
    import onnxruntime
    from insightface.app import FaceAnalysis

    if gpu >= 0 and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        print("WARNING: this onnxruntime build has no CUDA support; running on CPU (slow).\n"
              "         Make sure only onnxruntime-gpu is installed (not onnxruntime).", file=sys.stderr)
        gpu = -1

    if gpu >= 0:
        providers = [("CUDAExecutionProvider", {"device_id": gpu}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name="buffalo_l", root=os.environ.get("INSIGHTFACE_HOME", "~/.insightface"),
                       providers=providers,
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=gpu, det_size=(det_size, det_size))

    # CUDA can be compiled in yet still fail to start (no GPU passed to the container,
    # missing CUDA/cuDNN libs); onnxruntime then silently falls back to CPU.
    active = app.models["detection"].session.get_providers()
    if gpu >= 0 and "CUDAExecutionProvider" not in active:
        print("WARNING: CUDA failed to initialise, running on CPU (slow). Check that the GPU is visible\n"
              "         (nvidia-smi inside the container) and the onnxruntime log above for the reason.",
              file=sys.stderr)
    print(f"Running on: {'GPU ' + str(gpu) if 'CUDAExecutionProvider' in active else 'CPU'}")
    return app


def face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


# -------------------------------------------------------------- references ---

def images_in(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"reference path not found: {path}")


def collect_reference_images(ref_specs: list[str], refs_dir: str | None) -> dict[str, list[Path]]:
    """Build {character: [image paths]} from --ref NAME=PATH and/or --refs-dir."""
    refs: dict[str, list[Path]] = {}

    for spec in ref_specs:
        if "=" not in spec:
            sys.exit(f"--ref must look like NAME=PATH, got: {spec!r}")
        name, path = spec.split("=", 1)
        refs.setdefault(name.strip(), []).extend(images_in(Path(path)))

    if refs_dir:
        root = Path(refs_dir)
        if not root.is_dir():
            sys.exit(f"--refs-dir is not a directory: {root}")
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                refs.setdefault(entry.name, []).extend(images_in(entry))
            elif entry.suffix.lower() in IMAGE_EXTS:
                refs.setdefault(entry.stem, []).append(entry)

    return {name: paths for name, paths in refs.items() if paths}


def detect_reference_face(app, img):
    """Find the main face in a reference image. Tightly cropped headshots often
    fail detection, so retry with a padded border."""
    faces = app.get(img)
    if not faces:
        h, w = img.shape[:2]
        pad = max(h, w) // 2
        faces = app.get(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0))
    if not faces:
        return None, 0
    return max(faces, key=face_area), len(faces)


def embed_references(app, ref_images: dict[str, list[Path]]):
    """Returns (names, embeddings) where names[i] is the character for row i."""
    names, embs = [], []
    print("Loading reference images:")
    for name, paths in ref_images.items():
        usable = 0
        for p in paths:
            img = read_image(p)
            if img is None:
                print(f"  ! {p}: could not read image")
                continue
            face, n = detect_reference_face(app, img)
            if face is None:
                print(f"  ! {p}: no face found, skipping")
                continue
            if n > 1:
                print(f"  ! {p}: {n} faces found, using the largest")
            names.append(name)
            embs.append(face.normed_embedding)
            usable += 1
        print(f"  {name}: {usable}/{len(paths)} images usable")
        if usable == 0:
            print(f"  ! {name} has no usable reference images and will never match")
    if not embs:
        sys.exit("No usable reference faces. Check your reference images.")
    return names, np.stack(embs).astype(np.float32)


# ------------------------------------------------------------------ videos ---

def find_videos(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def format_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}"


def match_faces(faces, names, ref_embs, threshold):
    """Returns {character: (similarity, face)}, keeping the best face per character."""
    if not faces:
        return {}
    sims = np.stack([f.normed_embedding for f in faces]) @ ref_embs.T
    matches = {}
    for face, row in zip(faces, sims):
        j = int(row.argmax())
        sim = float(row[j])
        if sim < threshold:
            continue
        name = names[j]
        if name not in matches or sim > matches[name][0]:
            matches[name] = (sim, face)
    return matches


def annotate(frame, matches):
    img = frame.copy()
    for name, (sim, face) in matches.items():
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {sim:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def process_video(app, video: Path, names, ref_embs, args, out_dir: Path, writer) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  ! could not open {video}, skipping")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:
        fps = 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    step = max(1, round(fps / args.fps)) if args.fps > 0 else 1

    last_saved: dict[str, float] = {}
    saved = 0
    idx = 0
    with tqdm(total=total, unit="frame", desc=video.name, leave=False) as bar:
        # grab() every frame but only decode+analyze every `step`th one.
        while cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    saved += handle_frame(app, frame, idx, idx / fps, video, names, ref_embs,
                                          args, out_dir, writer, last_saved)
            idx += 1
            bar.update(1)
    cap.release()
    return saved


def handle_frame(app, frame, idx, seconds, video, names, ref_embs, args, out_dir, writer, last_saved) -> int:
    faces = [f for f in app.get(frame)
             if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= args.min_face]
    matches = match_faces(faces, names, ref_embs, args.threshold)
    if args.only:
        matches = {n: m for n, m in matches.items() if n in args.only}
    matches = {n: m for n, m in matches.items()
               if seconds - last_saved.get(n, -1e9) >= args.min_gap}
    if not matches:
        return 0

    img = annotate(frame, matches) if args.annotate else frame
    filename = f"{video.stem}_{format_ts(seconds)}.jpg"
    for name, (sim, face) in matches.items():
        char_dir = out_dir / name
        char_dir.mkdir(parents=True, exist_ok=True)
        dest = char_dir / filename
        write_image(dest, img, args.quality)
        last_saved[name] = seconds
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        writer.writerow({
            "video": str(video), "timestamp": format_ts(seconds), "seconds": f"{seconds:.3f}",
            "frame": idx, "character": name, "similarity": f"{sim:.4f}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "image": str(dest),
        })
    return len(matches)


# -------------------------------------------------------------------- main ---

def parse_args():
    p = argparse.ArgumentParser(
        description="Find frames containing specific characters in a folder of videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--videos", default="./videos", help="folder of videos (searched recursively) or a single video")
    p.add_argument("--ref", action="append", default=[], metavar="NAME=PATH",
                   help="reference image or folder of images for a character; repeatable")
    p.add_argument("--refs-dir", help="folder where each subfolder (or image file name) is a character "
                                      "(defaults to ./refs if it exists and no --ref is given)")
    p.add_argument("--only", nargs="+", metavar="NAME",
                   help="only save frames for these characters (others still help avoid mismatches)")
    p.add_argument("--out", default="./output", help="output folder")
    p.add_argument("--fps", type=float, default=2.0, help="frames per second to analyze (0 = every frame)")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="min cosine similarity to count as a match; raise for fewer false positives")
    p.add_argument("--min-face", type=int, default=40, help="ignore faces smaller than this many pixels")
    p.add_argument("--min-gap", type=float, default=0.0,
                   help="min seconds between saved frames of the same character in a video (reduces near-duplicates)")
    p.add_argument("--annotate", action="store_true", help="draw boxes and names on saved frames")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality of saved frames")
    p.add_argument("--gpu", type=int, default=0, help="GPU id; -1 for CPU")
    p.add_argument("--det-size", type=int, default=640, help="face detector input size")
    p.add_argument("--no-resume", action="store_true", help="reprocess videos already finished in a previous run")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.ref and not args.refs_dir and Path("refs").is_dir():
        args.refs_dir = "refs"
    if not args.ref and not args.refs_dir:
        sys.exit("Specify reference images with --ref NAME=PATH and/or --refs-dir DIR.")

    ref_images = collect_reference_images(args.ref, args.refs_dir)
    if not ref_images:
        sys.exit("No reference images found.")
    if args.only:
        unknown = set(args.only) - ref_images.keys()
        if unknown:
            sys.exit(f"--only names have no reference images: {', '.join(sorted(unknown))}")

    videos = find_videos(Path(args.videos))
    if not videos:
        sys.exit(f"No videos found in {args.videos}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_file = out_dir / ".processed"
    done = set() if args.no_resume or not done_file.exists() else set(done_file.read_text().splitlines())

    app = load_face_model(args.gpu, args.det_size)
    names, ref_embs = embed_references(app, ref_images)

    todo = [v for v in videos if str(v.resolve()) not in done]
    if len(todo) < len(videos):
        print(f"Skipping {len(videos) - len(todo)} already-processed video(s) (use --no-resume to redo).")
    print(f"Processing {len(todo)} video(s) at {args.fps or 'every'} fps...")

    csv_path = out_dir / "detections.csv"
    new_csv = not csv_path.exists() or args.no_resume
    total_saved = 0
    with open(csv_path, "w" if new_csv else "a", newline="", encoding="utf-8") as fh, \
         open(done_file, "w" if args.no_resume else "a", encoding="utf-8") as done_fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_csv:
            writer.writeheader()
        for video in tqdm(todo, unit="video", desc="Videos"):
            n = process_video(app, video, names, ref_embs, args, out_dir, writer)
            total_saved += n
            fh.flush()
            done_fh.write(str(video.resolve()) + "\n")
            done_fh.flush()
            tqdm.write(f"{video.name}: {n} frame(s) saved")

    print(f"Done. {total_saved} frame(s) saved to {out_dir}/, log in {csv_path}")


if __name__ == "__main__":
    main()
