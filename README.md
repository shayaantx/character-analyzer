# Character Analyzer

Scans a folder of videos, finds faces in sampled frames, matches them against
reference images you provide, and saves the frames containing each character.

## Running with Docker (recommended)

Requires Docker, an NVIDIA driver, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image includes CUDA, cuDNN and the face model, so nothing else needs installing.

Put your files next to `docker-compose.yml`:

```
videos/   the show
refs/     reference images, one subfolder per character (see below)
output/   created for results
```

```bash
mkdir -p output          # create it first so it isn't owned by root
docker compose build
docker compose run --rm analyzer                          # all characters in refs/
docker compose run --rm analyzer --only walter --min-gap 3
```

Any arguments after `analyzer` are passed to the app (see options below).
Output files are written as UID/GID 1000 by default. If your user is different, run
`export UID GID=$(id -g)` before `docker compose run`.

Without compose:

```bash
docker build -t character-analyzer .
docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/videos:/data/videos:ro" -v "$PWD/refs:/data/refs:ro" -v "$PWD/output:/data/output" \
  character-analyzer --annotate
```

To check that the container sees the GPU, run
`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
If CUDA isn't usable, the app prints a warning and falls back to the CPU.

## Setup without Docker

Requires Python 3.10+ and an NVIDIA GPU with CUDA 12 + cuDNN 9 (what current
`onnxruntime-gpu` builds expect; check the onnxruntime CUDA compatibility
table if you're on something else).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- On Windows, `insightface` may need the "Desktop development with C++"
  workload from Visual Studio Build Tools to install.
- If `onnxruntime` (CPU-only) is also installed, uninstall it, since it can
  shadow `onnxruntime-gpu`: `pip uninstall onnxruntime`.
- The face model (`buffalo_l`, ~300 MB) downloads to `~/.insightface` on
  first run. Its license allows non-commercial use only.

## Reference images

Give each character 5–10 clear face images, ideally screenshots from the show
with a mix of angles, lighting and looks. There are two ways to pass them in:

**A folder per character** (recommended):

```
refs/
  walter/  1.jpg 2.jpg 3.jpg
  jesse/   1.jpg 2.jpg
```
```bash
python analyze.py --refs-dir refs   # or just: python analyze.py (uses ./refs if present)
```

**Individually on the command line** (a file or a folder, repeatable):

```bash
python analyze.py --ref walter=refs/walter --ref jesse=pics/jesse_closeup.png
```

At startup it reports how many images per character were usable, and warns
about images with no face or more than one face.

Tip: include references for other main cast members even if you don't want
their frames, and use `--only` to pick who gets saved. With more people to
compare against, fewer faces get wrongly matched to your target characters.

```bash
python analyze.py --refs-dir refs --only walter
```

## Output

```
output/
  walter/  S01E01_00h03m12s500.jpg ...
  jesse/   ...
  detections.csv   # video, timestamp, character, similarity, face box, image path
```

A frame with several matched characters is saved into each one's folder.
Re-running skips videos that already finished (`--no-resume` to redo them).

## Useful options

| Option | Default | Notes |
|---|---|---|
| `--videos` | `./videos` | Searched recursively |
| `--fps` | `2` | Frames analyzed per second of video; `0` = every frame |
| `--threshold` | `0.45` | Raise (0.5–0.55) if wrong people are matched; lower (0.35–0.4) if appearances are missed |
| `--min-gap` | `0` | e.g. `3` = at most one saved frame per character every 3 s |
| `--min-face` | `40` | Ignore tiny background faces (pixels) |
| `--annotate` | off | Draw boxes and names on saved frames. Handy while tuning |
| `--gpu` | `0` | GPU id, `-1` for CPU |

Run `python analyze.py --help` for everything.

## Tuning workflow

1. Run on one episode with `--annotate`:
   `python analyze.py --videos videos/S01E01.mkv --refs-dir refs --annotate --out test_run`
2. Look through the saved frames and the `similarity` column in `detections.csv`.
3. Adjust `--threshold` and/or add reference images for looks that were missed.
# character-analyzer
