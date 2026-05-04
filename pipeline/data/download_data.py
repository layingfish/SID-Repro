from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DATASETS: dict[str, dict[str, Any]] = {
    "microlens_50k": {
        "base_url": "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset",
        "files": [
            "MicroLens-50k_pairs.csv",
            "MicroLens-50k_pairs.tsv",
            "MicroLens-50k_titles.csv",
            "MicroLens-50k_likes_and_views.txt",
            "readme.txt",
        ],
    },
    "microlens_100k": {
        "base_url": "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset",
        "files": [
            "MicroLens-100k_pairs.csv",
            "MicroLens-100k_pairs.tsv",
            "MicroLens-100k_title_en.csv",
            "tags_to_summary.csv",
            "MicroLens-100k_likes_and_views.txt",
            "MicroLens-100k_comment_en.txt",
            "readme.txt",
        ],
    },
    "microlens_1m": {
        "base_url": "https://recsys.westlake.edu.cn/MicroLens-1M-Dataset",
        "files": [
            "MicroLens-1M_items.tsv",
            "MicroLens-1M_title.csv",
            "MicroLens-1M_pairs.csv",
            "MicroLens-1M_pairs.tsv",
        ],
    },
    "amazon23_vg": {
        "urls": {
            "Video_Games.jsonl.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz",
            "meta_Video_Games.jsonl.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz",
            "Video_Games_5core_rating_only.csv.gz": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz",
        },
    },
    "yelp": {
        "file_name": "yelp_dataset.tar",
        "env_url": "YELP_OPEN_DATASET_URL",
        "source_page": "https://business.yelp.com/data/resources/open-dataset/",
    },
}


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected object manifest: {path}")
    return data


def iter_jobs(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    if args.manifest:
        manifest = read_manifest(args.manifest)
        jobs: list[tuple[str, str, Path]] = []
        for name, spec in manifest.items():
            out_dir = args.output_root / str(name)
            if "urls" in spec:
                for file_name, url in dict(spec["urls"]).items():
                    jobs.append((str(url), str(file_name), out_dir / str(file_name)))
            else:
                base_url = str(spec["base_url"]).rstrip("/")
                files = list(spec["files"])
                for file_name in files:
                    jobs.append((f"{base_url}/{file_name}", str(file_name), out_dir / str(file_name)))
        return jobs

    spec = DATASETS[args.dataset]
    if args.dataset == "yelp":
        url = args.file_url or os.environ.get(str(spec["env_url"]), "")
        if not url:
            source_page = spec["source_page"]
            raise SystemExit(
                "Yelp Open Dataset requires an official archive URL. "
                f"Get it from {source_page} and pass --file_url, set {spec['env_url']}, "
                "or use --local_archive."
            )
        file_name = str(args.files[0] if args.files else spec["file_name"])
        out_dir = args.output_root / args.dataset
        return [(url, file_name, out_dir / file_name)]

    if "urls" in spec:
        selected = set(args.files or [])
        out_dir = args.output_root / args.dataset
        jobs = []
        for file_name, url in dict(spec["urls"]).items():
            if selected and file_name not in selected:
                continue
            jobs.append((str(url), str(file_name), out_dir / str(file_name)))
        return jobs

    base_url = str(args.base_url or spec["base_url"]).rstrip("/")
    files = args.files or spec["files"]
    out_dir = args.output_root / args.dataset
    return [(f"{base_url}/{file_name}", str(file_name), out_dir / str(file_name)) for file_name in files]


def download_one(url: str, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        eprint(f"[skip] {path}")
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    t0 = time.time()
    eprint(f"[download] {url} -> {path}")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)
    tmp.replace(path)
    eprint(f"[ok] {path} bytes={path.stat().st_size} elapsed={time.time() - t0:.1f}s")


def copy_local_archive(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        eprint(f"[skip] {dst}")
        return
    eprint(f"[copy] {src} -> {dst}")
    shutil.copy2(src, dst)
    eprint(f"[ok] {dst} bytes={dst.stat().st_size}")


def extract_archive(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eprint(f"[extract] {path} -> {out_dir}")
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            target_root = out_dir.resolve()
            members = archive.getmembers()
            for member in members:
                target = (out_dir / member.name).resolve()
                if target != target_root and target_root not in target.parents:
                    raise ValueError(f"Unsafe archive member path: {member.name}")
            archive.extractall(out_dir, members=members)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            target_root = out_dir.resolve()
            for member in archive.namelist():
                target = (out_dir / member).resolve()
                if target != target_root and target_root not in target.parents:
                    raise ValueError(f"Unsafe archive member path: {member}")
            archive.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive format: {path}")
    eprint(f"[ok] extracted {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw data files used by the experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="microlens_50k")
    parser.add_argument("--output_root", type=Path, default=Path(os.environ.get("DATA_ROOT", "data/raw")))
    parser.add_argument("--base_url", default="")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--file_url", default="")
    parser.add_argument("--local_archive", type=Path, default=None)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.local_archive:
        if not args.local_archive.exists():
            raise FileNotFoundError(args.local_archive)
        spec = DATASETS[args.dataset]
        file_name = str(args.files[0] if args.files else spec.get("file_name", args.local_archive.name))
        dst = args.output_root / args.dataset / file_name
        copy_local_archive(args.local_archive, dst, args.overwrite)
        if args.extract:
            extract_archive(dst, args.output_root / args.dataset / "extracted")
        return

    downloaded_paths: list[Path] = []
    for url, _, path in iter_jobs(args):
        download_one(url, path, args.overwrite)
        downloaded_paths.append(path)

    if args.extract:
        for path in downloaded_paths:
            extract_archive(path, path.parent / "extracted")


if __name__ == "__main__":
    main()
