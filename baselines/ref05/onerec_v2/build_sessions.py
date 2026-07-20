from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def _parse_yelp_datetime(date_str: str) -> float:
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()


def load_setrec_maps(setrec_dir: Path):
    user_map = np.load(setrec_dir / "user_map.npy", allow_pickle=True).item()
    item_map = np.load(setrec_dir / "item_map.npy", allow_pickle=True).item()
    manifest = json.loads((setrec_dir / "manifest.json").read_text(encoding="utf-8"))
    train_cutoff = str(manifest["split_time2"])
    return user_map, item_map, train_cutoff


def load_training_events(
    raw_review_path: Path,
    user_map: dict,
    item_map: dict,
    train_cutoff: str,
    progress_every: int,
):
    user_item_latest = defaultdict(dict)
    stats = {
        "lines": 0,
        "kept": 0,
        "bad": 0,
        "user_miss": 0,
        "item_miss": 0,
        "future": 0,
        "dupe": 0,
    }

    with raw_review_path.open("r", encoding="utf-8") as f:
        for line in f:
            stats["lines"] += 1
            if progress_every > 0 and stats["lines"] % progress_every == 0:
                print(
                    f"[load] lines={stats['lines']} users={len(user_item_latest)} "
                    f"kept={stats['kept']} dupe={stats['dupe']} "
                    f"user_miss={stats['user_miss']} item_miss={stats['item_miss']} future={stats['future']}"
                )

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad"] += 1
                continue

            user_old = obj.get("user_id")
            item_old = obj.get("business_id")
            date_str = obj.get("date")
            if not user_old or not item_old or not date_str:
                stats["bad"] += 1
                continue

            if user_old not in user_map:
                stats["user_miss"] += 1
                continue
            if item_old not in item_map:
                stats["item_miss"] += 1
                continue
            if str(date_str) >= train_cutoff:
                stats["future"] += 1
                continue

            user_id = int(user_map[user_old])
            item_id = int(item_map[item_old])
            ts = str(date_str)

            prev = user_item_latest[user_id].get(item_id)
            if prev is None:
                user_item_latest[user_id][item_id] = ts
                stats["kept"] += 1
            else:
                stats["dupe"] += 1
                if ts > prev:
                    user_item_latest[user_id][item_id] = ts

    events_by_user = {}
    for user_id, item_time in user_item_latest.items():
        pairs = sorted(item_time.items(), key=lambda kv: kv[1])
        events_by_user[user_id] = [(_parse_yelp_datetime(ts), int(item_id)) for item_id, ts in pairs]

    return events_by_user, stats


def build_sessions_for_user(events: list[tuple[float, int]], gap_seconds: float):
    sessions = []
    current = []
    prev_ts = None

    for ts, item_id in events:
        if prev_ts is not None and ts - prev_ts > gap_seconds and current:
            sessions.append(current)
            current = []
        current.append(int(item_id))
        prev_ts = ts

    if current:
        sessions.append(current)
    return sessions


def export_pseudo_sessions(
    events_by_user: dict[int, list[tuple[float, int]]],
    output_path: Path,
    min_session_items: int,
    max_session_items: int | None,
    min_hist: int,
    session_gap_hours: float,
):
    gap_seconds = float(session_gap_hours) * 3600.0
    stats = {
        "users_with_events": 0,
        "raw_sessions": 0,
        "qualified_sessions": 0,
        "dropped_short_sessions": 0,
        "dropped_long_sessions": 0,
        "exported_samples": 0,
        "target_len_min": None,
        "target_len_max": 0,
        "target_len_sum": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for user_id in sorted(events_by_user):
            sessions = build_sessions_for_user(events_by_user[user_id], gap_seconds=gap_seconds)
            if not sessions:
                continue

            stats["users_with_events"] += 1
            stats["raw_sessions"] += len(sessions)
            history = []

            for session_items in sessions:
                session_len = len(session_items)
                if session_len < min_session_items:
                    stats["dropped_short_sessions"] += 1
                    history.extend(session_items)
                    continue
                if max_session_items is not None and session_len > max_session_items:
                    stats["dropped_long_sessions"] += 1
                    history.extend(session_items)
                    continue

                stats["qualified_sessions"] += 1
                if len(history) >= min_hist:
                    row = {
                        "user_id": int(user_id),
                        "history_items": history,
                        "target_items": session_items,
                    }
                    out.write(json.dumps(row) + "\n")
                    stats["exported_samples"] += 1
                    stats["target_len_min"] = session_len if stats["target_len_min"] is None else min(stats["target_len_min"], session_len)
                    stats["target_len_max"] = max(stats["target_len_max"], session_len)
                    stats["target_len_sum"] += session_len

                history.extend(session_items)

    if stats["exported_samples"] > 0:
        stats["target_len_mean"] = stats["target_len_sum"] / stats["exported_samples"]
    else:
        stats["target_len_mean"] = 0.0
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build pseudo session training samples for OneRec from raw Yelp timestamps.")
    parser.add_argument("--raw_review_path", required=True, type=Path)
    parser.add_argument("--setrec_dir", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--min_session_items", type=int, default=5)
    parser.add_argument("--max_session_items", type=int, default=None)
    parser.add_argument("--min_hist", type=int, default=3)
    parser.add_argument("--session_gap_hours", type=float, default=168.0)
    parser.add_argument("--progress_every", type=int, default=1_000_000)
    args = parser.parse_args()

    user_map, item_map, train_cutoff = load_setrec_maps(args.setrec_dir)
    print(
        f"[maps] users={len(user_map)} items={len(item_map)} "
        f"train_cutoff={train_cutoff}"
    )

    events_by_user, load_stats = load_training_events(
        raw_review_path=args.raw_review_path,
        user_map=user_map,
        item_map=item_map,
        train_cutoff=train_cutoff,
        progress_every=args.progress_every,
    )
    print(f"[load] stats={json.dumps(load_stats, ensure_ascii=False)}")
    print(f"[load] users_with_training_events={len(events_by_user)}")

    export_stats = export_pseudo_sessions(
        events_by_user=events_by_user,
        output_path=args.output_path,
        min_session_items=args.min_session_items,
        max_session_items=args.max_session_items,
        min_hist=args.min_hist,
        session_gap_hours=args.session_gap_hours,
    )
    stats_path = args.output_path.with_suffix(args.output_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(export_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[export] output={args.output_path}")
    print(f"[export] stats={json.dumps(export_stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
