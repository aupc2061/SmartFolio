import json
import os
import pickle
import sys
from typing import Dict

import torch
from torch.utils import data

class AllGraphDataSampler(data.Dataset):
    def __init__(self, base_dir, gname_list=None,
                 data_start=None, data_middle=None, data_end=None,
                 train_start_date=None, train_end_date=None,
                 val_start_date=None, val_end_date=None,
                 test_start_date=None, test_end_date=None,
                 idx=False, date=True,
                 mode="train"):
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.manifest_path = os.path.join(self.data_dir, "monthly_manifest.json")
        self.manifest = self._load_manifest()
        self.monthly_index = self._build_monthly_index()
        self._monthly_cache: Dict[str, Dict[str, object]] = {}
        if gname_list is None:
            base_entries = []
            if os.path.isdir(self.data_dir):
                base_entries = [f for f in os.listdir(self.data_dir) if f.endswith(".pkl") and f != "monthly_manifest.json"]
            monthly_dates = sorted(self.monthly_index.keys())
            self.gnames_all = monthly_dates + sorted(base_entries)
        if idx:
            if mode == "train":
                self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
            elif mode == "val":
                self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
            elif mode == "test":
                self.gnames_all = self.gnames_all[self.data_end:]
        if date:
            def _safe_slice(start_date, end_date):
                si = self.date_to_idx(start_date)
                ei = self.date_to_idx(end_date)
                if si is None:
                    si = 0  # fallback to first entry
                if ei is None:
                    ei = len(self.gnames_all) - 1  # fallback to last entry
                if ei < si:
                    ei = si
                return self.gnames_all[si:ei + 1]
            if mode == "train":
                self.gnames_all = _safe_slice(train_start_date, train_end_date)
            elif mode == "val":
                self.gnames_all = _safe_slice(val_start_date, val_end_date)
            elif mode == "test":
                self.gnames_all = _safe_slice(test_start_date, test_end_date)
        self.data_all = self.load_state()

    def __len__(self):
        return len(self.data_all)

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            try:
                name = self.gnames_all[i]
                if name in self.monthly_index:
                    item = self._load_from_monthly(name)
                else:
                    item = pickle.load(open(os.path.join(self.data_dir, name), "rb"))
            except Exception as e:
                print(f"\nWarning: failed to load {self.gnames_all[i]}: {e}. Skipping.")
                continue

            try:
                feats = item.get('features', None)
                labels = item.get('labels', None)
                ts_feats = item.get('ts_features', None)
                valid = True
                if isinstance(feats, torch.Tensor):
                    n = feats.shape[0]
                else:
                    n = feats.shape[0] if feats is not None else 0
                if n is None or n == 0:
                    valid = False
                if isinstance(labels, torch.Tensor):
                    if labels.shape[0] != n:
                        valid = False
                elif labels is None:
                    valid = False
                if isinstance(ts_feats, torch.Tensor):
                    if ts_feats.shape[0] != n:
                        valid = False
                elif ts_feats is None:
                    valid = False
                if not valid:
                    print(f"\nSkipping {self.gnames_all[i]} due to empty or inconsistent shapes (n={n}).")
                    continue
            except Exception as e:
                print(f"\nWarning: sanity check failed for {self.gnames_all[i]}: {e}. Skipping.")
                continue

            data_all.append(item)
        print('{} data loaded!'.format(self.mode))
        return data_all

    def __getitem__(self, idx):
        return self.data_all[idx]

    def date_to_idx(self, date):
        """Return index of the first entry whose date prefix >= given date."""
        if date is None:
            return None
        for i in range(len(self.gnames_all)):
            try:
                d = self.gnames_all[i][:10]
            except Exception:
                continue
            if d >= date:
                return i
        return None

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: manifest {self.manifest_path} is invalid JSON; ignoring.")
        return {}

    def _build_monthly_index(self) -> Dict[str, str]:
        daily_index = self.manifest.get("daily_index", {}) if isinstance(self.manifest, dict) else {}
        # Fallback: if daily_index is missing/empty, build it by scanning monthly shard files on disk
        if isinstance(self.manifest, dict) and not daily_index:
            monthly_dir = os.path.join(self.data_dir, "monthly")
            if os.path.isdir(monthly_dir):
                shard_files = [f for f in os.listdir(monthly_dir) if f.endswith(".pkl")]
                for fname in shard_files:
                    rel_path = os.path.join("monthly", fname)
                    shard_path = os.path.join(self.data_dir, rel_path)
                    try:
                        cache = pickle.load(open(shard_path, "rb"))
                    except Exception:
                        continue
                    dates = cache.get("dates") if isinstance(cache, dict) else None
                    if not dates:
                        continue
                    for dt in dates:
                        daily_index[dt] = rel_path
        resolved = {}
        for date, rel_path in daily_index.items():
            resolved[date] = os.path.join(self.data_dir, rel_path)
        return resolved

    def _load_from_monthly(self, date: str):
        shard_path = self.monthly_index[date]
        cache = self._monthly_cache.get(shard_path)
        if cache is None:
            try:
                cache = pickle.load(open(shard_path, "rb"))
            except Exception as exc:  # pragma: no cover - safeguard
                raise RuntimeError(f"Failed to read monthly shard {shard_path}: {exc}") from exc
            if not isinstance(cache, dict) or "dates" not in cache or "items" not in cache:
                raise RuntimeError(
                    f"Monthly shard {shard_path} is malformed; expected a dict with 'dates' and 'items'."
                )
            # Build lookup for quick access
            cache["_map"] = {d: item for d, item in zip(cache["dates"], cache["items"])}
            self._monthly_cache[shard_path] = cache
        item = cache["_map"].get(date)
        if item is None:
            raise KeyError(f"Date {date} not present in monthly shard {shard_path}.")
        return item
