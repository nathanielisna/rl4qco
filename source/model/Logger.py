#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy.py

This script contains the environment information logging class.

__author__ = ""
__email__ = ""
"""

import csv
import os
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback


class OptimizationLogger:
    def __init__(self, path, fieldnames=None):
        self.path = path
        self.fieldnames = fieldnames or [
            "timestamp",
            "monitor_r",
            "monitor_l",
            "monitor_t",
            "baseline_depth",
            "baseline_gates",
            "optimized_depth",
            "optimized_gates",
            "total_reward",
        ]

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, entry: dict):
        row = {k: entry.get(k, "") for k in self.fieldnames}
        row["timestamp"] = entry.get(
            "timestamp", datetime.utcnow().isoformat())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class OptimizationLoggingCallback(BaseCallback):
    def __init__(self, csv_logger, verbose=0):
        super().__init__(verbose)
        self.csv_logger = csv_logger

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            if isinstance(info, dict):
                ep = info.get("episode")
            else:
                ep = None

            if ep:
                opt_ep = ep.get("opt_episode", {}) if isinstance(
                    ep, dict) else {}
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "monitor_r": ep.get("r", ""),
                    "monitor_l": ep.get("l", ""),
                    "monitor_t": ep.get("t", ""),
                    "baseline_depth": opt_ep.get(
                        "baseline_depth", ep.get("baseline_depth", "")
                    ),
                    "baseline_gates": opt_ep.get(
                        "baseline_gates", ep.get("baseline_gates", "")
                    ),
                    "optimized_depth": opt_ep.get(
                        "optimized_depth", ep.get("optimized_depth", "")
                    ),
                    "optimized_gates": opt_ep.get(
                        "optimized_gates", ep.get("optimized_gates", "")
                    ),
                    "total_reward": opt_ep.get(
                        "total_reward", ep.get("total_reward", "")
                    ),
                }
                try:
                    self.csv_logger.log(row)
                except Exception:
                    pass
        return True
