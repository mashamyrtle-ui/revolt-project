"""
Chronological and Group-Wise Data Splitter with Leakage Verification.
Guarantees early -> train, middle -> validation, late -> test without temporal leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TimeSeriesSplitter:
    """Performs strictly chronological and battery-wise splits with temporal barrier verification."""

    def __init__(
        self,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def chronological_split(
        self,
        df: pd.DataFrame,
        battery_col: str = "battery_id",
        time_or_cycle_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits each battery's timeline chronologically:
          - First train_ratio fraction -> Train
          - Next val_ratio fraction -> Validation
          - Final test_ratio fraction -> Test
        Preserves chronological ordering and prevents future leakage.
        """
        train_dfs: List[pd.DataFrame] = []
        val_dfs: List[pd.DataFrame] = []
        test_dfs: List[pd.DataFrame] = []

        if battery_col in df.columns:
            groups = df.groupby(battery_col, sort=False)
        else:
            groups = [("ALL", df)]

        for _, group in groups:
            # Ensure sorting
            if time_or_cycle_col in group.columns:
                group = group.sort_values(by=time_or_cycle_col).reset_index(drop=True)
            n = len(group)
            n_train = int(np.floor(n * self.train_ratio))
            n_val = int(np.floor(n * self.val_ratio))

            train_part = group.iloc[:n_train].copy()
            val_part = group.iloc[n_train : n_train + n_val].copy()
            test_part = group.iloc[n_train + n_val :].copy()

            train_dfs.append(train_part)
            val_dfs.append(val_part)
            test_dfs.append(test_part)

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        return train_df, val_df, test_df

    def battery_wise_split(
        self,
        df: pd.DataFrame,
        battery_col: str = "battery_id",
        test_batteries: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits by distinct battery/module entities (Leave-Battery-Out)
        to evaluate zero-shot generalization to unseen battery packs.
        """
        if battery_col not in df.columns:
            raise ValueError(f"Battery column '{battery_col}' not found in dataframe.")

        all_bats = list(df[battery_col].unique())
        if len(all_bats) < 2:
            raise ValueError("Battery-wise splitting requires at least 2 distinct batteries.")

        if test_batteries is None:
            # Select final 20% of batteries for test
            n_test = max(1, int(len(all_bats) * self.test_ratio))
            test_batteries = all_bats[-n_test:]

        train_val_df = df[~df[battery_col].isin(test_batteries)].copy()
        test_df = df[df[battery_col].isin(test_batteries)].copy()

        return train_val_df, test_df

    @staticmethod
    def verify_no_temporal_leakage(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        time_or_cycle_col: str = "cycle_number",
        battery_col: str = "battery_id",
    ) -> Dict[str, Any]:
        """
        Formally verifies that for every battery, no future time/cycle data
        from validation or test exists in training data.
        """
        leakage_detected = False
        details: List[str] = []

        if battery_col in train_df.columns:
            bats = train_df[battery_col].unique()
        else:
            bats = ["ALL"]

        for b in bats:
            b_train = train_df[train_df[battery_col] == b] if b != "ALL" else train_df
            b_val = val_df[val_df[battery_col] == b] if b != "ALL" else val_df
            b_test = test_df[test_df[battery_col] == b] if b != "ALL" else test_df

            if time_or_cycle_col in b_train.columns and len(b_train) > 0 and len(b_val) > 0:
                max_train = b_train[time_or_cycle_col].max()
                min_val = b_val[time_or_cycle_col].min()
                if max_train > min_val:
                    leakage_detected = True
                    details.append(f"Leakage in battery {b}: max train ({max_train}) > min val ({min_val})")

            if time_or_cycle_col in b_val.columns and len(b_val) > 0 and len(b_test) > 0:
                max_val = b_val[time_or_cycle_col].max()
                min_test = b_test[time_or_cycle_col].min()
                if max_val > min_test:
                    leakage_detected = True
                    details.append(f"Leakage in battery {b}: max val ({max_val}) > min test ({min_test})")

        return {
            "passed": not leakage_detected,
            "leakage_detected": leakage_detected,
            "violations": details,
        }
