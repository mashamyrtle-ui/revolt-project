"""
Unified Feature Engineering Pipeline for ABIA.
Applies electrical, thermal, usage, temporal, interaction, and physics feature modules
strictly grouped per battery in chronological order.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
from src.abia.features.electrical import compute_electrical_features
from src.abia.features.thermal import compute_thermal_features
from src.abia.features.usage import compute_usage_features
from src.abia.features.temporal import compute_temporal_features
from src.abia.features.interactions import compute_interaction_features
from src.abia.features.physics import compute_physics_features


class FeaturePipeline:
    """Unified causal feature generation pipeline."""

    def __init__(
        self,
        battery_col: str = "battery_id",
        nominal_capacity_ah: float = 50.0,
    ):
        self.battery_col = battery_col
        self.nominal_capacity_ah = nominal_capacity_ah

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms preprocessed battery dataframe into enriched feature dataframe."""
        if self.battery_col in df.columns:
            groups = df.groupby(self.battery_col, sort=False)
        else:
            groups = [("DEFAULT", df)]

        enriched_groups: List[pd.DataFrame] = []

        for _, group in groups:
            # Base data
            base = group.copy()

            # 1. Electrical
            elec = compute_electrical_features(base)

            # 2. Thermal
            therm = compute_thermal_features(base)

            # 3. Usage
            usage = compute_usage_features(base, rated_capacity_ah=self.nominal_capacity_ah)

            # 4. Temporal
            temp = compute_temporal_features(base)

            # 5. Interactions
            inter = compute_interaction_features(base)

            # 6. Physics
            phys = compute_physics_features(base, nominal_capacity_ah=self.nominal_capacity_ah)

            # Combine all along columns
            combined = pd.concat([base, elec, therm, usage, temp, inter, phys], axis=1)
            # Remove any potential duplicate columns
            combined = combined.loc[:, ~combined.columns.duplicated()]
            enriched_groups.append(combined)

        out_df = pd.concat(enriched_groups, ignore_index=True)
        return out_df
