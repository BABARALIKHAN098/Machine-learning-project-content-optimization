from __future__ import annotations

import pandas as pd


def add_cutoff_safe_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ratios derived only from previous-period measurements."""
    result = dataframe.copy()
    impressions = result.get("impressions_prev_30d")
    clicks = result.get("clicks_prev_30d")
    sessions = result.get("sessions_prev_30d")
    if impressions is not None and clicks is not None:
        denominator = impressions.where(impressions > 0)
        result["previous_ctr"] = (clicks / denominator).fillna(0.0)
    if clicks is not None and sessions is not None:
        denominator = clicks.where(clicks > 0)
        result["previous_sessions_per_click"] = (sessions / denominator).fillna(0.0)
    return result
