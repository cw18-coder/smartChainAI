import pandas as pd
import numpy as np
import os


def mstl_season_selector(min_ts_len:int, freq:str) -> list[int]:
    """Selects appropriate season lengths for MSTL decomposition based on the frequency and minimum time series length.
    The maximum season length should not be more than half the length of the time series.

    Args:
        min_ts_len (int): Minimum length of the time series.
        freq (str): Frequency of the time series. Options are 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'H' (half-yearly).

    Returns:
        list[int]: List of season lengths to be used for MSTL decomposition.
    """
    if freq == 'D':
        possible_seasons = [7, 14, 30, 90, 180]  # weekly, bi-weekly, monthly, quarterly, half-yearly
    elif freq == 'W':
        possible_seasons = [4, 8, 13, 26]  # monthly, bi-monthly, quarterly, half-yearly
    elif freq == 'M':
        possible_seasons = [3, 6]  # quarterly, half-yearly
    elif freq == 'Q':
        possible_seasons = [4]  # yearly
    elif freq == 'H':
        possible_seasons = [2]  # yearly
    else:
        raise ValueError("Unsupported frequency. Use 'D', 'W', 'M', 'Q', or 'H'.")

    max_season_length = min_ts_len // 2
    selected_seasons = [s for s in possible_seasons if s <= max_season_length]

    if not selected_seasons:
        raise ValueError("Time series length is too short for any seasonality based on the given frequency.")

    return selected_seasons