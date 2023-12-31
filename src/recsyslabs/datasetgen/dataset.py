"""A class for dataset objects."""

import pandas as pd


class Dataset(pd.DataFrame):
    """A class for dataset objects."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
    