"""A class for dataset objects."""

import pandas as pd


class Dataset(pd.DataFrame):
    """A class for dataset objects."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def to_sparse(self, aggfunc='mean'):
        """Converts the dataset to a sparse DataFrame."""
        return self.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            aggfunc=aggfunc)
