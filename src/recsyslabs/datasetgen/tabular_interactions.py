"""A class to generate DataFrames of tabular user-item interactions."""

import numpy as np
import pandas as pd
from typing import Tuple
from recsyslabs.datasetgen.dataset import Dataset
from recsyslabs.datasetgen.interactions_sim import (
    single_user_item_interaction,
    multi_user_item_interaction)


class TabularData:

    def __init__(
            self,
            n_users: int,
            n_items: int,
            items_pmf: np.array,
            ratings_domain: Tuple[int, int] = None) -> Dataset:
        self.n_users = n_users
        self.n_items = n_items
        self.items_pmf = items_pmf
        if ratings_domain is not None:
            assert ratings_domain[0] < ratings_domain[1]
            self.ratings_domain = ratings_domain
        else:
            self.ratings_domain = (-1, 1)

    def generate_fixed_num_users_single_interaction(self) -> Dataset:
        """Controls the number of users in the dataset.
        Every user has a single interaction with an item.

        Returns:
            Dataset: _description_
        """

        interactions = np.array([
            single_user_item_interaction(
                n_items=self.n_items,
                items_pmf=self.items_pmf,
                ratings_domain=self.ratings_domain)
            for _ in range(self.n_users)])
        df = pd.DataFrame(
            interactions,
            columns=['item_id', 'rating'])
        df['user_id'] = np.arange(self.n_users)
        return Dataset(df)

    def generate_fixed_num_users_multi_interaction(self, num_interactions: int) -> Dataset:
        """Controls the number of users in the dataset.
        Every user has multiple interactions with items.

        Returns:
            Dataset: _description_
        """

        interactions = np.array([
            multi_user_item_interaction(
                n_interactions=num_interactions,
                n_items=self.n_items,
                items_pmf=self.items_pmf,
                ratings_domain=self.ratings_domain)
            for _ in range(self.n_users)])
        df = pd.DataFrame(
            interactions.reshape(-1, 2),
            columns=['item_id', 'rating'])
        df['user_id'] = np.repeat(np.arange(self.n_users), num_interactions)
        return Dataset(df)
