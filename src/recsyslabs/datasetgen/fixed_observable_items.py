
import logging
import numpy as np
import pandas as pd
from recsyslabs.datasetgen.tabular_data import TabularData
from recsyslabs.datasetgen.dataset import Dataset
from recsyslabs.datasetgen.interactions_sim import (
    items_with_mininum_interactions,
    items_with_mininum_interactions_per_rating_symbol)

logger = logging.getLogger(__name__)


class FixedObservableItems(TabularData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_minimum_interactions(
            self,
            min_interactions=1,
            num_interactions=10,
            max_loops=10000) -> Dataset:
        """'Controls the number of items in the dataset.

        Args:
            min_interactions (int, optional): _description_. Defaults to 1.
        """

        min_samples_to_observe_least_frequent_item = np.ceil(1 / min(self.items_pmf)).astype(int)

        if self.n_users * num_interactions * max_loops < min_samples_to_observe_least_frequent_item * min_interactions:
            logger.warning(f'The number of samples is not enough to observe {min_interactions} times the least frequent item')
            logger.warning(f'Minimum samples to observe at least once the least frequent item is {min_samples_to_observe_least_frequent_item}')
            logger.warning(f'Thus for {min_interactions} observations, max loops should be at least {min_samples_to_observe_least_frequent_item * min_interactions}')

        interactions, num_samples = items_with_mininum_interactions(
                min_interactions=min_interactions,
                n_interactions=num_interactions,
                n_items=self.n_items,
                n_users=self.n_users,
                items_pmf=self.items_pmf,
                ratings_alphabet=self.ratings_alphabet,
                ratings_domain=self.ratings_domain,
                ratings_pmf=self.rating_pmf,
                max_loops=max_loops)
        df = pd.DataFrame(
            interactions.reshape(-1, 2),
            columns=['item_id', 'rating'])
        df['user_id'] = np.repeat(np.repeat(np.arange(self.n_users), num_interactions), num_samples)
        return Dataset(df)

    def generate_minimum_interactions_per_rating_symbol(
            self,
            min_interactions=1,
            num_interactions=10,
            max_loops=10000) -> Dataset:
        """'Controls the number of items in the dataset.

        Args:
            min_interactions (int, optional): _description_. Defaults to 1.
        """

        min_samples_to_observe_least_frequent_rating = np.ceil((1 / min(self.items_pmf) * (1 / min(self.rating_pmf)))).astype(int)
        total_samples = self.n_users * num_interactions * max_loops
        ideal_total_samples = min_samples_to_observe_least_frequent_rating * min_interactions
        if total_samples < ideal_total_samples:
            logger.warning('The number of samples is not enough to observe %s times the least frequent item', min_interactions)
            logger.warning('Minimum samples to observe at least once the least frequent item is %s', min_samples_to_observe_least_frequent_rating)
            logger.warning('Thus for %s observations, max loops should be at least %s', min_interactions, ideal_total_samples)

        interactions, num_samples = items_with_mininum_interactions_per_rating_symbol(
                min_interactions=min_interactions,
                n_interactions=num_interactions,
                n_items=self.n_items,
                n_users=self.n_users,
                items_pmf=self.items_pmf,
                ratings_alphabet=self.ratings_alphabet,
                ratings_domain=self.ratings_domain,
                ratings_pmf=self.rating_pmf,
                max_loops=max_loops)

        df = pd.DataFrame(
            interactions.reshape(-1, 2),
            columns=['item_id', 'rating'])
        df['user_id'] = np.repeat(np.repeat(np.arange(self.n_users), num_interactions), num_samples)
        return Dataset(df)
