"""Testing dataset generation classes and methods."""

import numpy as np
import pandas as pd
import pytest
from recsyslabs.datasetgen.fixed_users import FixedUsers
from recsyslabs.datasetgen.dataset import Dataset


def test_generate_controlled_users_single_interaction():
    n_users = 10
    n_items = 5
    items_pmf = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
    ratings_domain = (1, 5)

    tabular_data = FixedUsers(
        n_users=n_users,
        n_items=n_items,
        item_exposure_bias_pmf=items_pmf,
        ratings_domain=ratings_domain)
    dataset = tabular_data.generate_single_interaction()

    assert isinstance(dataset, Dataset)
    assert len(dataset) <= n_users * n_items
    assert set(dataset.columns) == {'user_id', 'item_id', 'rating'}
    assert dataset['user_id'].nunique() == n_users
    assert dataset['item_id'].nunique() <= n_items
    assert dataset['rating'].between(ratings_domain[0], ratings_domain[1] + 1).all()


def test_generate_controlled_users_multi_interaction():
    n_users = 10
    n_items = 5
    items_pmf = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
    ratings_domain = (-1, 1)
    num_interactions = 10

    tabular_data = FixedUsers(n_users, n_items, items_pmf, ratings_domain)
    dataset = tabular_data.generate_multi_interaction(num_interactions=num_interactions)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == n_users * num_interactions
    assert set(dataset.columns) == {'user_id', 'item_id', 'rating'}
    assert dataset['user_id'].nunique() == n_users
    assert dataset['item_id'].nunique() <= n_items
    assert dataset['rating'].between(*ratings_domain).all()


if __name__ == '__main__':
    pytest.main([__file__])
