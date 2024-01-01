import pytest
import logging
from recsyslabs.datasetgen.interactions_sim import (
    single_user_item_interaction,
    multi_user_item_interaction,
    items_with_mininum_interactions,
    items_with_mininum_interactions_per_rating_symbol
    )
import numpy as np


logger = logging.getLogger(__name__)


def test_single_user_item_interaction(np_seed):
    logger.debug(f'seed = {np_seed}')
    n_items = 5
    items_pmf = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
    ratings_domain = (1, 5)

    item_id, rating = single_user_item_interaction(n_items, items_pmf, ratings_domain)

    assert 0 <= item_id < n_items
    assert ratings_domain[0] <= rating <= ratings_domain[1]
    assert item_id in range(n_items)
    assert rating in range(ratings_domain[0], ratings_domain[1] + 1)


def test_multi_user_item_interactions(np_seed):
    logger.debug(f'seed = {np_seed}')
    n_items = 5
    items_pmf = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
    ratings_domain = (-1, 1)
    n_interactions = 10

    interactions = multi_user_item_interaction(
        n_interactions=n_interactions,
        n_items=n_items,
        items_pmf=items_pmf,
        ratings_domain=ratings_domain)

    assert isinstance(interactions, np.ndarray)
    assert interactions.shape == (n_interactions, 2)
    assert np.isin(interactions[:, 0], range(n_items)).all()
    assert np.isin(interactions[:, 1], range(ratings_domain[0], ratings_domain[1] + 1)).all()


def test_multi_user_item_interactions_alphabets(np_seed):
    logger.debug(f'seed = {np_seed}')
    n_items = 4
    items_pmf = np.array([0.2, 0.3, 0.4, 0.1])
    ratings_alphabet= (1, 5, 10)
    n_interactions = 2

    interactions = multi_user_item_interaction(
        n_interactions=n_interactions,
        n_items=n_items,
        items_pmf=items_pmf,
        ratings_alphabet=ratings_alphabet)

    assert isinstance(interactions, np.ndarray)
    assert interactions.shape == (n_interactions, 2)
    assert np.isin(interactions[:, 0], range(n_items)).all()
    assert np.isin(interactions[:, 1], ratings_alphabet).all()


def test_items_with_mininum_interactions(np_seed):
    logger.debug(f'seed = {np_seed}')
    n_items = 5
    items_pmf = np.array([90e-2, 8e-2, 1e-2, 9e-3, 1e-3])
    ratings_domain = (-1, 1)
    n_interactions = 10
    n_users = 10
    min_interactions = 15
    items_interactions= np.zeros(n_items, dtype=int)

    interactions, num_samples = items_with_mininum_interactions(
        min_interactions=min_interactions,
        n_interactions=n_interactions,
        n_items=n_items,
        n_users=n_users,
        items_pmf=items_pmf,
        ratings_domain=ratings_domain)

    unique_items, item_counts = np.unique(interactions[:, 0], return_counts=True)
    items_interactions[unique_items] += item_counts

    assert isinstance(interactions, np.ndarray)
    assert interactions.shape == (n_interactions * n_users * num_samples, 2)
    assert np.isin(interactions[:, 0], range(n_items)).all()
    assert np.isin(interactions[:, 1], range(ratings_domain[0], ratings_domain[1] + 1)).all()
    assert np.min(items_interactions) >= min_interactions


def test_items_with_mininum_interactions_per_rating_symbol(np_seed):
    logger.debug(f'seed = {np_seed}')
    n_items = 5
    items_pmf = np.array([90e-2, 5e-2, 3e-2, 1e-2, 1e-2])
    ratings_alphabet = (-10, 0, 10)
    user_bias_rating_pmf = np.array([0.1, 0.8, 0.1])
    n_interactions = 10
    n_users = 100
    min_interactions = 5
    n_symbols = len(ratings_alphabet)
    
    item_symbol_counts= np.zeros(shape=(n_items, n_symbols), dtype=int)

    interactions, num_samples = items_with_mininum_interactions_per_rating_symbol(
        min_interactions=min_interactions,
        n_interactions=n_interactions,
        n_items=n_items,
        n_users=n_users,
        items_pmf=items_pmf,
        ratings_alphabet=ratings_alphabet)

    for symbol_id, symbol in enumerate(ratings_alphabet):
        row_mask = interactions[:, 1] == symbol
        unique_items, item_counts = np.unique(
            interactions[row_mask, 0],
            return_counts=True)
        item_symbol_counts[unique_items, symbol_id] += item_counts

    assert isinstance(interactions, np.ndarray)
    assert interactions.shape == (n_interactions * n_users * num_samples, 2)
    assert np.isin(interactions[:, 0], range(n_items)).all()
    assert np.isin(interactions[:, 1], ratings_alphabet).all()
    assert np.min(item_symbol_counts) >= min_interactions


if __name__ == '__main__':
    pytest.main([__file__])
