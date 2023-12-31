import pytest
from recsyslabs.datasetgen.interactions_sim import (
    single_user_item_interaction,
    multi_user_item_interaction)
import numpy as np


def test_single_user_item_interaction():
    n_items = 5
    items_pmf = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
    ratings_domain = (1, 5)

    item_id, rating = single_user_item_interaction(n_items, items_pmf, ratings_domain)

    assert isinstance(item_id, int)
    assert isinstance(rating, int)
    assert item_id in range(n_items)
    assert rating in range(*ratings_domain)


def test_multi_user_item_interactions():
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
    assert np.isin(interactions[:, 1], range(*ratings_domain)).all()


if __name__ == '__main__':
    pytest.main([__file__])
