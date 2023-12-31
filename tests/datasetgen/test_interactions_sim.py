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

    assert 0 <= item_id < n_items
    assert ratings_domain[0] <= rating <= ratings_domain[1]
    assert item_id in range(n_items)
    assert rating in range(ratings_domain[0], ratings_domain[1] + 1)


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
    assert np.isin(interactions[:, 1], range(ratings_domain[0], ratings_domain[1] + 1)).all()



def test_multi_user_item_interactions_alphabets():
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


if __name__ == '__main__':
    pytest.main([__file__])
