import numpy as np
from typing import Tuple, Union


def single_user_item_interaction(
            n_items: int,
            items_pmf: Union[list, tuple, np.ndarray],
            ratings_domain: Tuple[int, int] = None,
            ratings_alphabet: Union[list, tuple, np.ndarray] = None,
            ratings_pmf: Union[list, tuple, np.ndarray] = None
) -> int:
    """Returns a user-item interaction.

    Returns:
        int: _description_
    """

    item_id, rating = multi_user_item_interaction(
        n_interactions=1,
        n_items=n_items,
        items_pmf=items_pmf,
        ratings_domain=ratings_domain,
        ratings_pmf=ratings_pmf)[0]

    return item_id, rating


def multi_user_item_interaction(
        n_interactions: int,
        n_items: int,
        items_pmf: Union[list, tuple, np.ndarray],
        ratings_alphabet: Union[list, tuple, np.ndarray] = None,
        ratings_domain: Tuple[int, int] = None,
        ratings_pmf: Union[list, tuple, np.ndarray] = None
        ) -> np.array:
    """Returns an array of user-item interactions.

    Args:
        n_interactions (int): _description_

    Returns:
        np.array: _description_
    """

    item_ids = np.random.choice(n_items, size=n_interactions, p=items_pmf)

    if not ratings_alphabet:
        if ratings_domain is None:
            ratings_domain = (-1, 1)
        domain = np.arange(ratings_domain[0], ratings_domain[1] + 1)
    else:
        domain = ratings_alphabet

    if ratings_pmf:
        assert len(ratings_pmf) == len(domain)

    ratings = np.random.choice(
        domain,
        size=n_interactions,
        p=ratings_pmf)
    
    interactions = np.vstack((item_ids, ratings)).T
    return interactions


def items_with_mininum_interactions(
        min_interactions: int,
        n_interactions: int,
        n_items: int,
        n_users: int,
        items_pmf: Union[list, tuple, np.ndarray],
        ratings_alphabet: Union[list, tuple, np.ndarray] = None,
        ratings_domain: Tuple[int, int] = None,
        ratings_pmf: Union[list, tuple, np.ndarray] = None,
        max_loops: int = 10000
        ) -> np.array:
    """Runs a numerical simulation to sample from the exposure bias pmf of the items.
    The loop stops when the number of interactions of the item with the least number of interactions is equal to min_interactions.
    """

    num_loops = 0
    all_interactions = []
    items_interactions= np.zeros(n_items, dtype=int)
    while np.min(items_interactions) < min_interactions and num_loops < max_loops:
        interactions = np.array([
            multi_user_item_interaction(
                n_interactions=n_interactions,
                n_items=n_items,
                items_pmf=items_pmf,
                ratings_alphabet=ratings_alphabet,
                ratings_domain=ratings_domain,
                ratings_pmf=ratings_pmf
                )
            for _ in range(n_users)])

        interactions = interactions.reshape(-1, 2)
        all_interactions.append(interactions)
        unique_items, item_counts = np.unique(interactions[:, 0], return_counts=True)
        items_interactions[unique_items] += item_counts
        num_loops += 1
    return np.vstack(all_interactions), num_loops


def items_with_mininum_interactions_per_rating_symbol(
        min_interactions: int,
        n_interactions: int,
        n_items: int,
        n_users: int,
        items_pmf: Union[list, tuple, np.ndarray],
        ratings_alphabet: Union[list, tuple, np.ndarray] = None,
        ratings_domain: Tuple[int, int] = None,
        ratings_pmf: Union[list, tuple, np.ndarray] = None,
        max_loops: int = 10000
        ) -> np.array:
    """Runs a numerical simulation to sample from the exposure bias pmf of the items.
    The loop stops when every item has a minimum number of interactions per rating symbol.
    """

    if ratings_alphabet is None and ratings_domain is None:
        raise ValueError('Either ratings_alphabet or ratings_domain should be provided')

    num_loops = 0
    all_interactions = []

    if ratings_alphabet is None:
        n_symbols = ratings_domain[1] - ratings_domain[0] + 1
        alphabet = np.arange(ratings_domain[0], ratings_domain[1] + 1)
    else:
        n_symbols = len(ratings_alphabet)
        alphabet = np.array(ratings_alphabet).copy()

    item_symbol_counts = np.zeros(shape=(n_items, n_symbols), dtype=int)

    while np.min(item_symbol_counts) < min_interactions and num_loops < max_loops:
        interactions = np.array([
            multi_user_item_interaction(
                n_interactions=n_interactions,
                n_items=n_items,
                items_pmf=items_pmf,
                ratings_alphabet=ratings_alphabet,
                ratings_domain=ratings_domain,
                ratings_pmf=ratings_pmf
                )
            for _ in range(n_users)])

        interactions = interactions.reshape(-1, 2)
        all_interactions.append(interactions)

        assert np.isin(interactions[:, 1], alphabet).all(), 'The ratings symbols should be in the alphabet'

        for symbol_id, symbol in enumerate(alphabet):
            row_mask = interactions[:, 1] == symbol
            unique_items, item_counts = np.unique(
                interactions[row_mask, 0],
                return_counts=True)
            item_symbol_counts[unique_items, symbol_id] += item_counts
        num_loops += 1
    return np.vstack(all_interactions), num_loops