import numpy as np
from typing import Tuple


def single_user_item_interaction(
            n_items: int,
            items_pmf: np.array,
            ratings_domain: Tuple[int, int] = None
) -> int:
    """Returns a user-item interaction.

    Returns:
        int: _description_
    """
    item_id = np.random.choice(n_items, p=items_pmf)
    rating = np.random.randint(ratings_domain[0], ratings_domain[1])
    return item_id, rating


def multi_user_item_interaction(
        n_interactions: int,
        n_items: int,
        items_pmf: np.array,
        ratings_domain: Tuple[int, int] = None
        ) -> np.array:
    """Returns an array of user-item interactions.

    Args:
        n_interactions (int): _description_

    Returns:
        np.array: _description_
    """
    item_ids = np.random.choice(n_items, size=n_interactions, p=items_pmf)
    ratings = np.random.randint(ratings_domain[0], ratings_domain[1], size=n_interactions)
    interactions = np.vstack((item_ids, ratings)).T
    return interactions