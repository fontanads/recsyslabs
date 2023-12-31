import numpy as np
from typing import Tuple
from recsyslabs.datasetgen.dataset import Dataset


class TabularData:

    def __init__(
            self,
            n_users: int,
            n_items: int,
            item_exposure_bias_pmf: np.array = None,
            ratings_alphabet: np.array = None,
            ratings_domain: Tuple[int, int] = None,
            user_rating_bias_pmf: np.array = None):

        self.n_users = n_users
        self.n_items = n_items

        if item_exposure_bias_pmf is None:
            self.items_pmf = np.ones(n_items) / n_items
        else:
            self.items_pmf = item_exposure_bias_pmf

        self.rating_pmf = user_rating_bias_pmf
        self.ratings_alphabet = ratings_alphabet

        if ratings_domain is not None:
            assert ratings_domain[0] < ratings_domain[1]
        self.ratings_domain = ratings_domain

        if user_rating_bias_pmf is not None:
            if ratings_alphabet is not None:
                assert len(user_rating_bias_pmf) == len(ratings_alphabet)
            elif ratings_domain is not None:
                assert len(user_rating_bias_pmf) == ratings_domain[1] - ratings_domain[0] + 1
            else:
                assert len(user_rating_bias_pmf) == 3