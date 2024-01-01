
import pytest
import numpy as np

# Global variable to store the current seed
current_seed = 12345


@pytest.fixture
def np_seed():
    global current_seed
    np.random.seed(current_seed)
    yield current_seed
    current_seed += 1
