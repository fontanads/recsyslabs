import pytest
import os

# Get the path of the current file
current_file_path = os.path.dirname(os.path.abspath(__file__))

# Define pytest parameters
pytest_params = [
    "-v",
    "-s",
    "--cov=.",
    "--cov-report=html",
    "--log-cli-level=DEBUG"]

# Run all tests with pytest parameters
pytest.main(pytest_params)
