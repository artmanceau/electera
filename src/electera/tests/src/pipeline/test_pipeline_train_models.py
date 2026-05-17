import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from electera.pipeline.pipeline_train_models import ElectionModelTrainer


class TestTrainTrivialModels:
    def setup_method(self):
        """Setup method run before each test"""
        self.trainer = ElectionModelTrainer()

        # Mock the data attributes
        self.trainer.y_test = pd.Series([0.3, 0.4, 0.5, 0.6], name="pvote1")
        self.trainer.y_prev = pd.Series([0.25, 0.35, 0.45, 0.55], name="pvoteprevious1")

        # Mock the evaluate method to capture calls
        self.trainer.evaluate = MagicMock()

    def test_train_trivial_models_with_previous_data(self):
        """Test trivial models training when previous data exists"""

        # Run the function
        self.trainer.train_trivial_models()

        # Check that evaluate was called twice (previous + mean)
        assert self.trainer.evaluate.call_count == 2

        # Check the calls
        calls = self.trainer.evaluate.call_args_list

        # First call should be for trivial_previous
        assert calls[0][0][1] == "trivial_previous"  # second argument is model_name

        # Second call should be for trivial_mean
        second_call_args = calls[1][0]
        assert calls[1][0][1] == "trivial_mean"

        # Check that mean prediction has correct values
        expected_mean = self.trainer.y_test.mean()
        mean_predictions = second_call_args[0]
        assert np.allclose(mean_predictions.values, expected_mean)

    def test_train_trivial_models_no_previous_data(self):
        """Test trivial models training when no previous data exists"""

        # Set y_prev to None
        self.trainer.y_prev = None

        # Run the function
        self.trainer.train_trivial_models()

        # Check that evaluate was called only once (just mean, no previous)
        assert self.trainer.evaluate.call_count == 1

        # Check the call was for trivial_mean
        call_args = self.trainer.evaluate.call_args_list[0]
        assert call_args[0][1] == "trivial_mean"

    def test_train_trivial_models_with_nan_in_previous(self):
        """Test trivial models when previous data contains NaN values"""

        # Add NaN to previous data
        self.trainer.y_prev = pd.Series([0.25, np.nan, 0.45, np.nan])

        # Run the function
        self.trainer.train_trivial_models()

        # Check that evaluate was called twice
        assert self.trainer.evaluate.call_count == 2

        # Get the previous predictions (first call)
        previous_predictions = self.trainer.evaluate.call_args_list[0][0][0]

        # Check that NaN values were filled with y_test mean
        expected_mean = self.trainer.y_test.mean()
        assert previous_predictions.iloc[1] == expected_mean  # Second value was NaN
        assert previous_predictions.iloc[3] == expected_mean  # Fourth value was NaN
        assert previous_predictions.iloc[0] == 0.25  # First value unchanged
        assert previous_predictions.iloc[2] == 0.45  # Third value unchanged
