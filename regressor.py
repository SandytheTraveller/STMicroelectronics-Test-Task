"""
Copyright 2019 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

import custom_logger
import data_preparation.column_selection
import data_preparation.ernest
import data_preparation.inversion
import data_preparation.product
import data_preparation.rename_columns
import regression_inputs


class Regressor:
    """
    The main type of object returned by the library. It includes preprocessing step plus the actual regressor

    Attributes
    ----------
    _campaign_configuration: dict of str : dict of str : str
        The set of options specified during the generation of this regressor

    _regressor
        The actual object performing the regression

    _x_columns
        The columns used by the regressor

    _scalers
        The scalers used for the normalization of each column

    _logger
        The internal logger

    Methods
    -------
    predict()
        Predict the target column

    get_regressor()
        Return the regressor associated with this experiment configuration
    """
    def __init__(self, campaign_configuration, configuration, regressor, x_columns, scalers):
        """
        Parameters
        regressor
            The wrapped regressor
        """
        assert regressor
        self._campaign_configuration = campaign_configuration
        self.configuration = configuration
        self._regressor = regressor
        self._x_columns = x_columns
        self._scalers = scalers
        self._logger = custom_logger.get_logger(__name__)

    def predict(self, inputs):
        """
        Perform the prediction on a set of input data

        Parameters
        ----------
        inputs: pandas.DataFrame
            The input on which prediction has to be applied
        """
        inputs_split = {}
        column_names = inputs.columns.values.tolist()
        data = regression_inputs.RegressionInputs(inputs, inputs_split, column_names,
                                                  self._campaign_configuration['General']['y'])
        self._logger.info("Created input regression")

        # Adding column renaming if required
        if 'rename_columns' in self._campaign_configuration['DataPreparation']:
            rename_columns_step = data_preparation.rename_columns.RenameColumns(self._campaign_configuration)
            data = rename_columns_step.process(data)
            self._logger.debug("Performed column renaming")

        # Adding column selection if required
        if 'use_columns' in self._campaign_configuration['DataPreparation'] or "skip_columns" in \
                self._campaign_configuration['DataPreparation']:
            column_selection_step = data_preparation.column_selection.ColumnSelection(self._campaign_configuration)
            data = column_selection_step.process(data)
            self._logger.debug("Performed column selection")

        # One-hot encoding step
        onehot_encoding_step = data_preparation.onehot_encoding.OnehotEncoding(self._campaign_configuration)
        data = onehot_encoding_step.process(data)

        # Compute inverse if required
        if 'inverse' in self._campaign_configuration['DataPreparation'] and \
                self._campaign_configuration['DataPreparation']['inverse']:
            inversion_step = data_preparation.inversion.Inversion(self._campaign_configuration)
            data = inversion_step.process(data)
            self._logger.debug("Performed inversion")

        # Compute product features if required
        if 'product_max_degree' in self._campaign_configuration['DataPreparation'] and \
                self._campaign_configuration['DataPreparation']['product_max_degree']:
            inversion_step = data_preparation.product.Product(self._campaign_configuration)
            data = inversion_step.process(data)
            self._logger.debug("Performed product")

        # Create ernest features if required
        if 'ernest' in self._campaign_configuration['DataPreparation'] and \
                self._campaign_configuration['DataPreparation']['ernest']:
            ernest_step = data_preparation.ernest.Ernest(self._campaign_configuration)
            data = ernest_step.process(data)
            self._logger.debug("Performed ernest feature computation")

        raw_data = data.data

        y_column = self._campaign_configuration['General']['y']

        try:
            # Apply normalization
            for column in self._scalers:
                if column == y_column:
                    continue
                self._logger.debug("---Applying scaler to %s", column)

                # Ensure data is converted to float32
                data_to_be_normalized = raw_data[column].astype(np.float32).to_numpy()
                data_to_be_normalized = np.reshape(data_to_be_normalized, (-1, 1))
                normalized_data = self._scalers[column].transform(data_to_be_normalized)
                raw_data[column] = normalized_data

            self._logger.debug("Performed normalization")

            # Filter columns for prediction
            raw_data = raw_data[self._x_columns].astype(np.float32)  # Ensuring float32 conversion
            self._logger.debug("Performed columns filtering: %s", str(self._x_columns))

            # Perform the prediction
            y_value = self._regressor.predict(raw_data)

        except (ValueError, KeyError) as error:
            self._logger.error("Input raw data:\n%s", str(raw_data))
            raise error

        # If scaling was applied to the target column (y_column), reverse it
        if y_column in self._scalers:
            y_scaler = self._scalers[y_column]
            # Ensure reshaping of y_value if necessary
            y_value = y_scaler.inverse_transform(y_value.reshape(-1, 1))

        return y_value

    def get_regressor(self):
        """
        Return the internal regressor"
        """
        return self._regressor
