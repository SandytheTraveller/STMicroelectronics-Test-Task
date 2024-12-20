"""
Main module of the library

Copyright 2019 Marjan Hosseini
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

This module defines the SequenceDataProcessing class which is the only class that has to be accessed to generate regressors
"""
import ast
import configparser as cp
import logging
import os
import pickle
import pprint
import random
import shutil
import sys
import time

import numpy as np

import custom_logger
import data_preparation.column_selection
import data_preparation.data_check
import data_preparation.data_loading
import data_preparation.ernest
import data_preparation.extrapolation
import data_preparation.inversion
import data_preparation.onehot_encoding
import data_preparation.product
import data_preparation.rename_columns
import data_preparation.xgboost_feature_selection

import model_building.model_building


class SequenceDataProcessing:
    """
    Main class which performs the whole design space exploration and builds the regressors

    Its main method is process which performs three main steps:
    1. generate the set of points (i.e., combination of training data, technique, hyper-parameters) to be evaluated
    2. build the regressor corresponding to each point
    3. evaluate the results of all the regressors to identify the best one

    Attributes
    ----------
    _data_preprocessing_list: list of DataPreparation
        The list of steps to be executed for data preparation

    _model_building: ModelBuilding
        The object which performs the actual model building

    _random_generator: RandomGenerator
        The random generator used in the whole application both to generate random numbers and to initialize other random generators
    """

    def __init__(self, configuration_file, debug=False, seed=0, output="output", j=1, generate_plots=False, self_check=True, details=False):
        """
        Constructor of the class

        - Copy the parameters to member variables
        - Initialize the logger
        - Build the data preparation flow adding or not data preparation steps on the basis of the content of the loaded configuration file

        Parameters
        ----------
        configuration_file: str
            The configuration file describing the experimental campaign to be performed

        debug: bool
            True if debug messages should be printed

        seed: integer
            The seed to be used to initialize the random generator engine

        output: str
            The directory where all the outputs will be written; it is created by this library and cannot exist before using this module

        j: integer
            The number of processes to be used in the grid search

        generate_plots: bool
            True if plots have to be used

        self_check: bool
            True if the generated regressor should be tested

        details: bool
            True if the results of the single experiments should be added
        """

        self._data_preprocessing_list = []

        self.random_generator = random.Random(seed)
        self.debug = debug
        self._self_check = self_check

        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        self._logger = custom_logger.get_logger(__name__)

        # Check if the configuration file exists
        if not os.path.exists(configuration_file):
            self._logger.error("%s does not exist", configuration_file)
            sys.exit(-1)

        self.conf = cp.ConfigParser()
        self.conf.optionxform = str
        self.conf.read(configuration_file)
        self.conf['General']['configuration_file'] = configuration_file
        self.conf['General']['output'] = output
        self.conf['General']['seed'] = str(seed)
        self.conf['General']['j'] = str(j)
        self.conf['General']['debug'] = str(debug)
        self.conf['General']['generate_plots'] = str(generate_plots)
        self.conf['General']['details'] = str(details)
        self._campaign_configuration = {}
        self.load_campaign_configuration()

        # Check if output path already exist
        if os.path.exists(output):
            self._logger.error("%s already exists", output)
            sys.exit(1)
        os.mkdir(self._campaign_configuration['General']['output'])
        shutil.copyfile(configuration_file, os.path.join(output, 'configuration_file.ini'))
        self.conf.write(open(os.path.join(output, "enriched_configuration_file.ini"), 'w'))

        # Check that validation method has been specified
        if 'validation' not in self._campaign_configuration['General']:
            self._logger.error("Validation not specified")
            sys.exit(1)

        # Check that if HoldOut is selected, hold_out_ratio is specified
        if self._campaign_configuration['General']['validation'] == "HoldOut" or self._campaign_configuration['General']['hp_selection'] == "HoldOut":
            if "hold_out_ratio" not in self._campaign_configuration['General']:
                self._logger.error("hold_out_ratio not set")
                sys.exit(1)

        # Check that if Extrapolation is selected, extrapolation_columns is specified
        if self._campaign_configuration['General']['validation'] == "Extrapolation":
            if "extrapolation_columns" not in self._campaign_configuration['General']:
                self._logger.error("extrapolation_columns not set")
                sys.exit(1)

        # Check that if XGBoost is used for feature selection tolerance is specified
        if 'FeatureSelection' in self._campaign_configuration and self._campaign_configuration['FeatureSelection']['method'] == "XGBoost":
            if "XGBoost_tolerance" not in self._campaign_configuration['FeatureSelection']:
                self._logger.error("XGBoost tolerance not set")
                sys.exit(1)

        # Check that if ernest is used, normalization, product, column_selection, and inversion are disabled
        if 'ernest' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['ernest']:
            if 'use_columns' in self._campaign_configuration['DataPreparation'] or "skip_columns" in self._campaign_configuration['DataPreparation']:
                logging.error("use_columns and skip_columns cannot be used with ernest")
                sys.exit(1)
            if 'inverse' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['inverse']:
                logging.error("inverse cannot be used with ernest")
                sys.exit(1)
            if 'product_max_degree' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['product_max_degree']:
                logging.error("product cannot be used with ernest")
                sys.exit(1)
            if 'normalization' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['normalization']:
                logging.error("normalization cannot be used with ernest")
                sys.exit(1)

        # Adding read on input to data preprocessing step
        self._data_preprocessing_list.append(data_preparation.data_loading.DataLoading(self._campaign_configuration))

        # Adding column renaming if required
        if 'rename_columns' in self._campaign_configuration['DataPreparation']:
            self._data_preprocessing_list.append(data_preparation.rename_columns.RenameColumns(self._campaign_configuration))

        # Adding column selection if required
        if 'use_columns' in self._campaign_configuration['DataPreparation'] or "skip_columns" in self._campaign_configuration['DataPreparation']:
            self._data_preprocessing_list.append(data_preparation.column_selection.ColumnSelection(self._campaign_configuration))

        # Transform categorical features in onehot encoding
        self._data_preprocessing_list.append(data_preparation.onehot_encoding.OnehotEncoding(self._campaign_configuration))

        # Split according to extrapolation values if required
        if self._campaign_configuration['General']['validation'] == "Extrapolation":
            self._data_preprocessing_list.append(data_preparation.extrapolation.Extrapolation(self._campaign_configuration))

        # Adding inverted features if required
        if 'inverse' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['inverse']:
            self._data_preprocessing_list.append(data_preparation.inversion.Inversion(self._campaign_configuration))

        # Adding product features if required
        if 'product_max_degree' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['product_max_degree']:
            self._data_preprocessing_list.append(data_preparation.product.Product(self._campaign_configuration))

        # Create ernest features if required
        if 'ernest' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['ernest']:
            self._data_preprocessing_list.append(data_preparation.ernest.Ernest(self._campaign_configuration))

        # Adding data check
        self._data_preprocessing_list.append(data_preparation.data_check.DataCheck(self._campaign_configuration))

        self._model_building = model_building.model_building.ModelBuilding(self.random_generator.random())

    def load_campaign_configuration(self):
        """
        Load the campaign configuration from config file named _campaign_configuration.ini and put all the information into a dictionary
        """

        self._campaign_configuration = {}

        for section in self.conf.sections():
            self._campaign_configuration[section] = {}
            for item in self.conf.items(section):
                try:
                    value = ast.literal_eval(item[1])
                    if isinstance(value, list):
                        value = tuple(value)
                    self._campaign_configuration[section][item[0]] = value
                except (ValueError, SyntaxError):
                    self._campaign_configuration[section][item[0]] = item[1]

        self._logger.debug("Parameters configuration is:")
        self._logger.debug("-->")
        self._logger.debug(pprint.pformat(self._campaign_configuration, width=1))
        self._logger.debug("<--")

    def process(self):
        """
        the main code which actually performs the design space exploration of models

        Only a single regressor is returned: the best model of the best technique.

        These are the main steps:
        - data are preprocessed and dumped to preprocessed.csv
        - design space exploration of the required models (i.e., the models specified in the configuration file) is performed
        - eventually, best model is used to predict all the data
        - best model is returned

        Returns
        -------
        Regressor
            The regressor containing the overall best model and the preprocessing steps used to preprocess the input data
        """

        os.environ["OMP_NUM_THREADS"] = "1"

        start = time.time()

        self._logger.info("-->Starting experimental campaign")
        # performs reading data, drops irrelevant columns
        # initial_df = self.preliminary_data_processing.process(self._campaign_configuration)
        # logging.info("Loaded and cleaned data")

        # performs inverting of the columns and adds combinatorial terms to the df
        # ext_df = self.data_preprocessing.process(initial_df, self._campaign_configuration)
        # logging.info("Preprocessed data")

        data_processing = None

        for data_preprocessing_step in self._data_preprocessing_list:
            self._logger.info("-->Executing %s", data_preprocessing_step.get_name())
            data_processing = data_preprocessing_step.process(data_processing)
            self._logger.debug("Current data frame is:\n%s", str(data_processing))
            self._logger.info("<--")

        data_processing.data.to_csv(os.path.join(self.conf['General']['output'], "preprocessed.csv"))

        regressor = self._model_building.process(self._campaign_configuration, data_processing, int(self.conf['General']['j']))

        end = time.time()
        execution_time = str(end - start)
        self._logger.info("<--Execution Time : %s", execution_time)

        if self._self_check:
            self._logger.info("-->Performing self check")
            check_data_loading = data_preparation.data_loading.DataLoading(self._campaign_configuration)
            check_data = None
            check_data = check_data_loading.process(check_data)
            check_data = check_data.data

            # Getting the values of the y
            real_y = check_data[self._campaign_configuration['General']['y']].values
            check_data = check_data.drop(columns=[self._campaign_configuration['General']['y']])
            for technique in self._campaign_configuration['General']['techniques']:
                pickle_file_name = os.path.join(self._campaign_configuration['General']['output'],
                                                technique + ".pickle")
                pickle_file = open(pickle_file_name, "rb")
                regressor = pickle.load(pickle_file)
                pickle_file.close()
                # My code
                # Added try-except block to catch any potential errors
                try:
                    self._logger.info("Starting prediction with regressor %s", regressor)
                    predicted_y = regressor.predict(check_data).flatten()
                    # self._logger.info("Prediction completed. Predicted values: %s", predicted_y)
                except Exception as e:
                    self._logger.error("Error during prediction: %s", str(e))
                    raise
                difference = real_y - predicted_y
                mape = np.mean(np.abs(np.divide(difference, real_y)))
                self._logger.info("---MAPE of %s: %s", technique, str(mape))

            self._logger.info("<--Performed self check")

        return regressor
