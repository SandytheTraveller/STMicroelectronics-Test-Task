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
import multiprocessing
import os
import pickle
import random
import tqdm

import custom_logger
import data_preparation.normalization
import model_building.experiment_configuration as ec
import model_building.generators_factory as gf
import regressor
import results as re


def process_wrapper(experiment_configuration):
    """Wrapper for multiprocessing."""
    try:
        experiment_configuration.train()
        return experiment_configuration
    except Exception as e:
        experiment_configuration._logger.error(f"Error during training: {str(e)}")
        return None


class ModelBuilding:
    """
    Entry point of the model building phase, i.e., where all the regressions are actually performed

    The process method do the following steps:
        - Create the generators through the factory
        - Build the model for each ExperimentConfiguration
        - Evaluate the MAPE on different sets of each ExperimentConfiguration
        - Identify the best regressor of each technique
        - Retrain the best regressors with the whole dataset
        - Dump the best regressors in pickle format

    Attributes
    ----------
    random_generator : Random
        The internal random generator

    _logger: Logger
        The logger used by this class

    Methods
    ------
    process()
        Generates the set of expriment configurations to be evaluated
    """

    def __init__(self, seed):
        """
        Parameters
        ----------
        seed: integer
            The seed to be used for the internal random generator
        """
        self._random_generator = random.Random(seed)
        self._logger = custom_logger.get_logger(__name__)

    def process(self, campaign_configuration, regression_inputs, processes_number):
        """
        Perform the actual regression

        Parameters
        ----------
        campaign_configuration: dict of str: dict of str: tr
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem

        processes_number: integer
            The number of processes which can be used

        Return
        ------
        Regressor
            The best regressor of the best technique
        """
        self._logger.info("-->Generate generators")
        factory = gf.GeneratorsFactory(campaign_configuration, self._random_generator.random())
        top_generator = factory.build()
        self._logger.info("<--Generated generators")

        self._logger.info("-->Generate experiments")
        expconfs = top_generator.generate_experiment_configurations([], regression_inputs)
        self._logger.info("<--Generated experiments")

        assert expconfs, "Experiment configurations cannot be empty."

        if processes_number == 1:
            self._logger.info("-->Run experiments sequentially")
            for exp in tqdm.tqdm(expconfs, dynamic_ncols=True):
                exp.train()
            self._logger.info("<--Completed sequential experiments")
        else:
            self._logger.info("-->Run experiments in parallel")
            with multiprocessing.Pool(processes_number) as pool:
                expconfs = list(tqdm.tqdm(pool.imap(process_wrapper, expconfs), total=len(expconfs)))
            expconfs = [exp for exp in expconfs if exp is not None]  # Filter out failed experiments
            self._logger.info("<--Completed parallel experiments")

        self._logger.info("-->Collecting results")
        results = re.Results(campaign_configuration, expconfs)
        results.collect_data()
        self._logger.info("<--Collected results")

        # Debug log MAPE results
        for metric, mapes in results.raw_results.items():
            for experiment_configuration, mape in mapes.items():
                self._logger.debug("%s of %s is %f", metric, experiment_configuration, mape)

        best_confs, best_technique = results.get_bests()
        best_regressors = {}
        self._logger.info("-->Building the final regressors")

        # Create a shadow copy
        all_data = regression_inputs.copy()

        # Set all sets equal to the whole input set
        all_data.inputs_split["training"] = all_data.inputs_split["all"]
        all_data.inputs_split["validation"] = all_data.inputs_split["all"]
        all_data.inputs_split["hp_selection"] = all_data.inputs_split["all"]

        # Handle normalization
        if 'normalization' in campaign_configuration['DataPreparation'] and campaign_configuration['DataPreparation'][
            'normalization']:
            for column in all_data.scaled_columns:
                original_column_name = "original_" + column
                if original_column_name in all_data.data.columns:
                    all_data.data[column] = all_data.data[original_column_name]
                    all_data.data = all_data.data.drop(columns=[original_column_name])

            all_data.scaled_columns = []
            self._logger.debug("Denormalized inputs are:%s\n", str(all_data))

            normalizer = data_preparation.normalization.Normalization(campaign_configuration)
            all_data = normalizer.process(all_data)

        # Train best configurations and build final regressors
        for technique in best_confs:
            best_conf = best_confs[technique]
            all_data.x_columns = best_conf.get_x_columns()
            best_conf.set_training_data(all_data)

            # Train and evaluate
            best_conf.train()
            best_conf.evaluate()
            self._logger.info("Validation MAPE on full dataset for %s: %s", technique,
                              str(best_conf.mapes["validation"]))

            # Build the regressor and handle pickle file creation
            best_regressors[technique] = regressor.Regressor(campaign_configuration, best_conf.get_regressor(),
                                                             best_conf.get_x_columns(), all_data.scalers)

            pickle_file_name = os.path.join(campaign_configuration['General']['output'],
                                            ec.enum_to_configuration_label[technique] + ".pickle")
            with open(pickle_file_name, "wb") as pickle_file:
                try:
                    pickle.dump(best_regressors[technique], pickle_file)
                except pickle.PicklingError as e:
                    self._logger.error(f"Failed to pickle the regressor for {technique}: {str(e)}")

        self._logger.info("<--Built the final regressors")

        # Return the regressor
        return best_regressors[best_technique]

