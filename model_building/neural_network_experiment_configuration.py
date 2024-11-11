import io
import logging
import os
import tempfile

import tensorflow as tf
import keras
from keras.saving.saving_api import load_model

import model_building.experiment_configuration as ec


class NeuralNetworkExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for a neural network regression model

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., a unique identifier) of this experiment

    get_regressor()
        Initialize the neural network model based on the hyperparameters

    build_model()
        Build a new neural network model based on the experiment's hyperparameters

    _train()
        Performs the actual building of the neural network model

    compute_estimations()
        Compute the predicted values for a given set of data

    save_model()
        Save the model separately in Keras format

    load_model()
        Load the model from the saved file

    __getstate__()
        Customize the serialization process for the neural network

    __setstate__()
        Customize the deserialization process for the neural network
    """

    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix):
        """
        Parameters
        ----------
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dict of str: object
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved

        prefix: list of str
            The prefix to be added to the signature of this experiment configuration
        """
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.NNETWORK
        self.backend = campaign_configuration['General'].get('keras_backend', 'tensorflow')
        self.use_cpu = campaign_configuration['General'].get('keras_use_cpu', False)
        self.model_file = "model.h5"  # Path to save the model


        if self.model_file and os.path.exists(self.model_file):
            self._regressor = tf.keras.models.load_model(self.model_file)
            self._logger.debug(f"Loaded model from {self.model_file}")
        else:
            self.build_model()
            self._regressor = self.get_regressor()


    # This will store the neural network model

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration

        Parameters
        ----------
        prefix: list of str
            The signature of this experiment configuration without considering hyperparameters

        Returns
        -------
        signature: tuple of str
            The signature of the experiment
        """
        signature = prefix.copy()
        signature.append("n_features_" + str(self._hyperparameters['n_features']))
        signature.append("depth_" + str(self._hyperparameters['depth']))
        signature.append("width_" + str(self._hyperparameters['width']))
        signature.append("activations_" + str(self._hyperparameters['activations']))
        signature.append("dropouts_" + str(self._hyperparameters.get('dropouts', 0.0)))
        signature.append("optimizer_" + self._hyperparameters['optimizer'])
        signature.append("learning_rate_" + str(self._hyperparameters['learning_rate']))
        signature.append("loss_" + self._hyperparameters['loss'])
        signature.append("batch_size_" + str(self._hyperparameters['batch_size']))
        signature.append("epochs_" + str(self._hyperparameters['epochs']))

        return signature

    def get_regressor(self):
        """
        Initialize the neural network model based on the hyperparameters
        """
        os.environ['KERAS_BACKEND'] = self.backend
        if self.use_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        # Check if model already exists
        if os.path.exists(self.model_file):
            self._regressor = tf.keras.models.load_model(self.model_file)
            self._logger.debug(f"Loaded model from {self.model_file}")
        else:
            # Build a new model if no file exists
            self.build_model()

        if self._regressor is None:
            raise ValueError("Model could not be initialized correctly.")

        return self._regressor

    def build_model(self):
        """
        Build a new neural network model based on the experiment's hyperparameters
        """
        xdata, _ = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        input_shape = (xdata.shape[1],)
        depth = self._hyperparameters['depth']
        width = self._hyperparameters['width']
        activations = self._hyperparameters['activations']
        dropouts = self._hyperparameters.get('dropouts', 0.0)

        # Construct the neural network architecture
        layers = [keras.layers.Input(shape=input_shape)]
        for i in range(depth):
            layers.append(keras.layers.Dense(width, activation=activations[i % len(activations)]))
            layers.append(keras.layers.Dropout(dropouts if isinstance(dropouts, float) else dropouts[i]))

        # Output layer
        layers.append(keras.layers.Dense(1))

        # Create the Sequential model
        self._regressor = keras.Sequential(layers)

        # Compile the model
        self._regressor.compile(
            loss=self._hyperparameters['loss'],
            optimizer=self._hyperparameters['optimizer'],
            metrics=[keras.metrics.RootMeanSquaredError()]
        )

        # Set learning rate
        self._regressor.optimizer.learning_rate.assign(self._hyperparameters['learning_rate'])

        # Save model path after build
        if self._regressor:
            self._logger.debug("Model successfully built.")

    def _train(self):
        """
        Train the neural network model using the provided training data
        """
        self._logger.debug("Building and training the model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        xdata = xdata.astype(float)
        ydata = ydata.astype(float)

        self._regressor.fit(xdata, ydata, batch_size=self._hyperparameters['batch_size'],
                            epochs=self._hyperparameters['epochs'], verbose=0)
        self._logger.debug("Model trained.")

        if self.model_file:
            self._regressor.save(self.model_file)
            self._logger.debug(f"Model saved to {self.model_file}")

    def compute_estimations(self, rows):
        """
        Compute the predictions for a given set of data points

        Parameters
        ----------
        rows: list of int
            The set of rows to be used for estimation

        Returns
        -------
        predictions: array-like
            The values predicted by the neural network model
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        xdata = xdata.astype(float)
        predictions = self._regressor.predict(xdata, verbose=0)
        return predictions

    def save_model(self, filepath):
        """Save the model separately in Keras format."""
        self._regressor.save(filepath)

    def load_model(self, filepath):
        """Load the model from the saved file."""
        self._regressor = load_model(filepath)

    def __getstate__(self):
        """Customize the serialization process for the neural network."""
        state = self.__dict__.copy()  # Copy the object’s state
        model_path = os.path.join(tempfile.gettempdir(), "model.h5")
        state['model_path'] = model_path
        self.save_model(model_path)  # Save the model to a file
        del state['_regressor']  # Remove the model from the state, as it will be saved separately
        return state

    def __setstate__(self, state):
        """Customize the deserialization process for the neural network."""
        self.__dict__.update(state)  # Restore the object’s state
        self.load_model(self.__dict__['model_path'])  # Load the model from the saved file