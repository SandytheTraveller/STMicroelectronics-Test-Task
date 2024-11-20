import io
import logging
import os
import tempfile

import h5py
import tensorflow as tf
import keras

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
        # Path to save the model
        self.model_file = "model.h5"
        xdata, _ = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._regressor = self.get_regressor()

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
        signature.append("batch_size_" + str(self._hyperparameters.get('batch_size', 10)))
        signature.append("epochs_" + str(self._hyperparameters.get('epochs', 10)))

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
            model = tf.keras.models.load_model(self.model_file)
            self._logger.debug(f"Loaded model from {self.model_file}")
        else:
            # Build a new model if no file exists
            model = self.build_model()

        if model is None:
            raise ValueError("Model could not be initialized correctly.")

        return model

    def build_model(self):
        """
        Build a new neural network model based on the experiment's hyperparameters
        """
        # Prepare input data
        xdata, _ = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # Define input layer's shape by using number of columns in the input data
        input_shape = (xdata.shape[1],)
        # Number of layers
        depth = self._hyperparameters['depth']
        # Number of units in each layer
        width = self._hyperparameters['width']
        # Activation functions to be used on each layer
        activations = self._hyperparameters['activations']

        # Ensure activation list is trimmed to the exact required depth
        if len(activations) < depth:
            activations *= (depth // len(activations)) + 1
        activations = activations[:depth]

        # Dropout rates
        dropouts = self._hyperparameters.get('dropouts', 0.0)

        # Construct the neural network architecture
        # The first layer is an input layer with the shape corresponding to the input data
        layers_list = [keras.layers.Input(shape=input_shape)]

        model = None

        for i in range(depth):
            # Add fully connected Dense layer
            layers_list.append(keras.layers.Dense(width, activation=activations[i % len(activations)]))
            # Add activation function
            layers_list.append(keras.layers.Activation(activations[i]))
            # Apply dropout if specified
            if isinstance(dropouts, list):
                layers_list.append(keras.layers.Dropout(dropouts[i] if i < len(dropouts) else dropouts[-1]))
            else:
                layers_list.append(keras.layers.Dropout(dropouts))

        # Output layer
        layers_list.append(keras.layers.Dense(1))

        # Create the Sequential model
        model = keras.Sequential(layers_list)

        # Compile the model
        model.compile(
            loss=self._hyperparameters['loss'],
            optimizer=self._hyperparameters['optimizer'],
            metrics=[keras.metrics.RootMeanSquaredError()]
        )

        # Set learning rate
        model.optimizer.learning_rate.assign(self._hyperparameters['learning_rate'])

        # Save model path after build
        if model:
            self._logger.info("Model successfully built.")
        return model

    def _train(self):
        """
        Train the neural network model using the provided training data
        """
        self._logger.info("Building and training the model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        xdata = xdata.astype(float)
        ydata = ydata.astype(float)

        self._regressor.fit(xdata, ydata,
                            batch_size=self._hyperparameters['batch_size'],
                            epochs=self._hyperparameters['epochs'],
                            verbose=0)

        self._logger.info("Model trained.")

        if self.model_file:
            self._regressor.save(self.model_file)
            self._logger.info(f"Model saved to {self.model_file}")

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

    def __getstate__(self):
        """Customize the serialization process for the neural network, storing it as bytes."""
        state = self.__dict__.copy()

        # Save the Keras model to a temporary file
        if self._regressor is not None:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
                self._regressor.save(tmp_file.name)  # Save the model to a temporary file
                tmp_file.seek(0)
                with open(tmp_file.name, 'rb') as model_file:
                    state['_regressor_bytes'] = model_file.read()  # Store the model as bytes

            os.remove(tmp_file.name)  # Clean up the temporary file
        else:
            state['_regressor_bytes'] = None

        # Remove the actual Keras model from the state to avoid pickling issues
        state.pop('_regressor', None)
        # self._logger.info('Serialization for NN object')

        return state

    def __setstate__(self, state):
        """Customize the deserialization process for the neural network from the serialized bytes."""
        self.__dict__.update(state)

        # Load the Keras model from the byte content
        model_bytes = state.get('_regressor_bytes')
        if model_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
                tmp_file.write(model_bytes)  # Write the model bytes to a temporary file
                tmp_file.flush()  # Ensure all bytes are written
                tmp_file.seek(0)
                self._regressor = keras.models.load_model(tmp_file.name)  # Load the model from the temporary file

            # Clean up the temporary file
            os.remove(tmp_file.name)
        else:
            self._regressor = None