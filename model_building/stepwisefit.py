"""
Copyright 2019 Eugenio Gianniti

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

import numpy
import pandas
from scipy import linalg, stats


class Stepwise:
    """
    Implementation of the stepwise technique

    Attributes
    ----------
    _p_to_add: float
        The minimum significance to add a feature

    _p_to_discard: float
        The maximum significance to not remove a feature

    _add_intercept: book
        True if a constant term has to be added to the linear model

    _max_iterations: integer
        The maximum number of iterations of the algorithm of adding features

    coef
        The coefficients of the linear model

    intercept: float
        The value of the intercept term of the linear model

    k_feature_names: list of str
        The list of selected features
    """
    def __init__(self, p_enter=0.05, p_remove=0.1, fit_intercept=True, max_iter=100):
        """
        p_enter: float
            The minimum significance to add a feature

        p_remove: float
            The maximum significance to not remove a feature

        fit_intercept: bool
            True if the intercept term has to be added to the linear model

        max_iter: integer
            The maximum number of iterations
        """
        self._p_to_add = p_enter
        self._p_to_discard = p_remove
        self._add_intercept = fit_intercept
        self._max_iterations = max_iter
        self.coef_ = None
        self.intercept_ = None
        self.k_feature_names_ = None

    def fit(self, x_data, y_data):
        """
        https://sourceforge.net/p/octave/statistics/ci/default/tree/inst/stepwisefit.m

        x_data
            The input of the regression

        y_data
            The expected results
        """
        regressors = len(x_data.columns)
        go_on = True
        residuals = y_data
        counter = 0
        self.k_feature_names_ = []
        n_residuals = len(residuals)

        while go_on and counter < self._max_iterations:
            counter += 1
            added = False
            dropped = False

            if len(self.k_feature_names_) < regressors:
                not_in_use = [c for c in x_data.columns if c not in self.k_feature_names_]
                possible_additions = x_data.loc[:, not_in_use]
                rho = possible_additions.join(residuals).corr()
                most_correlated = rho.iloc[-1, :-1].abs().idxmax(axis="columns")
                current_columns = self.k_feature_names_ + [most_correlated]
                current_data = x_data.loc[:, current_columns]
                b_new, b_int_new, r_new = self._regress(current_data, y_data)
                z_new = numpy.abs(b_new[-1] / (b_int_new[-1, 1] - b_new[-1]))

                if z_new > 1:  # which means you accept
                    added = True
                    b_coefficients = b_new
                    b_int = b_int_new
                    residuals = pandas.Series(r_new, name="r")
                    self.k_feature_names_.append(most_correlated)

            if self.k_feature_names_:
                variables = len(self.k_feature_names_)
                dof = n_residuals - variables - 1 if self._add_intercept else n_residuals - variables
                t_ratio = stats.t.ppf(1 - self._p_to_discard / 2, dof) / stats.t.ppf(1 - self._p_to_add / 2, dof)

                if self._add_intercept:
                    z_coefficients = numpy.abs(b_coefficients[1:] / (b_int[1:, 1] - b_coefficients[1:]))
                else:
                    z_coefficients = numpy.abs(b_coefficients / (b_int[:, 1] - b_coefficients))

                z_min = numpy.min(z_coefficients, axis=None)
                idx = numpy.argmin(z_coefficients, axis=None)

                if z_min < t_ratio:
                    dropped = True
                    del self.k_feature_names_[idx]
                    current_data = x_data.loc[:, self.k_feature_names_]
                    b_coefficients, b_int, r_coefficients = self._regress(current_data, y_data)
                    residuals = pandas.Series(r_coefficients, name="r")

            go_on = added or dropped

        if self._add_intercept:
            self.intercept_ = b_coefficients[0]
            self.coef_ = b_coefficients[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = b_coefficients

    def _regress(self, xmatrix_df, y_df):
        """
        https://sourceforge.net/p/octave/statistics/ci/default/tree/inst/regress.m
        """
        xmatrix = xmatrix_df.to_numpy()
        y_data = y_df.to_numpy()
        n_coefficients = y_data.size

        if self._add_intercept:
            xmatrix = numpy.c_[numpy.ones(n_coefficients), xmatrix]

        q_data, r_data = linalg.qr(xmatrix, mode="economic")
        beta = linalg.solve(r_data, q_data.T.dot(y_data))
        residuals = y_data - xmatrix.dot(beta)

        _, p_data = xmatrix.shape
        dof = n_coefficients - p_data
        sse = residuals.T.dot(residuals)
        mse = sse / dof
        t_alpha_2 = stats.t.ppf(self._p_to_add / 2, dof)
        c_values = numpy.diag(linalg.inv(r_data.T.dot(r_data)))
        # delta is negative, because alpha is small and t_alpha_2 negative
        delta = t_alpha_2 * numpy.sqrt(mse * c_values)
        beta_interval = numpy.c_[beta + delta, beta - delta]

        return beta, beta_interval, residuals

    def predict(self, xmatrix_df):
        """."""
        xmatrix = xmatrix_df.loc[:, self.k_feature_names_]
        n_coefficients = len(xmatrix.index)

        if self._add_intercept:
            x_matrix = numpy.c_[numpy.ones(n_coefficients), xmatrix]
            b_coefficients = numpy.r_[self.intercept_, self.coef_]
        else:
            b_coefficients = self.coef_

        return numpy.dot(x_matrix, b_coefficients)
