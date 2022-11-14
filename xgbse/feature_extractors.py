from typing import Any, Dict, Optional, Sequence, Tuple

import xgboost as xgb

from xgbse.converters import convert_data_to_xgb_format, convert_y
from xgbse.non_parametric import get_time_bins


class FeatureExtractor:
    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = 1,
    ):
        """
        Args:
            xgb_params (Dict, None): Parameters for XGBoost model.
                If not passed, will use XGBoost default parameters and set objective as `survival:aft`.
                Check <https://xgboost.readthedocs.io/en/latest/parameter.html> for options.

            lr_params (Dict, None): Parameters for Logistic Regression models, if not passed will use LogisticRegression defaults.
                Check <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html> for options.
            n_jobs (Int): Number of CPU cores used to fit logistic regressions via joblib, if -1 will use all available cores. Defaults to 1.
        """
        if not xgb_params:
            xgb_params = {}
        xgb_params = check_xgboost_parameters(xgb_params)

        self.xgb_params = xgb_params
        self.n_jobs = n_jobs
        self.persist_train = False
        self.feature_importances_ = None

    def fit(
        self,
        X,
        y,
        time_bins: Optional[Sequence] = None,
        validation_data: Optional[Tuple[Any, Any]] = None,
        num_boost_round: int = 10,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: int = 0,
    ):
        """
        Transform feature space by fitting a XGBoost model and returning its leaf indices.
        Leaves are transformed and considered as dummy variables to fit multiple logistic
        regression models to each evaluated time bin.

        Args:
            X ([pd.DataFrame, np.array]): Features to be used while fitting XGBoost model

            y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
                and time of event or time of censoring as second field.

            time_bins (np.array): Specified time windows to use when making survival predictions

            validation_data (Tuple): Validation data in the format of a list of tuples [(X, y)]
                if user desires to use early stopping

            num_boost_round (Int): Number of boosting iterations, defaults to 10

            early_stopping_rounds (Int): Activates early stopping.
                Validation metric needs to improve at least once
                in every **early_stopping_rounds** round(s) to continue training.
                See xgboost.train documentation.

            persist_train (Bool): Whether or not to persist training data to use explainability
                through prototypes

            index_id (pd.Index): User defined index if intended to use explainability
                through prototypes


            verbose_eval ([Bool, Int]): Level of verbosity. See xgboost.train documentation.

        Returns:
            XGBSEDebiasedBCE: Trained XGBSEDebiasedBCE instance
        """

        E_train, T_train = convert_y(y)
        if not time_bins:
            time_bins = get_time_bins(T_train, E_train)
        self.time_bins = time_bins

        # converting data to xgb format
        dtrain = convert_data_to_xgb_format(X, y, self.xgb_params["objective"])

        # converting validation data to xgb format
        evals = ()
        if validation_data:
            X_val, y_val = validation_data
            dvalid = convert_data_to_xgb_format(
                X_val, y_val, self.xgb_params["objective"]
            )
            evals = [(dvalid, "validation")]

        # training XGB
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            verbose_eval=verbose_eval,
        )
        self.feature_importances_ = self.bst.get_score()

    def predict_leaves(self, X):
        """
        Predict leaf indices of XGBoost model.

        Args:
            X (pd.DataFrame, np.array): Features to be used while predicting leaf indices

        Returns:
            np.array: Leaf indices of XGBoost model
        """
        if not hasattr(self, "bst"):
            raise ValueError("XGBoost model not fitted yet.")

        dmatrix = xgb.DMatrix(X)
        return self.bst.predict(
            dmatrix, pred_leaf=True, iteration_range=(0, self.bst.best_iteration + 1)
        )

    def predict_hazard(self, X):
        if not hasattr(self, "bst"):
            raise ValueError("XGBoost model not fitted yet.")

        return self.bst.predict(
            xgb.DMatrix(X), iteration_range=(0, self.bst.best_iteration + 1)
        )


def check_xgboost_parameters(xgb_params: Dict[str, Any]) -> Dict[str, Any]:
    """Check if XGBoost objective parameter is valid.

    Args:
        xgb_params (Dict): Parameters for XGBoost model.

    Returns:
        xgb_params (Dict): Parameters for XGBoost model.

    Raises:
        ValueError: If XGBoost parameters are not valid for survival analysis.
    """
    if "objective" not in xgb_params:
        xgb_params["objective"] = "survival:aft"
    if xgb_params["objective"] not in ("survival:aft", "survival:cox"):
        raise ValueError(
            "XGBoost objective must be either 'survival:aft' or 'survival:cox'"
        )

    return xgb_params
