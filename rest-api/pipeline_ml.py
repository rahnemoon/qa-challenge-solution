"""include classes that initiate the ML model and provide tools to keep their
metric and update them.
"""
import json
import pandas as pd
from river import linear_model
from river import metrics
from river import evaluate
from river import optim
from river import preprocessing
from river import facto
from river import compose
from river import neural_net as nn


class ClassificationModel(object):

    """Classification method is used for binary classification
    the class has two mode in normal mode uses the river one record at time
    for train and scheduling and another mode pass the whole mini batch of data
    to pipeline of model and scaler. LogisticRegression and FFMClassifier are
    respectively models used for training and in both case the scaler is
    StandardScaler.
    """

    def __init__(self, is_mini_batch_mode=False):
        """initiate the class and crate the ML model base on the mode plus define
        the metric objects used for model

        Parameters
        ----------
        is_mini_batch_mode : bool, optional
            optional variable set the mode of training but by default is false
            meaning object is in single mode training
        """
        self.__metric_report = self.__init_metric_report()
        self.__is_batch = is_mini_batch_mode
        if self.__is_batch:
            self.__model = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression(optimizer=optim.SGD(0.1)),
            )
        else:
            self.__model = compose.Pipeline(
                preprocessing.StandardScaler(),
                facto.FFMClassifier(
                    n_factors=10,
                    intercept=0.5,
                    seed=42,
                    weight_optimizer=optim.SGD(0.1),
                    latent_optimizer=optim.SGD(0.1),
                ),
            )

    def __init_metric_report(self):
        """method used to initialize a dictionary containing the metrics as
        confusion_matrix, f1beta, accuracy, rocauc, precision, mcc  for
        classification model

        Returns
        -------
        TYPE
            a dictionary with the metric's name as key and their object as value
            but confusion_matrix is nested dictionary
        """
        return {
            "confusion_matrix": {
                "false_negatives": 0,
                "false_positives": 0,
                "true_positives": 0,
                "true_negatives": 0,
            },
            "f1beta": metrics.FBeta(beta=1.0),
            "accuracy": metrics.Accuracy(),
            "rocauc": metrics.ROCAUC(),
            "precision": metrics.Precision(),
            "mcc": metrics.MCC(),
        }

    def __refresh_confusion_matrix(self, true_data, pred_data):
        """Update the nested confusion matrix in the metric dictionary

        Parameters
        ----------
        true_data : obj
            the true data
        pred_data : obj
            the prediction record from the model
        """
        if true_data == True and pred_data == False:
            self.__metric_report["confusion_matrix"]["false_negatives"] += 1
        if true_data == False and pred_data == True:
            self.__metric_report["confusion_matrix"]["false_positives"] += 1
        if true_data == True and pred_data == True:
            self.__metric_report["confusion_matrix"]["true_positives"] += 1
        if true_data == False and pred_data == False:
            self.__metric_report["confusion_matrix"]["true_negatives"] += 1

    def __update_metric_report(self, true_data, pred_data):
        """update the dictionary containing the metrics

        Parameters
        ----------
        true_data : obj
            the true data
        pred_data : obj
            the predicted data
        """
        for key, value in self.__metric_report.items():
            if key == "confusion_matrix":
                self.__refresh_confusion_matrix(true_data, pred_data)
            else:
                value.update(true_data, pred_data)

    def __batch_update_metric_report(self, true_df, pred_df):
        """perform the refresh on the metrics in batch model as in this mode the
        result of prediction is list or serial

        Parameters
        ----------
        true_df : list or serial
            the true data
        pred_df : list or serial
            the predicted data
        """
        for true, pred in zip(true_df, pred_df):
            self.__update_metric_report(true, pred)

    def __attach_pred_to_data(self, true_data, pred_df):
        """Concat the result of prediction with data use for prediction and
        true data for the batch mode

        Parameters
        ----------
        true_data : list or serial
            the true data
        pred_df : list or serial
            the true data

        Returns
        -------
        TYPE
            return list of dictionary which prediction attached to each record
        """
        for true, pred in zip(true_data, pred_df):
            true.update({"pred_promoted": pred})
        return true_data

    def train(self, mini_batch):
        """get mini batch data and perform the training process for each one and
        training model is FFMClassifier

        Parameters
        ----------
        mini_batch : list
            list of dictionaries which are true data

        Returns
        -------
        list
            return a list of dictionaries that prediction is attach to each record
        """
        for data_item in mini_batch:
            features = {
                "competence": data_item.get("competence"),
                "network_ability": data_item.get("network_ability"),
            }
            pred = self.__model.predict_one(features)
            """ attachment of prediction to true data in single mode happens here """
            data_item.update({"pred_promoted": pred})
            self.__model.learn_one(features, data_item.get("promoted"))
            self.__update_metric_report(data_item.get("promoted"), pred)
        return mini_batch

    def train_batch(self, mini_batch):
        """feed the true data to model for training and prediction as whole instead of
        preforming the training process for each record

        Parameters
        ----------
        mini_batch : list
            list of dictionaries as true data

        Returns
        -------
        list
            list of dictionaries which prediction is attach to record
        """
        learning_df = pd.DataFrame(
            mini_batch, columns=["id", "competence", "network_ability", "promoted"]
        )
        learning_df.set_index("id", inplace=True)
        pred = self.__model.predict_many(learning_df[["competence", "network_ability"]])
        self.__model.learn_many(
            learning_df[["competence", "network_ability"]], learning_df["promoted"]
        )
        self.__batch_update_metric_report(learning_df["promoted"], pred)
        return self.__attach_pred_to_data(mini_batch, pred)

    def serialize_report(self):
        """serialize or jsonfy the metric report to can be stored or passed around

        Returns
        -------
        JSON
            jsonfied object of metrics report
        """
        data = {
            "confusion_matrix": {
                "false_negatives": self.__metric_report.get("confusion_matrix").get(
                    "false_negatives"
                ),
                "false_positives": self.__metric_report.get("confusion_matrix").get(
                    "false_positives"
                ),
                "true_negatives": self.__metric_report.get("confusion_matrix").get(
                    "true_negatives"
                ),
                "true_positives": self.__metric_report.get("confusion_matrix").get(
                    "true_positives"
                ),
            },
            "f1beta": self.__metric_report.get("f1beta").get(),
            "accuracy": self.__metric_report.get("accuracy").get(),
            "rocauc": self.__metric_report.get("rocauc").get(),
            "precision": self.__metric_report.get("precision").get(),
            "mcc": self.__metric_report.get("mcc").get(),
        }
        return data

    def get_model(self):
        """return the ML model

        Returns
        -------
        obj
            return the model
        """
        return self.__model


class RegressionModel(object):

    """initiation of regression ML model and metrics related and PARegressor is
    the ML model used. This class only provides single mode of training.
    """

    def __init__(self):
        """initiate the dictionary containing the metrics of model and model itself"""
        self.__metric_report = self.__init_metric_report()
        self.__model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.PARegressor(C=0.01, mode=2, eps=0.1, learn_intercept=False),
        )

    def __init_metric_report(self):
        """create the dictionary that each key is the metric and value is object
        related to the metric. MSE, RMSE, and MEA are included in the report

        Returns
        -------
        dict
            dictionary of metric objects
        """
        return {
            "MSE": metrics.MSE(),
            "RMSE": metrics.RMSE(),
            "MEA": metrics.MAE(),
        }

    def __update_metric_report(self, true_data, pred_data):
        """refresh the metrics of model in the report

        Parameters
        ----------
        true_data : obj
            the true data
        pred_data : obj
            the predicted data
        """
        for key, value in self.__metric_report.items():
            value.update(true_data, pred_data)

    def train(self, mini_batch):
        """train the model by getting the mini batch and executing the training
        process for each record

        Parameters
        ----------
        mini_batch : list
            list of dictionaries which are true record

        Returns
        -------
        list
            list of dictionaries that predicted data is attached
        """
        for data_item in mini_batch:
            features = {
                "competence": data_item.get("competence"),
                "promoted": data_item.get("promoted"),
            }
            pred = self.__model.predict_one(features)
            """ predicted data is attach to the true data"""
            data_item.update({"pred_network_ability": pred})
            self.__model.learn_one(features, data_item.get("network_ability"))
            self.__update_metric_report(data_item.get("network_ability"), pred)
        return mini_batch

    def serialize_report(self):
        """Serialize or jsonify the report for storing or passing

        Returns
        -------
        JSON
            return a dictionary that is in JSON style
        """
        data = {
            "MSE": self.__metric_report.get("MSE").get(),
            "RMSE": self.__metric_report.get("RMSE").get(),
            "MEA": self.__metric_report.get("MEA").get(),
        }
        return data

    def get_model(self):
        """return the model object

        Returns
        -------
        obj
            return the regressor obj
        """
        return self.__model


if __name__ == "__main__":
    regression()
