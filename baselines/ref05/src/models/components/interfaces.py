from typing import List, Union

import torch


class ModelOutput:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def list_of_row_format(self):


        raise NotImplementedError

    def _convert_to_list(self, prediction: Union[torch.Tensor, List]) -> List:


        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().tolist()

        return prediction


class SharedKeyAcrossPredictionsOutput(ModelOutput):


    def __init__(
        self,
        key,
        predictions,
        key_name: str = "idx",
        prediction_name: str = "prediction",
    ):
        self.key = key
        self.predictions = predictions
        self.key_name = key_name
        self.prediction_name = prediction_name

    @property
    def list_of_row_format(self):
        return [
            {self.key_name: self.key, self.prediction_name: pred}
            for pred in self._convert_to_list(self.predictions)
        ]


class OneKeyPerPredictionOutput(ModelOutput):


    def __init__(
        self,
        keys,
        predictions,
        key_name: str = "idx",
        prediction_name: str = "prediction",
    ):
        self.keys = keys
        self.predictions = predictions
        self.key_name = key_name
        self.prediction_name = prediction_name

    @property
    def list_of_row_format(self):
        return [
            {self.key_name: key, self.prediction_name: pred}
            for key, pred in zip(
                self._convert_to_list(self.keys),
                self._convert_to_list(self.predictions),
            )
        ]
