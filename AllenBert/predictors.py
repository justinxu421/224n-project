from typing import List
from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
import json
import numpy as np

#Based on https://github.com/mhagiwara/realworldnlp/blob/e5e36ca90c61e153ac26797ca1798bd6df0a9f12/realworldnlp/predictors.py

class SentenceClassifierPredictor(Predictor):
	def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
		super().__init__(model, dataset_reader)

	def predict(self, tokens: List[str]) -> JsonDict:
		return self.predict_json({"tokens" : tokens})

	def _json_to_instance(self, json_dict: JsonDict) -> Instance:
		tokens = json_dict["tokens"]
		return self._dataset_reader.text_to_instance(tokens)

	def dump_line(self, outputs: JsonDict):
		return json.dumps(outputs) + "\n"

# def predict_batches(self, ds: Iterable[Instance]) ->np.darray:
	# 	 pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
 #        self.model.eval()
 #        pred_generator_tqdm = tqdm(pred_generator,
 #                                   total=self.iterator.get_num_batches(ds))
 #        preds = []
 #        with torch.no_grad():
 #            for batch in pred_generator_tqdm:
 #                batch = nn_util.move_to_device(batch, self.cuda_device)
 #                preds.append(self._extract_data(batch))
 #        return np.concatenate(preds, axis=0)