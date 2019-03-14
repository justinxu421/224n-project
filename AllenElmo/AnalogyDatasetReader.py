from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.vocabulary import Vocabulary
from typing import Iterator, List, Dict

#Modified based on https://github.com/allenai/allennlp and tutorial on RealWorldNLP


class AnalogyDatasetReader(DatasetReader):
    """
    DatasetReader for Analogies
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], label: str = None) -> Instance:
        analogies_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": analogies_field}
        if label:
            label_field = LabelField(label, label_namespace="labels")
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                analogyList = line.lower().strip().split(' | ')[1:]
                analogySource = analogyList[0:3]
                analogyTarget = analogyList[3]
                analogies = []
                for phrase in analogySource:
                    analogies += phrase.split()
                tag = analogyTarget.split()
                yield self.text_to_instance([Token(word) for word in analogies], analogyTarget)