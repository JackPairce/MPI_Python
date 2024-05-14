from typing import Dict

from numpy import array, dtype, ndarray, random
from sklearn.preprocessing import LabelEncoder


class WordsEncoder:
    def __init__(self, words: ndarray[str, dtype]) -> None:
        self.words = words
        self.dict = self._DictEncodedWords()

    def EncodingWords(self) -> ndarray[int, dtype]:
        return array([self.dict[w] for w in self.words])

    def _DictEncodedWords(self) -> Dict[str, int]:
        classes = list(set(self.words))
        encoded_classes = LabelEncoder().fit_transform(classes)
        doc = {c: e for c, e in zip(classes, encoded_classes)}  # type: ignore
        return doc


def random_arrays_choice(array: ndarray, N: int) -> list[ndarray]:
    return [array[i] for i in random.choice(array.shape[0], N, replace=False)]
