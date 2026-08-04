from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def patient_folds(
    patient_ids: np.ndarray,
    metastasis: np.ndarray,
    folds: int = 5,
    seeds: tuple[int, ...] = (42, 123, 256, 512, 1024),
) -> list[Fold]:
    if patient_ids.shape != metastasis.shape:
        raise ValueError("patient and label arrays must align")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seeds[0])
    output: list[Fold] = []
    for index, (development, test) in enumerate(splitter.split(patient_ids, metastasis)):
        train, validation = train_test_split(
            development,
            test_size=0.2,
            random_state=seeds[index],
            stratify=metastasis[development],
        )
        output.append(Fold(train=train, validation=validation, test=test))
    return output

