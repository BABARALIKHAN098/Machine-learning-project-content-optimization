import numpy as np
import pandas as pd

from machine_learning_project.features.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_and_unseen_values():
    train = pd.DataFrame(
        {"age": [20.0, np.nan, 40.0], "city": ["A", "B", None]}
    )
    inference = pd.DataFrame({"age": [30.0], "city": ["UNSEEN"]})
    transformer = build_preprocessor(["age"], ["city"])
    transformer.fit(train)
    output = transformer.transform(inference)
    assert output.shape[0] == 1
