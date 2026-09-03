# machine-learning-project

A specification-driven machine-learning project owned by Babar Ali Khan.

## Development flow

1. Approve the relevant file in `specs/`.
2. Write or update tests from its acceptance criteria.
3. Implement the smallest approved scope.
4. Run the test suite and verify the acceptance criteria.
5. Record artifacts, results, assumptions, and limitations.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the first data validation

1. Put the original CSV at `data/raw/dataset.csv`.
2. Update `configs/data.yaml` with the target and column roles.
3. Run:

```bash
python scripts/validate_data.py --config configs/data.yaml
pytest
```

Never modify the source file in `data/raw/`. Learned preprocessing
objects must be fitted on training data only.
