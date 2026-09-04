install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

validate-data:
	python scripts/validate_data.py --config configs/data.yaml

eda:
	python scripts/run_eda.py --data-config configs/data.yaml --eda-config configs/eda.yaml

split-data:
	python scripts/split_data.py --data-config configs/data.yaml --training-config configs/training.yaml
