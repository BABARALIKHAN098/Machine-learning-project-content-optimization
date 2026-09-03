install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

validate-data:
	python scripts/validate_data.py --config configs/data.yaml
