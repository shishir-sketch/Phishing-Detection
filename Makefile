.PHONY: install test lint demo clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

demo:
	python -m phishing_detector.train --synthetic

clean:
	rm -rf results/* models/* .pytest_cache build dist *.egg-info
	touch results/.gitkeep models/.gitkeep
