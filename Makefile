MAIN = main.py
SRC = src

install: uv sync

run: install
		uv run python3 $(MAIN)

debug: install
		uv run python3 -m pdb $(MAIN)

lint: install
		uv run flake8 $(SRC)
		uv run mypy $(SRC) --warn-return-any \
        --warn-unused-ignores \
        --ignore-missing-imports \
        --disallow-untyped-defs \
        --check-untyped-defs

lint-strict: install
		uv run flake8 $(SRC)
		uv run mypy --explicit-package-bases $(SRC) --strict

clean:
		rm -rf .mypy_cache
		find . -name "*.pyc" -exec rm -f {} \;
		find . -type d -name "__pycache__" -exec rm -r {} \+

fclean:
		fclean: clean
		rm -rf .venv

re: fclean run

.PHONY: install run debug clean lint lint-strict fclean re
.SILENT: install run debug clean lint lint-strict fclean re