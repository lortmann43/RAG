SRC = src

install:
		uv sync

run: install
		uv run python3 -m $(SRC)

debug: install
		uv run python3 -m pdb -m $(SRC)

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

fclean: clean
		rm -rf .venv

re: fclean run

.PHONY: install run debug clean lint lint-strict fclean re
.SILENT: install run debug clean lint lint-strict fclean re