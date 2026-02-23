#!/usr/bin/env python3
"""Runner semplice per lo scoring dei match holdout."""

from pathlib import Path

from score_match import (
    DEFAULT_DATA_ROOT,
    DEFAULT_HOLDOUT_PATH,
    DEFAULT_MODEL_PATH,
    run_holdout_scoring,
)

# Puoi personalizzare i path qui sotto. Lascia i default se non serve.
MODEL_PATH = DEFAULT_MODEL_PATH
DATA_ROOT = DEFAULT_DATA_ROOT
HOLDOUT_PATH = DEFAULT_HOLDOUT_PATH


def main() -> None:
    run_holdout_scoring(
        model_path=Path(MODEL_PATH),
        data_root=Path(DATA_ROOT),
        holdout_path=Path(HOLDOUT_PATH),
        require_360=False,
    )


if __name__ == "__main__":
    main()
