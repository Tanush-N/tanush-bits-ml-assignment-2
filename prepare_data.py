from __future__ import annotations

import csv
import random
from pathlib import Path

from model.constants import FEATURE_COLUMNS, TARGET_COLUMN


RAW_DATA_PATH = Path("raw_data/wdbc.data")
FULL_DATA_PATH = Path("dataset.csv")
TRAIN_DATA_PATH = Path("training_data.csv")
TEST_DATA_PATH = Path("test_data.csv")


def load_rows() -> list[dict[str, str]]:
    columns = ["id", TARGET_COLUMN, *FEATURE_COLUMNS]
    rows = []

    with RAW_DATA_PATH.open(newline="", encoding="utf-8") as raw_file:
        reader = csv.reader(raw_file)
        for row in reader:
            if not row:
                continue

            if len(row) != len(columns):
                raise ValueError(f"Expected {len(columns)} columns but found {len(row)}")

            record = dict(zip(columns, row))
            record.pop("id")
            rows.append(record)

    return rows


def stratified_split(rows: list[dict[str, str]], test_ratio: float = 0.2) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    random.seed(42)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[TARGET_COLUMN], []).append(row)

    training_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for group_rows in groups.values():
        random.shuffle(group_rows)
        test_count = round(len(group_rows) * test_ratio)
        test_rows.extend(group_rows[:test_count])
        training_rows.extend(group_rows[test_count:])

    random.shuffle(training_rows)
    random.shuffle(test_rows)
    return training_rows, test_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()
    training_rows, test_rows = stratified_split(rows)

    write_csv(FULL_DATA_PATH, rows)
    write_csv(TRAIN_DATA_PATH, training_rows)
    write_csv(TEST_DATA_PATH, test_rows)

    print(f"Wrote {len(rows)} rows to {FULL_DATA_PATH}")
    print(f"Wrote {len(training_rows)} rows to {TRAIN_DATA_PATH}")
    print(f"Wrote {len(test_rows)} rows to {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
