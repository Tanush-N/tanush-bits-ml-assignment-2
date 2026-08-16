from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

try:
    from model.model_utils import (
        METRICS_PATH,
        MODEL_CONFIGS,
        TEST_DATA_PATH,
        evaluate_all_models,
        load_default_test_data,
        load_training_data,
        split_features_target,
    )
except ModuleNotFoundError:
    from model_utils import (
        METRICS_PATH,
        MODEL_CONFIGS,
        TEST_DATA_PATH,
        evaluate_all_models,
        load_default_test_data,
        load_training_data,
        split_features_target,
    )


def main() -> None:
    x_train, y_train = load_training_data()
    test_data = load_default_test_data()
    x_test, y_test = split_features_target(test_data)

    fitted_models = {}
    for model_name, config in MODEL_CONFIGS.items():
        model = config["estimator"]()
        model.fit(x_train, y_train)

        artifact_path = Path(config["artifact"])
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact_path)
        fitted_models[model_name] = model

    metrics = evaluate_all_models(fitted_models, x_test, y_test)
    metrics.to_csv(METRICS_PATH, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(test_data)),
        "test_data_file": str(TEST_DATA_PATH.name),
        "models": list(MODEL_CONFIGS),
    }
    metadata_path = Path(__file__).resolve().parent / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
