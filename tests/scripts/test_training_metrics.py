import json

from lerobot.scripts.lerobot_train import _append_training_metrics


def test_append_training_metrics_writes_flushable_jsonl_records(tmp_path):
    path = tmp_path / "weights" / "training_metrics.jsonl"

    _append_training_metrics(path, {"step": 100, "loss": 0.25, "action_loss": 0.2})
    _append_training_metrics(path, {"step": 200, "loss": 0.15, "tactile/summary_aux_loss": 0.03})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records] == [100, 200]
    assert records[0]["loss"] == 0.25
    assert records[1]["tactile/summary_aux_loss"] == 0.03
    assert all(record["timestamp"] for record in records)
