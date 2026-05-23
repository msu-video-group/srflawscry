from pathlib import Path

import pandas as pd

from srflawscryui.demo_data import demo_result_dirs
from srflawscryui.handlers.load import comparison_tables, sort_per_item_table_handler


def _write_metrics(root, df: pd.DataFrame) -> None:
    root.mkdir(parents=True)
    df.to_csv(root / "metrics.csv", index=False)


def test_demo_result_dirs_downloads_huggingface_dataset_when_local_data_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    def fake_snapshot_download(**kwargs):
        assert kwargs["local_dir"] == Path("assets")
        for method in ("RealESRGAN", "SwinIR"):
            (tmp_path / kwargs["local_dir"] / "data" / method).mkdir(parents=True)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(
        "srflawscryui.demo_data.snapshot_download",
        fake_snapshot_download,
    )

    assert demo_result_dirs() == (
        str(Path("assets") / "data" / "RealESRGAN"),
        str(Path("assets") / "data" / "SwinIR"),
    )


def test_comparison_tables_sort_per_item_by_first_metric_column():
    df_a = pd.DataFrame(
        [
            {
                "item_id": "low.png",
                "hallucinated_texture": 0.10,
                "structured_pattern_distortion": 0.10,
            },
            {
                "item_id": "high.png",
                "hallucinated_texture": 0.90,
                "structured_pattern_distortion": 0.10,
            },
            {
                "item_id": "mid.png",
                "hallucinated_texture": 0.30,
                "structured_pattern_distortion": 0.30,
            },
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "item_id": "low.png",
                "hallucinated_texture": 0.20,
                "structured_pattern_distortion": 0.10,
            },
            {
                "item_id": "high.png",
                "hallucinated_texture": 0.10,
                "structured_pattern_distortion": 0.10,
            },
            {
                "item_id": "mid.png",
                "hallucinated_texture": 0.60,
                "structured_pattern_distortion": 0.30,
            },
        ]
    )

    _, per_item_table = comparison_tables(df_a, df_b, "A", "B")

    assert per_item_table["item_id"].tolist() == [
        "mid.png",
        "low.png",
        "high.png",
    ]
    assert per_item_table.columns.tolist() == [
        "item_id",
        "hallucinated_texture",
        "structured_pattern_distortion",
    ]


def test_sort_per_item_table_handler_toggles_metric_direction(tmp_path):
    df_a = pd.DataFrame(
        [
            {"item_id": "negative.png", "hallucinated_texture": 0.90},
            {"item_id": "small.png", "hallucinated_texture": 0.10},
            {"item_id": "positive.png", "hallucinated_texture": 0.30},
        ]
    )
    df_b = pd.DataFrame(
        [
            {"item_id": "negative.png", "hallucinated_texture": 0.10},
            {"item_id": "small.png", "hallucinated_texture": 0.20},
            {"item_id": "positive.png", "hallucinated_texture": 0.60},
        ]
    )
    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    _write_metrics(dir_a, df_a)
    _write_metrics(dir_b, df_b)

    per_item_table = sort_per_item_table_handler(
        '{"metric": "hallucinated_texture", "ascending": false}',
        str(dir_a),
        str(dir_b),
    )
    assert per_item_table["item_id"].tolist() == [
        "positive.png",
        "small.png",
        "negative.png",
    ]

    per_item_table = sort_per_item_table_handler(
        '{"metric": "hallucinated_texture", "ascending": true}',
        str(dir_a),
        str(dir_b),
    )
    assert per_item_table["item_id"].tolist() == [
        "negative.png",
        "small.png",
        "positive.png",
    ]


def test_average_metrics_zero_values_are_fixed_precision():
    df_a = pd.DataFrame(
        [
            {
                "item_id": "example.png",
                "hallucinated_texture": 0.0,
            }
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "item_id": "example.png",
                "hallucinated_texture": 0.0,
            }
        ]
    )

    average_table, _ = comparison_tables(df_a, df_b, "A", "B")

    assert average_table.loc[0, "A"] == "0.0000"
    assert average_table.loc[0, "B"] == "0.0000"
    assert "+0.0000" in average_table.loc[0, "Diff"]
