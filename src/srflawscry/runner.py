from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from srflawscry.layouts import AbstractEvaluationLayout
from srflawscry.metrics import AbstractMetric

ProgressCallback = Callable[[tuple[int, int], str], None]


def _noop_progress(value: tuple[int, int], desc: str) -> None:
    return None


class MetricRunner:
    def __init__(self, metrics: Sequence[AbstractMetric]) -> None:
        self.metrics = list(metrics)

    def evaluate(
        self,
        layout: AbstractEvaluationLayout,
        progress: ProgressCallback = _noop_progress,
        close_metrics: bool = True,
    ) -> pd.DataFrame:
        return self.evaluate_many(
            [layout],
            progress=progress,
            close_metrics=close_metrics,
        )[0]

    def evaluate_many(
        self,
        layouts: Sequence[AbstractEvaluationLayout],
        progress: ProgressCallback = _noop_progress,
        close_metrics: bool = True,
    ) -> list[pd.DataFrame]:
        batches = []
        for layout in layouts:
            samples = layout.iter_samples()
            if not samples:
                raise ValueError(f"No result images found in {layout.name}")

            df = self._load(layout, [sample.item_id for sample in samples])
            missing = {
                metric: [
                    sample
                    for sample in samples
                    if not self._has_values(df, sample.item_id, metric, layout)
                ]
                for metric in self.metrics
            }
            batches.append((layout, samples, df, missing))

        open_step_count = sum(
            len(metric.open_steps)
            for metric in self.metrics
            if any(missing[metric] for _, _, _, missing in batches)
            and not metric.is_open
        )
        total = max(
            sum(
                len(items) for _, _, _, missing in batches for items in missing.values()
            )
            + open_step_count,
            1,
        )
        done = 0

        for metric in self.metrics:
            if not any(missing[metric] for _, _, _, missing in batches):
                continue

            if not metric.is_open:
                for step in metric.open_steps:
                    progress((done, total), step)
                    metric.open_step(step)
                    done += 1

            for layout, _, df, missing in batches:
                for sample in missing[metric]:
                    progress(
                        (done, total),
                        (
                            f"Computing {metric.metric_id} metric for "
                            f"{layout.name}/{sample.item_id}"
                        ),
                    )
                    result = metric.evaluate(sample, layout)
                    for name, value in result.values.items():
                        df.loc[sample.item_id, name] = value
                    done += 1

        if close_metrics:
            for metric in self.metrics:
                metric.close()

        dataframes = []
        for layout, _, df, _ in batches:
            df = df.reset_index().rename(columns={"index": "item_id"})
            layout.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(layout.metrics_path, index=False)
            dataframes.append(df)
        return dataframes

    def _load(
        self,
        layout: AbstractEvaluationLayout,
        item_ids: list[str],
    ) -> pd.DataFrame:
        if layout.metrics_path.exists():
            df = pd.read_csv(layout.metrics_path, dtype={"item_id": str})
            df = df.set_index("item_id")
        else:
            df = pd.DataFrame()

        df.index = df.index.astype(str)
        missing_items = [item for item in item_ids if item not in df.index]
        return df.reindex([*df.index, *missing_items])

    def _has_values(
        self,
        df: pd.DataFrame,
        item_id: str,
        metric: AbstractMetric,
        layout: AbstractEvaluationLayout,
    ) -> bool:
        return all(
            spec.name in df.columns and pd.notna(df.loc[item_id, spec.name])
            for spec in metric.specs
        ) and metric.has_cached_outputs(layout, item_id)
