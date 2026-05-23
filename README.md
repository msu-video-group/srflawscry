<p align="center">
  <h1 align="center">SR Flaw Scry: Artifact Detection for Super-Resolution</h1>
  <h3 align="center">
    <a href="https://huggingface.co/spaces/egorchistov/sr-artifact-detection-srflawscry">🤗 Demo</a>
    |
    <a href="https://huggingface.co/egorchistov/sr-artifact-detection-wasd">📦 Model</a>
    |
    <a href="https://huggingface.co/datasets/egorchistov/sr-artifact-detection-demo-dataset">🗂️ Dataset</a>
  </h3>
</p>

## 🏅 Overview

Find **where** super-resolution goes wrong and **which artifacts** appear

## 🤗 Demo

Try our hosted demo on
[Hugging Face](https://huggingface.co/spaces/egorchistov/sr-artifact-detection-srflawscry). The demo compares **RealESRGAN** and **SwinIR** on examples with texture
hallucinations, stripe-like pattern artifacts, and blur-like surface defects.

## 🖥️ Run the UI

To run the project locally, install the package first:

```shell
pip3 install git+https://github.com/msu-video-group/srflawscry
```

Then launch the interface:

```shell
python3 -m srflawscryui
```

It will automatically download our demo dataset and model. To use your own
images, put them into `assets/data` folder with this structure:

```text
assets/data/
  MyMethod/
    sources/<item_id>
    results/<item_id>
```

where the `sources` folder contains the original low-resolution inputs and the
`results` folder contains the 4x-upscaled images with matching file names.
Metric computation is supported on CPUs, Apple Silicon Macs, and CUDA-capable GPUs.
At least **3 GB of VRAM** is recommended for GPU inference.

## 🚀 Using WASD in Your Project

Our artifact detection metric is called WASD. You can easily add it to your
project with the following snippet:

```python
from pathlib import Path

from srflawscry import WASD

model = WASD.from_pretrained("egorchistov/sr-artifact-detection-wasd").eval()
model.run(
    image_path=Path("result.png"),
    source_path=Path("source.png"),
    output_path=Path("artifact_mask.png"),
)
```

## 🛠️ Extending This Repository

For local development, you need the full installation:

```shell
git clone https://github.com/msu-video-group/srflawscry.git
cd srflawscry
pip3 install --editable .[dev]
pre-commit install
```

You can extend this repository in many ways. For example, if you want to add a
simple scalar metric such as **LPIPS** or **MANIQA**, create a metric class that
returns one number per image and add it to `default_config()`. If the metric also
produces a heatmap or mask, save it in `outputs/` so the UI can display it beside
the compared images.

```text
FolderWithSourcesLayout
        │
        ▼
Sample(item_id, result_path, source_path)
        │
        ▼
Metric.evaluate(sample) ──► numbers in metrics.csv
        │
        └───────────────► optional masks in outputs/
```

## ❓ Need Help?

Feel free to open an issue if you have any questions.
