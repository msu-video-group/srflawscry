import gc
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from srflawscry.metrics.wasd.labels import ARTIFACT_LABELS
from srflawscry.utils.device import empty_device_cache, preferred_device
from srflawscry.utils.images import open_rgb_image

PROMPT = f"""The panel compares the source and super-resolution result. Red outlines mark candidate artifact regions in the result.
Classify only the red-outlined regions. Return labels from: {", ".join(ARTIFACT_LABELS)}, or none.
Use structured_pattern_distortion for damaged regular repeated patterns: speaker grilles, meshes, grids, stripes, fabric weave, knitted texture, chart/table lines, or parallel lines.
Use surface_blotch for smooth flat blotches on glass, plastic, metal, road, pavement, walls, signs, balls, or other plain surfaces.
Use hallucinated_texture for irregular natural/detail corruption such as fur, hair, grass, foliage, snow, smoke, water, noisy detail, or miscellaneous non-text texture.
Return labels only, separated by commas."""

CLASS_PROBES = {
    "structured_pattern_distortion": (
        "Do the red outlines cover a regular repeated pattern such as a speaker "
        "grille, mesh, grid, stripes, fabric weave, knitted texture, chart/table "
        "lines, or parallel lines? Answer yes or no."
    ),
    "surface_blotch": (
        "Do the red outlines mainly cover smooth flat surface blotches on glass, "
        "plastic, metal, road, pavement, wall, sign, ball, or another plain "
        "surface? Answer yes or no."
    ),
    "hallucinated_texture": (
        "Do the red outlines cover irregular natural or noisy texture such as "
        "fur, hair, grass, foliage, snow, smoke, water, or miscellaneous "
        "non-text detail? Answer yes or no."
    ),
}

LABEL_KEYWORDS = {
    "hallucinated_texture": (
        "texture",
        "fur",
        "hair",
        "grass",
        "foliage",
        "leaf",
        "leaves",
        "snow",
        "smoke",
        "water",
        "noise",
        "noisy",
        "natural",
        "detail",
    ),
    "structured_pattern_distortion": (
        "pattern",
        "mesh",
        "grille",
        "grid",
        "stripe",
        "stripes",
        "fabric",
        "weave",
        "woven",
        "knit",
        "knitted",
        "speaker",
        "radio",
        "parallel",
        "line",
        "lines",
        "chart",
        "table",
    ),
    "surface_blotch": (
        "surface",
        "blotch",
        "blob",
        "smooth",
        "flat",
        "glass",
        "plastic",
        "metal",
        "road",
        "pavement",
        "wall",
        "ball",
        "plain",
    ),
}

PANEL_SIZE = 512
PANEL_GAP = 8
GLOBAL_PANEL_SIZE = 248
CROP_PANEL_WIDTH = 496
CROP_PANEL_HEIGHT = 352
PROCESSOR_IMAGE_SIZE = 512


def normalize_labels(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if normalized in {"", "none", "no", "no_artifact", "not_artifact"}:
        return []

    labels = [label for label in ARTIFACT_LABELS if label in normalized]
    if _looks_like_prompt_echo(normalized, labels):
        return []

    for label, keywords in LABEL_KEYWORDS.items():
        if label in labels:
            continue
        if any(_has_keyword(normalized, keyword) for keyword in keywords):
            labels.append(label)

    return _dedupe_labels(labels)


def _has_keyword(normalized: str, keyword: str) -> bool:
    keyword = re.sub(r"[^a-z0-9]+", "_", keyword.lower()).strip("_")
    return bool(re.search(rf"(?:^|_){re.escape(keyword)}(?:_|$)", normalized))


def _dedupe_labels(labels: list[str]) -> list[str]:
    return [label for label in ARTIFACT_LABELS if label in labels]


def _looks_like_prompt_echo(normalized: str, labels: list[str]) -> bool:
    if len(labels) >= 3:
        return True
    return bool(labels) and any(
        marker in normalized
        for marker in (
            "return_labels",
            "from_hallucinated_texture",
            "or_none",
            "answer_one",
            "choose",
            "candidate_artifact",
        )
    )


def _probe_is_yes(label: str, text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if normalized.startswith("no"):
        return False
    if normalized.startswith("yes"):
        return True
    return any(_has_keyword(normalized, keyword) for keyword in LABEL_KEYWORDS[label])


def _expand_interval(
    start: int, end: int, min_size: int, limit: int
) -> tuple[int, int]:
    if limit <= min_size or end - start >= min_size:
        return start, end

    missing = min_size - (end - start)
    start -= missing // 2
    end += missing - missing // 2
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    return max(start, 0), end


def _mask_bbox(
    mask: np.ndarray,
    pad: int,
    min_size: int,
    width: int,
    height: int,
) -> tuple[int, ...]:
    ys, xs = np.where(mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, width, height)

    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, width)
    y2 = min(int(ys.max()) + pad + 1, height)
    x1, x2 = _expand_interval(x1, x2, min_size, width)
    y1, y2 = _expand_interval(y1, y2, min_size, height)
    return x1, y1, x2, y2


def _annotate_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    annotated = image.copy()
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask) == 0:
        return annotated

    draw = ImageDraw.Draw(annotated)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=(255, 0, 0), width=3)
    return annotated


def _paste_centered(
    canvas: Image.Image,
    image: Image.Image,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    paste_left = left + (width - image.width) // 2
    paste_top = top + (height - image.height) // 2
    canvas.paste(image, (paste_left, paste_top))


def _thumbnail_to_fit(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((width, height))
    return image


def prepare_artifact_image(
    source_path: Path,
    result_path: Path,
    mask_path: Path,
    pad: int = 24,
    min_crop_size: int = PANEL_SIZE,
) -> Image.Image:
    result = open_rgb_image(result_path)
    source = open_rgb_image(source_path).resize(result.size, Image.BICUBIC)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")

    x1, y1, x2, y2 = _mask_bbox(
        mask,
        pad=pad,
        min_size=min_crop_size,
        width=result.width,
        height=result.height,
    )
    source_crop = source.crop((x1, y1, x2, y2))
    result_crop = result.crop((x1, y1, x2, y2))
    mask_crop = mask[y1:y2, x1:x2]

    source_crop = _thumbnail_to_fit(source_crop, GLOBAL_PANEL_SIZE, GLOBAL_PANEL_SIZE)
    global_result = _thumbnail_to_fit(
        _annotate_mask(result, mask),
        GLOBAL_PANEL_SIZE,
        GLOBAL_PANEL_SIZE,
    )
    result_crop = _thumbnail_to_fit(
        _annotate_mask(result_crop, mask_crop),
        CROP_PANEL_WIDTH,
        CROP_PANEL_HEIGHT,
    )

    panel = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), "white")
    _paste_centered(
        panel,
        source_crop,
        PANEL_GAP,
        PANEL_GAP,
        GLOBAL_PANEL_SIZE,
        GLOBAL_PANEL_SIZE,
    )
    _paste_centered(
        panel,
        global_result,
        PANEL_GAP * 2 + GLOBAL_PANEL_SIZE,
        PANEL_GAP,
        GLOBAL_PANEL_SIZE,
        GLOBAL_PANEL_SIZE,
    )
    _paste_centered(
        panel,
        result_crop,
        PANEL_GAP,
        PANEL_GAP * 2 + GLOBAL_PANEL_SIZE,
        CROP_PANEL_WIDTH,
        CROP_PANEL_HEIGHT,
    )
    return panel


class SmolVLMArtifactClassifier:
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        processor_image_size: int = PROCESSOR_IMAGE_SIZE,
    ) -> None:
        self.model_id = model_id
        self.device = device or preferred_device()
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        self.processor_image_size = processor_image_size
        self.processor = None
        self.model = None

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        empty_device_cache()

    def load(self) -> None:
        if self.model is not None:
            return

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.max_image_size = {
            "longest_edge": self.processor_image_size
        }
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            _attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

    def _generate(
        self, image: Image.Image, prompt_text: str, max_new_tokens: int
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        del inputs, generated_ids
        empty_device_cache()
        return generated_text

    @torch.inference_mode()
    def classify(
        self,
        source_path: Path,
        result_path: Path,
        mask_path: Path,
    ) -> list[str]:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        if np.count_nonzero(mask > 127) == 0:
            return []

        self.load()
        image = prepare_artifact_image(source_path, result_path, mask_path)
        labels = normalize_labels(self._generate(image, PROMPT, max_new_tokens=16))

        if not labels or len(labels) >= 3:
            labels = self._probe_labels(image)

        labels = self._apply_cv_priors(labels, result_path, mask)
        return _dedupe_labels(labels) or ["hallucinated_texture"]

    def _probe_labels(self, image: Image.Image) -> list[str]:
        labels = []
        for label, prompt in CLASS_PROBES.items():
            answer = self._generate(image, prompt, max_new_tokens=8)
            if _probe_is_yes(label, answer):
                labels.append(label)
        return labels

    def _apply_cv_priors(
        self,
        labels: list[str],
        result_path: Path,
        mask: np.ndarray,
    ) -> list[str]:
        labels = list(labels)
        edge_density, gradient_p90, line_count = _masked_edge_stats(result_path, mask)
        if _has_structured_pattern(edge_density, line_count):
            labels.append("structured_pattern_distortion")
        if _has_smooth_surface_blotch(edge_density, gradient_p90, line_count):
            labels.append("surface_blotch")

        labels = _dedupe_labels(labels)
        specific_labels = {
            "structured_pattern_distortion",
            "surface_blotch",
        }
        if any(label in labels for label in specific_labels):
            labels = [label for label in labels if label != "hallucinated_texture"]
        return labels


def _masked_edge_stats(result_path: Path, mask: np.ndarray) -> tuple[float, float, int]:
    image = cv2.imread(str(result_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, 0.0, 0
    if mask.shape != image.shape:
        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    active = mask > 127
    if np.count_nonzero(active) == 0:
        return 0.0, 0.0, 0

    active = cv2.dilate(active.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edges = cv2.Canny(image, 50, 150)
    edge_density = float(np.mean(edges[active] > 0))

    gradients_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradients_x * gradients_x + gradients_y * gradients_y)
    gradient_p90 = float(np.quantile(gradient[active], 0.9))

    ys, xs = np.where(active)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop_edges = edges[y1:y2, x1:x2].copy()
    crop_active = active[y1:y2, x1:x2]
    crop_edges[~crop_active] = 0
    min_line_length = max(8, min(crop_edges.shape[:2]) // 8)
    lines = cv2.HoughLinesP(
        crop_edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=min_line_length,
        maxLineGap=4,
    )
    line_count = 0 if lines is None else len(lines)
    return edge_density, gradient_p90, line_count


def _has_structured_pattern(edge_density: float, line_count: int) -> bool:
    return line_count >= 24 and edge_density >= 0.18


def _has_smooth_surface_blotch(
    edge_density: float,
    gradient_p90: float,
    line_count: int,
) -> bool:
    return edge_density <= 0.12 and gradient_p90 <= 140 and line_count < 12
