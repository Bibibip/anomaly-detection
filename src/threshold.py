"""이미지 수준 예측 결과를 제출용 이진 라벨로 변환한다."""

from collections.abc import Iterable
from pathlib import Path


def _as_list(value) -> list:
    """Tensor, 단일 값, 반복 가능 값을 리스트로 통일한다."""
    if hasattr(value, "detach"):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def prediction_labels(predictions: Iterable) -> dict[str, int]:
    """파일명과 image-level pred_label을 대응시킨다."""
    labels: dict[str, int] = {}
    for batch in predictions:
        paths = _as_list(batch.image_path)
        raw_labels = getattr(batch, "pred_label", None)
        if raw_labels is None:
            # PostProcessor 라벨이 없을 때만 정규화 점수 0.5를 임시 기준으로 사용한다.
            raw_labels = [score >= 0.5 for score in _as_list(batch.pred_score)]
        for path, label in zip(paths, _as_list(raw_labels), strict=True):
            labels[Path(path).name] = int(float(label) >= 0.5)
    return labels
