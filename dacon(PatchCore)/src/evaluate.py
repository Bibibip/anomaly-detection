"""저장된 체크포인트를 synthetic test 분할에서 평가한다."""

import argparse
import json
from pathlib import Path

from anomalib.engine import Engine

from .config import Settings
from .data_module import get_datamodule
from .models import build_dinomaly, build_patchcore, build_patchflow_effnet


def build_model(name: str, image_size: tuple[int, int]):
    """모델 이름에 해당하는 모델 객체를 생성한다."""
    if name == "patchflow":
        return build_patchflow_effnet(image_size)
    if name == "patchcore":
        return build_patchcore(image_size)
    return build_dinomaly(image_size)


def get_image_size(name: str, settings: Settings) -> tuple[int, int]:
    """모델별로 사용할 입력 이미지 크기를 반환한다."""
    return (224, 224) if name == "patchcore" else settings.image_size


def parse_args() -> argparse.Namespace:
    """평가에 필요한 명령행 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="체크포인트의 F1, AUROC, threshold를 출력합니다.")
    parser.add_argument("--model", choices=["patchflow", "patchcore", "dinomaly"], default="patchflow")
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """체크포인트 평가 결과와 image/pixel threshold를 JSON으로 저장한다."""
    args = parse_args()
    settings = Settings()
    model = build_model(args.model, get_image_size(args.model, settings))
    engine = Engine(default_root_dir=settings.output_dir / args.model, accelerator="gpu", devices=1)
    results = engine.test(
        model=model,
        datamodule=get_datamodule(settings),
        ckpt_path=args.checkpoint,
    )
    metrics = [{key: float(value) for key, value in result.items()} for result in results]
    post_processor = model.post_processor
    thresholds = {
        "image_threshold": float(post_processor.image_threshold),
        "pixel_threshold": float(post_processor.pixel_threshold),
    }
    payload = {"metrics": metrics, "thresholds": thresholds}
    metrics_path = settings.output_dir / args.model / "metrics" / "synthetic_test.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"평가 결과를 저장했습니다: {metrics_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
