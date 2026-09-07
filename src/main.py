"""학습, 실제 테스트 추론, 제출 파일 생성을 한 번에 실행한다."""

import argparse
from pathlib import Path

from .config import Settings
from .data_module import get_datamodule
from .models import build_dinomaly, build_patchcore, build_patchflow_effnet
from .predict import predict_scores
from .threshold import prediction_labels
from .train import train_model
from .utils import write_submission


def build_model(name: str, image_size: tuple[int, int]):
    """선택한 모델 이름에 맞는 모델을 생성한다."""
    if name == "patchflow":
        return build_patchflow_effnet(image_size)
    if name == "patchcore":
        return build_patchcore(image_size)
    return build_dinomaly(image_size)


def get_image_size(name: str, settings: Settings) -> tuple[int, int]:
    """모델별로 사용할 입력 이미지 크기를 반환한다."""
    return (224, 224) if name == "patchcore" else settings.image_size


def parse_args() -> argparse.Namespace:
    """학습과 추론에 필요한 명령행 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="이상 탐지 모델을 학습하고 제출 CSV를 생성합니다.")
    parser.add_argument("--model", choices=["patchflow", "patchcore", "dinomaly"], default="patchflow")
    parser.add_argument("--max-epochs", type=int, help="학습 epoch 수를 직접 지정합니다.")
    parser.add_argument("--checkpoint", type=Path, help="학습을 생략하고 기존 체크포인트로 추론합니다.")
    return parser.parse_args()


def main() -> None:
    """모델을 학습하거나 불러온 뒤 실제 테스트 예측과 제출 저장을 수행한다."""
    args = parse_args()
    settings = Settings()
    image_size = get_image_size(args.model, settings)
    model = build_model(args.model, image_size)
    default_epochs = 1 if args.model == "patchcore" else 20
    max_epochs = args.max_epochs if args.max_epochs is not None else default_epochs
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        engine = train_model(
            model,
            get_datamodule(settings),
            args.model,
            max_epochs,
            str(settings.output_dir),
        )
        checkpoint = engine.best_model_path
        if not checkpoint:
            raise RuntimeError("학습 후 체크포인트가 생성되지 않았습니다.")

    model_output_dir = settings.output_dir / args.model
    predictions = predict_scores(model, checkpoint, settings.test_dir, model_output_dir, image_size)
    output = model_output_dir / "submission.csv"
    write_submission(settings.data_dir / "test.csv", prediction_labels(predictions), output)
    print(f"제출 파일을 저장했습니다: {output}")


if __name__ == "__main__":
    main()
