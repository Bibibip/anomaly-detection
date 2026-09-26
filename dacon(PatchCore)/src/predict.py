"""실제 테스트 이미지의 이상 탐지 결과를 생성한다."""

from anomalib.data import PredictDataset
from anomalib.engine import Engine


def predict_scores(model, ckpt_path, test_dir, output_dir, image_size=(512, 512)):
    """테스트 폴더의 모든 이미지에 대한 Anomalib 예측 결과를 반환한다."""
    dataset = PredictDataset(path=test_dir, image_size=image_size)
    predictions = Engine(default_root_dir=output_dir).predict(model=model, dataset=dataset, ckpt_path=ckpt_path)
    if predictions is None:
        raise RuntimeError("테스트 이미지 예측 결과가 생성되지 않았습니다.")
    return predictions
