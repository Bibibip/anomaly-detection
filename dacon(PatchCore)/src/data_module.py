"""정상 이미지와 synthetic 이상 데이터를 구성하는 데이터 모듈이다."""

from anomalib.data import Folder

from .config import Settings


def get_datamodule(settings: Settings, val_split_ratio: float = 0.5) -> Folder:
    """정상 학습 이미지에서 학습·검증·synthetic 평가 분할을 만든다."""
    return Folder(
        name="semiconductor",
        root=settings.data_dir,
        normal_dir="train",  # 정상 이미지가 있는 폴더만 학습 데이터로 사용한다.
        test_split_mode="synthetic",  # 정상 이미지 기반의 인공 이상 평가 분할을 생성한다.
        val_split_ratio=val_split_ratio,
        train_batch_size=settings.train_batch_size,
        eval_batch_size=settings.eval_batch_size,
        num_workers=settings.num_workers,
        seed=42,  # 매 실행에서 동일한 synthetic 분할을 재현한다.
    )
