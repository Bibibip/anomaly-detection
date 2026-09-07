"""학습과 추론에서 공통으로 사용하는 설정을 정의한다."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """데이터 경로, 출력 경로, 입력 이미지 크기를 보관한다."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    image_size: tuple[int, int] = (512, 512)
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_workers: int = 0  # Windows 환경의 DataLoader 문제를 피하기 위해 0으로 둔다.

    @property
    def test_dir(self) -> Path:
        """실제 제출용 테스트 이미지 폴더 경로를 반환한다."""
        return self.data_dir / "test"
