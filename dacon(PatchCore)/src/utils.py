"""대회 제출 CSV를 저장하는 보조 함수이다."""

from pathlib import Path

import pandas as pd


def write_submission(test_csv: Path, labels_by_filename: dict[str, int], output_csv: Path) -> None:
    """test.csv의 ID 순서와 제출 형식을 유지해 CSV를 저장한다."""
    test = pd.read_csv(test_csv)
    filenames = test["img_path"].map(lambda value: Path(value).name)
    missing = sorted(set(filenames) - set(labels_by_filename))
    if missing:
        raise ValueError(f"예측이 없는 테스트 이미지가 {len(missing)}개 있습니다: {missing[:3]}")
    submission = pd.DataFrame({"id": test["id"], "label": filenames.map(labels_by_filename).astype(int)})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)
