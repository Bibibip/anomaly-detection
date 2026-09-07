"""ImageNet 사전학습 Wide ResNet 기반 PatchCore 모델을 생성합니다."""

from anomalib.models.image.patchcore import Patchcore
from anomalib.pre_processing import PreProcessor
from anomalib.post_processing import PostProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize


def build_patchcore(image_size: tuple[int, int] = (224, 224)) -> Patchcore:
    """기준선 설정의 메모리 뱅크 기반 PatchCore 이상 탐지 모델을 생성합니다."""
    pre_processor = PreProcessor(Compose([
        Resize(image_size, antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    post_processor = PostProcessor(
        image_sensitivity=0.50,  # 정규화된 image score 0.50을 이상 판정 기준으로 사용합니다.
    )
    return Patchcore(
        backbone="wide_resnet50_2",  # ImageNet 사전학습 특징 추출기를 사용합니다.
        layers=("layer2", "layer3"),  # 서로 다른 크기의 이상을 포착할 특징 계층입니다.
        pre_trained=True,
        coreset_sampling_ratio=0.1,  # 전체 patch 중 10%를 메모리 뱅크에 보관합니다.
        num_neighbors=9,  # 이상 점수 재가중에 사용할 최근접 이웃 수입니다.
        precision="float32",
        pre_processor=pre_processor,
        post_processor=post_processor,
    )
