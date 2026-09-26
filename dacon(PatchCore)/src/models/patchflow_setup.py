"""ImageNet 사전학습 EfficientNet-B5 기반 Patchflow 모델을 생성한다."""

from anomalib.models.image.patchflow import Patchflow
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize


def _pre_processor(image_size: tuple[int, int]) -> PreProcessor:
    """학습과 추론에 동일하게 적용할 이미지 전처리기를 생성한다."""
    return PreProcessor(Compose([
        Resize(image_size, antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))


def build_patchflow_effnet(image_size: tuple[int, int] = (512, 512)) -> Patchflow:
    """Patchflow 모델을 생성한다."""
    return Patchflow(
        backbone="tf_efficientnet_b5",  # ImageNet 사전학습 EfficientNet-B5 특징 추출기
        pre_trained=True,
        flow_steps=1,
        flow_feature_dim=64,
        num_scales=2,
        patch_size=5,
        flow_hidden_dim=64,
        crop_size=None,
        pre_processor=_pre_processor(image_size),
    )
