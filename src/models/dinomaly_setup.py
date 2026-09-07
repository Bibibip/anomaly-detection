"""개인 실험용 Dinomaly 모델을 생성한다."""

from anomalib.models import Dinomaly
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize


def build_dinomaly(image_size: tuple[int, int] = (512, 512)) -> Dinomaly:
    """DINOv2 기반 Dinomaly 모델을 생성한다.

    이 모델은 개인 실험용이며 대회 제출에는 사용하지 않는다.
    """
    return Dinomaly(
        encoder_name="vit_base_patch14_reg4_dinov2",
        decoder_depth=8,  # 디코더의 ViT 레이어 수
        pre_processor=PreProcessor(Compose([
            Resize(image_size, antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
    )
