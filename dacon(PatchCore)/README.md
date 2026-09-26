# Semiconductor Anomaly Detection

정상 이미지로 이상 탐지 모델을 학습하고, 테스트 이미지의 이진 라벨 제출 파일을 만드는 프로젝트입니다.

## 설치와 GPU 확인

프로젝트 루트에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

RTX 4060 GPU를 사용하려면 CPU용 PyTorch를 CUDA 빌드로 교체해야 합니다.

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
```

`CUDA: True`와 GPU 이름이 출력되어야 합니다.

## 실행

Patchflow는 ImageNet 사전학습 EfficientNet-B5를 사용하며 입력 크기 512 × 512으로 학습합니다. RTX 4060 8GB에 맞춰 배치 크기는 2로 설정되어 있습니다. synthetic test 중 50%를 validation으로 사용해 image-level F1 threshold를 추정하고, 이 validation의 `image_AUROC`가 가장 높은 epoch 체크포인트를 자동으로 선택합니다.

```powershell
.\.venv\Scripts\python.exe -m src.main --model patchflow --max-epochs 1
```

문제가 없으면 epoch를 늘립니다.

```powershell
.\.venv\Scripts\python.exe -m src.main --model patchflow --max-epochs 20
```

## PatchCore 비교 실험

PatchCore는 ImageNet 사전학습 Wide ResNet-50의 정상 patch 특징을 메모리 뱅크에 저장하고, 테스트 patch와의 최근접 이웃 거리를 이용해 이상을 판단합니다. 입력 크기는 224 × 224, coreset 비율은 0.1이며 실행 결과는 Patchflow와 섞이지 않도록 `outputs/patchcore/`에 저장됩니다.

```powershell
.\.venv\Scripts\python.exe -m src.main --model patchcore
```

PatchCore는 gradient 기반 모델이 아니므로 `optimizer 없음` 메시지는 정상입니다. 여러 epoch으로 일반 신경망처럼 가중치를 반복 갱신하지 않으므로 기본 실행은 1 epoch으로 설정합니다.

## 학습 그래프 (TensorBoard)

TensorBoard가 아직 없다면 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install tensorboard
```

학습을 실행한 상태에서 별도 PowerShell 창을 열어 TensorBoard를 시작합니다.

```powershell
.\.venv\Scripts\tensorboard.exe --logdir .\outputs\patchflow\tensorboard
```

표시되는 `http://localhost:6006` 주소를 브라우저에서 열면 epoch별 `train_loss`와 validation AUROC를 볼 수 있습니다. Dinomaly는 경로에서 `patchflow`를 `dinomaly`로 바꾸면 됩니다.

## F1 확인

가장 최근의 체크포인트를 찾아 synthetic test 분할의 F1/AUROC를 출력합니다.

```powershell
$ckpt = Get-ChildItem .\outputs\patchflow -Recurse -Filter model.ckpt |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName

.\.venv\Scripts\python.exe -m src.evaluate --model patchflow --checkpoint $ckpt
```

이 F1은 제공 정상 이미지에서 만든 synthetic test 분할 점수입니다. 실제 Public/Private 리더보드 F1과는 다릅니다.

## 결과물

```text
outputs/
└─ patchflow/
   ├─ .../weights/lightning/model.ckpt  # 학습 체크포인트
   ├─ .../images/test/                  # 이상 맵 시각화
   ├─ metrics/synthetic_test.json       # image/pixel threshold, F1/AUROC
   ├─ tensorboard/version_0/            # 학습 곡선 이벤트 파일
   └─ submission.csv                    # 제출 파일
```

이후 실행 결과는 `outputs/`에 모입니다.
