"""Zero-shot DINOv2 clustering and CLIP defect naming for MVTec.

Test masks are never read.  CLIP uses full and local views, since resizing an
object image to 224 pixels can otherwise erase a small defect.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CATEGORIES = ("wood", "tile", "transistor", "screw")
TEXTURE_GROUP = {"wood", "tile"}

DEFECT_DESCRIPTIONS = {
    "wood": {"color": "a color discoloration or stain", "combined": "multiple defects including a scratch and discoloration", "hole": "a hole in the wood surface", "liquid": "a liquid stain or spill", "scratch": "a scratch on the wood surface"},
    "tile": {"crack": "a crack in the tile", "glue_strip": "a strip of glue on the tile", "gray_stroke": "a gray streak or gray line", "oil": "an oil stain", "rough": "a rough damaged surface texture"},
    "transistor": {"bent_lead": "a bent metal lead", "cut_lead": "a cut or missing metal lead", "damaged_case": "a damaged or cracked plastic case", "misplaced": "a misplaced or misaligned component"},
    "screw": {"manipulated_front": "a damaged or manipulated front face", "scratch_head": "a scratch on the screw head", "scratch_neck": "a scratch on the screw neck", "thread_side": "damage on the side of the screw thread", "thread_top": "damage on the top of the screw thread"},
}
PROMPT_TEMPLATES = (
    "a product inspection photo of a {category} with {defect}",
    "a close-up industrial inspection image of a {category} showing {defect}",
    "a defective {category}; the defect is {defect}",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("./mvtec"))
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def collect_defect_images(data_root, category):
    images, labels = [], []
    for folder in sorted((data_root / category / "test").iterdir()):
        if not folder.is_dir() or folder.name == "good":
            continue
        for image_path in sorted(folder.glob("*.png")):
            images.append(image_path)
            labels.append(folder.name)
    return images, labels


def collect_normal_images(data_root, category):
    return sorted((data_root / category / "train" / "good").glob("*.png"))


def purity_nmi(prediction, ground_truth, cluster_count):
    total = sum(np.bincount(ground_truth[prediction == cluster]).max()
                for cluster in range(cluster_count) if (prediction == cluster).any())
    return total / len(ground_truth), normalized_mutual_info_score(ground_truth, prediction)


def inspection_views(image, anomaly_center):
    """Full, fixed local, and DINO-selected anomalous-region views."""
    width, height = image.size
    crop_width, crop_height = int(width * 0.60), int(height * 0.60)
    positions = ((0, 0), (width - crop_width, 0), (0, height - crop_height),
                 (width - crop_width, height - crop_height),
                 ((width - crop_width) // 2, (height - crop_height) // 2))
    fixed_views = [image.crop((x, y, x + crop_width, y + crop_height)) for x, y in positions]

    # The ROI covers 45% of the image.  Clamping keeps an edge patch centred
    # in the crop rather than producing an invalid crop rectangle.
    roi_width, roi_height = int(width * 0.45), int(height * 0.45)
    center_x, center_y = anomaly_center
    left = int(center_x * width - roi_width / 2)
    top = int(center_y * height - roi_height / 2)
    left = min(max(left, 0), width - roi_width)
    top = min(max(top, 0), height - roi_height)
    anomaly_view = image.crop((left, top, left + roi_width, top + roi_height))
    return [image] + fixed_views + [anomaly_view]


@torch.no_grad()
def encode_clip_views(model, preprocess, image_paths, anomaly_centers, batch_size):
    view_count, tensors = 7, []
    for path, anomaly_center in zip(image_paths, anomaly_centers):
        with Image.open(path) as source:
            tensors.extend(preprocess(view) for view in inspection_views(source.convert("RGB"), anomaly_center))
    features = []
    for start in range(0, len(tensors), batch_size):
        embedding = model.encode_image(torch.stack(tensors[start:start + batch_size]).to(DEVICE))
        features.append(embedding / embedding.norm(dim=-1, keepdim=True))
    return torch.cat(features).reshape(len(image_paths), view_count, -1)


@torch.no_grad()
def encode_texts(model, tokenizer, category, defect_types):
    prototypes = []
    for defect_type in defect_types:
        description = DEFECT_DESCRIPTIONS.get(category, {}).get(defect_type, defect_type.replace("_", " "))
        prompts = [template.format(category=category, defect=description) for template in PROMPT_TEMPLATES]
        embedding = model.encode_text(tokenizer(prompts).to(DEVICE))
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        prototype = embedding.mean(dim=0)
        prototypes.append(prototype / prototype.norm())
    return torch.stack(prototypes)


def aggregate_view_scores(view_features, text_features):
    """Combine product context, fixed crops, and the normal-reference ROI."""
    similarities = view_features @ text_features.T  # [images, 7, defects]
    global_score = similarities[:, 0, :]
    fixed_local_score = similarities[:, 1:-1, :].topk(k=2, dim=1).values.mean(dim=1)
    roi_score = similarities[:, -1, :]
    return (0.25 * global_score + 0.35 * fixed_local_score + 0.40 * roi_score).mean(dim=0)


@torch.no_grad()
def dino_global_and_patches(model, preprocess, image_paths, batch_size):
    """Return global DINO features and L2-normalised patch tokens."""
    global_features, patch_features = [], []
    for start in range(0, len(image_paths), batch_size):
        tensors = []
        for path in image_paths[start:start + batch_size]:
            with Image.open(path) as source:
                tensors.append(preprocess(source.convert("RGB")))
        outputs = model.forward_features(torch.stack(tensors).to(DEVICE))
        global_features.append(outputs["x_norm_clstoken"].cpu())
        patches = outputs["x_norm_patchtokens"]
        patch_features.append(patches / patches.norm(dim=-1, keepdim=True))
    return torch.cat(global_features).numpy(), torch.cat(patch_features)


@torch.no_grad()
def normal_patch_reference(model, preprocess, normal_paths, batch_size):
    """Build a position-wise normal DINO prototype from train/good images."""
    _, normal_patches = dino_global_and_patches(model, preprocess, normal_paths, batch_size)
    reference = normal_patches.mean(dim=0)
    return reference / reference.norm(dim=-1, keepdim=True)


def patch_anomaly_centers(test_patches, normal_reference):
    """Locate the least normal patch in each test image without using masks."""
    normal_similarity = (test_patches * normal_reference.unsqueeze(0)).sum(dim=-1)
    patch_indices = normal_similarity.argmin(dim=1).cpu().numpy()
    grid_size = int(np.sqrt(test_patches.shape[1]))
    if grid_size * grid_size != test_patches.shape[1]:
        raise ValueError("DINO patch token count is not a square grid")
    return [((index % grid_size + 0.5) / grid_size,
             (index // grid_size + 0.5) / grid_size) for index in patch_indices]


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE).eval()
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(DEVICE).eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    dino_preprocess = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_results = {}
    for category in args.categories:
        image_paths, labels = collect_defect_images(args.data_root, category)
        defect_types = sorted(set(labels))
        label_to_index = {label: index for index, label in enumerate(defect_types)}
        ground_truth = np.array([label_to_index[label] for label in labels])

        normal_paths = collect_normal_images(args.data_root, category)
        normal_reference = normal_patch_reference(
            dinov2, dino_preprocess, normal_paths, args.batch_size
        )
        dino_features, test_patches = dino_global_and_patches(
            dinov2, dino_preprocess, image_paths, args.batch_size
        )
        anomaly_centers = patch_anomaly_centers(test_patches, normal_reference)
        # KMeans on unit vectors approximates cosine clustering, which fits
        # DINO embeddings better than the unnormalised Euclidean distances.
        dino_features = normalize(dino_features)
        cluster_count = len(defect_types)
        prediction = KMeans(n_clusters=cluster_count, random_state=42, n_init=30).fit_predict(dino_features)
        purity, nmi = purity_nmi(prediction, ground_truth, cluster_count)

        clip_features = encode_clip_views(
            clip_model, clip_preprocess, image_paths, anomaly_centers, args.batch_size
        )
        text_features = encode_texts(clip_model, clip_tokenizer, category, defect_types)
        # Score every cluster against every defect before assigning a name.
        # A one-to-one global assignment prevents all clusters from selecting
        # the same CLIP-favoured prompt (for example, every tile -> "rough").
        score_matrix, cluster_masks = [], []
        for cluster in np.unique(prediction):
            mask = prediction == cluster
            scores = aggregate_view_scores(clip_features[mask], text_features).cpu().numpy()
            score_matrix.append(scores)
            cluster_masks.append(mask)

        score_matrix = np.asarray(score_matrix)
        cluster_indices, defect_indices = linear_sum_assignment(-score_matrix)

        confusion_pairs, margins, cluster_purities = [], [], []
        for cluster_index, defect_index in zip(cluster_indices, defect_indices):
            mask = cluster_masks[cluster_index]
            predicted = defect_types[defect_index]
            actual = defect_types[np.bincount(ground_truth[mask]).argmax()]
            confusion_pairs.append((predicted, actual))
            sorted_scores = np.sort(score_matrix[cluster_index])[::-1]
            margins.append(sorted_scores[0] - sorted_scores[1])
            cluster_purities.append(np.bincount(ground_truth[mask]).max() / mask.sum())

        naming_accuracy = np.mean([predicted == actual for predicted, actual in confusion_pairs])
        confident_clusters = np.mean(np.array(margins) > 0.01)
        correlation = np.corrcoef(cluster_purities, margins)[0, 1]
        all_results[category] = {"group": "texture" if category in TEXTURE_GROUP else "object", "naming_accuracy": naming_accuracy}
        print(f"[{category}] purity={purity:.3f} | nmi={nmi:.3f} | naming_acc={naming_accuracy:.2%} | "
            f"confident_clusters={confident_clusters:.2%} | margin-purity corr={correlation:+.3f}")
        print(f"  confusion: {dict(Counter(confusion_pairs))}\n")

    texture_accuracy = np.mean([r["naming_accuracy"] for r in all_results.values() if r["group"] == "texture"])
    object_accuracy = np.mean([r["naming_accuracy"] for r in all_results.values() if r["group"] == "object"])
    print(f"\nTexture-group mean naming accuracy: {texture_accuracy:.2%}")
    print(f"Object-group mean naming accuracy: {object_accuracy:.2%}")


if __name__ == "__main__":
    main()
