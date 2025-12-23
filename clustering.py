import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import torch
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RAW_IMAGE_DIR = "all_numbers_tiles"
OUTPUT_DIR = "clusteredData"
IMAGE_SIZE = 224
N_CLUSTERS = 12  
RANDOM_STATE = 42


# LOAD PRETRAINED MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()


# IMAGE TRANSFORM

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# FEATURE EXTRACTION FUNCTION

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img)

    return features.squeeze().cpu().numpy()


# LOAD IMAGES & EXTRACT FEATURES

features = []
image_paths = []

print("Extracting features...")
for filename in tqdm(os.listdir(RAW_IMAGE_DIR)):
    img_path = os.path.join(RAW_IMAGE_DIR, filename)

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    feat = extract_features(img_path)
    if feat is not None:
        features.append(feat)
        image_paths.append(img_path)

X = np.array(features)

print(f"Total images processed: {len(X)}")


# SCALE FEATURES

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# KMEANS CLUSTERING

print("Clustering images...")
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)
cluster_labels = kmeans.fit_predict(X_scaled)


# CREATE CLUSTER FOLDERS

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, cluster_id in enumerate(cluster_labels):
    cluster_folder = os.path.join(OUTPUT_DIR, f"group_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)

    shutil.copy(image_paths[idx], cluster_folder)

print("Clustering complete.")
print(f"Clusters saved in: {OUTPUT_DIR}")
