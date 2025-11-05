import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.segmentation as seg_models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# -----------------------------
# ENSNET Model for Classification
# -----------------------------
class ENSNET(nn.Module):
    def __init__(self, num_classes=3):
        super(ENSNET, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        resnet_features = self.resnet.fc.in_features
        densenet_features = self.densenet.classifier.in_features
        efficientnet_features = self.efficientnet.classifier[1].in_features

        self.resnet.fc = nn.Identity()
        self.densenet.classifier = nn.Identity()
        self.efficientnet.classifier = nn.Identity()

        total_features = resnet_features + densenet_features + efficientnet_features
        self.classifier = nn.Linear(total_features, num_classes)

    def forward(self, x):
        r = self.resnet(x)
        d = self.densenet(x)
        e = self.efficientnet(x)
        features = torch.cat([r, d, e], dim=1)
        return self.classifier(features)

# -----------------------------
# Mask Overlay Function
# -----------------------------
def overlay_mask_on_image(image, mask):
    image = image.resize((256, 256))
    img_np = np.array(image).astype(np.uint8)

    red_overlay = np.zeros_like(img_np)
    red_overlay[:, :, 0] = 255  # Red channel

    mask_3ch = np.stack([mask] * 3, axis=-1)
    alpha = 0.5
    overlayed = np.where(mask_3ch == 1, (1 - alpha) * img_np + alpha * red_overlay, img_np).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlayed = cv2.drawContours(overlayed, contours, -1, (255, 0, 0), 2)
    return overlayed

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Brain CT Classifier & Segmenter", layout="wide")
st.title("Brain CT Classification and Segmentation")
st.markdown("---")

# Layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload a Brain CT Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classification
        transform_cls = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform_cls(image).unsqueeze(0).to(device)

        cls_model = ENSNET(num_classes=3).to(device)
        cls_model.load_state_dict(torch.load("best_brain_ct_model.pth", map_location=device))
        cls_model.eval()

        with torch.no_grad():
            output = cls_model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()

        class_names = ['Normal', 'Hemorrhage', 'Ischemia']
        st.subheader("🔍 Prediction Result")
        st.success(f"**{class_names[pred_class]}**")

        if pred_class in [1, 2]:
            transform_seg = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
            ])
            input_tensor_seg = transform_seg(image).unsqueeze(0).to(device)

            seg_model = seg_models.deeplabv3_resnet50(pretrained=True)
            seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
            seg_model = seg_model.to(device)

            model_path = "deeplabv3_final_hem.pt" if pred_class == 1 else "deeplabv3_final.pt"
            seg_model.load_state_dict(torch.load(model_path, map_location=device))
            seg_model.eval()

            with torch.no_grad():
                output = seg_model(input_tensor_seg)['out']
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                binary_mask = (pred_mask > 0.5).astype(np.uint8)

            overlayed_image = overlay_mask_on_image(image, binary_mask)

            st.markdown("---")
            st.subheader("Segmentation Comparison")

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(image.resize((512, 512)), caption="Original Image", width =512)
            with img_col2:
                st.image(overlayed_image, caption="Overlayed Mask",width =512)
        else:
            st.info(" No segmentation required for Normal scans.")
