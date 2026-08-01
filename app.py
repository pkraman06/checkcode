import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image

from model import ResNetSEBrainTumor
from dataset import CLASSES, get_transforms
from xai import GradCAMPlusPlus, get_smoothgrad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetSEBrainTumor(num_classes=4).to(device)
checkpoint_path = "./checkpoints/best_model.pth"

if torch.cuda.is_available():
    model.load_state_dict(torch.load(checkpoint_path))
else:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Hook Grad-CAM++ into the last residual block
grad_cam_pp = GradCAMPlusPlus(model, target_layer=model.layer4)
_, val_transform = get_transforms()

def predict_and_explain(raw_img):
    if raw_img is None:
        return None, None, None
    
    img_pil = Image.fromarray(raw_img).convert("RGB")
    input_tensor = val_transform(img_pil).unsqueeze(0).to(device)

    # Class probabilities
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    conf_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    # Generate Grad-CAM++
    cam, target_cls = grad_cam_pp.generate(input_tensor)
    orig_img_cv = cv2.resize(raw_img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    gradcam_overlay = cv2.addWeighted(orig_img_cv, 0.6, heatmap, 0.4, 0)

    # Generate SmoothGrad Saliency Map
    saliency_input = val_transform(img_pil).unsqueeze(0).to(device)
    smooth_saliency = get_smoothgrad(model, saliency_input, n_samples=20)
    saliency_colored = cv2.applyColorMap(np.uint8(255 * smooth_saliency), cv2.COLORMAP_HOT)
    saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)

    return conf_dict, gradcam_overlay, saliency_colored

with gr.Blocks(title="Brain Tumor Diagnosis & Advanced XAI") as demo:
    gr.Markdown("# 🧠 Brain Tumor MRI Classification with Grad-CAM++ and SmoothGrad")
    gr.Markdown(
        "Upload a brain MRI scan to view classification confidence along with **Grad-CAM++** "
        "(sharper tumor localization) and **SmoothGrad** (noise-reduced pixel sensitivity)."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload MRI Scan")
            submit_btn = gr.Button("Analyze Scan", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(num_top_classes=4, label="Prediction Probabilities")

    with gr.Row():
        gradcam_output = gr.Image(label="Grad-CAM++ (Sharper Regional Attention)")
        saliency_output = gr.Image(label="SmoothGrad Saliency Map (Clean Pixel Sensitivity)")

    submit_btn.click(
        fn=predict_and_explain,
        inputs=[input_image],
        outputs=[label_output, gradcam_output, saliency_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
