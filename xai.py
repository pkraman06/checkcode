import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAMPlusPlus:
    """
    Grad-CAM++ Implementation.
    Uses second and third-order gradients to provide better localization,
    especially for small tumor lesions and multi-instance features.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        # Compute score for target class
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Grad-CAM++ weight calculations using higher-order derivatives
        g2 = gradients ** 2
        g3 = gradients ** 3

        # Compute alpha coefficients
        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
        alpha_denom = 2 * g2 + sum_activations * g3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = g2 / alpha_denom

        # Positive gradients contribution
        relu_grad = F.relu(gradients)
        weights = torch.sum(alpha * relu_grad, dim=(1, 2))  # [C]

        # Weighted combination of feature maps
        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        cam = F.relu(cam).cpu().data.numpy()

        # Resize to input dimensions and normalize
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, target_class


def get_smoothgrad(model, input_tensor, n_samples=25, noise_level=0.15):
    """
    SmoothGrad Implementation.
    Adds Gaussian noise to the input tensor across N samples and averages 
    the gradients to reduce pixel noise present in Vanilla Saliency maps.
    """
    model.eval()
    stdev = noise_level * (input_tensor.max() - input_tensor.min()).item()
    total_gradients = torch.zeros_like(input_tensor)

    target_class = model(input_tensor).argmax(dim=1).item()

    for _ in range(n_samples):
        # Generate Gaussian noise
        noise = torch.randn_like(input_tensor) * stdev
        noisy_input = (input_tensor + noise).clone().detach().requires_grad_(True)

        output = model(noisy_input)
        loss = output[0, target_class]
        loss.backward()

        total_gradients += noisy_input.grad.data.abs()
        model.zero_grad()

    # Average gradients across samples
    smooth_grad = total_gradients / n_samples
    saliency, _ = torch.max(smooth_grad, dim=1)
    saliency = saliency[0].cpu().numpy()
    
    # Normalize 0-1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency
