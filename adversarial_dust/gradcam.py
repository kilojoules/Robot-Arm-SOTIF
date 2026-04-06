"""GradCAM visualization for the safety predictor CNN.

Generates attention heatmaps showing which image regions drive the
failure prediction, providing mechanistic evidence for cross-corruption
generalization.

Usage:
    from adversarial_dust.gradcam import GradCAM
    cam = GradCAM(predictor)
    heatmap, overlay = cam.generate(image)
"""

import cv2
import numpy as np


class GradCAM:
    """Gradient-weighted Class Activation Mapping for SafetyPredictorCNN.

    Hooks into the last convolutional layer of the backbone to compute
    attention maps. For ResNet, this is layer4. For ViT, this uses the
    attention weights from the last transformer block.
    """

    def __init__(self, predictor):
        """Initialize with a SafetyPredictorCNN instance.

        Args:
            predictor: A SafetyPredictorCNN with a loaded model.
        """
        import torch
        self._torch = torch
        self.predictor = predictor
        self.model = predictor.model
        self._activations = None
        self._gradients = None

        # Register hooks on the last conv layer
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _get_target_layer(self):
        """Find the last convolutional layer in the backbone."""
        backbone_name = self.predictor.backbone
        if backbone_name in ("resnet18", "resnet50"):
            return self.model.backbone.layer4[-1]
        elif backbone_name == "vit_b_16":
            return self.model.backbone.encoder.layers[-1].ln_1
        else:
            raise ValueError(f"GradCAM not implemented for {backbone_name}")

    def _forward_hook(self, module, input, output):
        self._activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, image: np.ndarray, target_class: int = 1) -> tuple:
        """Generate GradCAM heatmap for an image.

        Args:
            image: HWC uint8 input image.
            target_class: 1 for failure attention (default), 0 for success.

        Returns:
            (heatmap, overlay): heatmap is (H, W) float32 in [0, 1],
                overlay is (H, W, 3) uint8 with heatmap blended on image.
        """
        torch = self._torch

        self.model.eval()
        tensor = self.predictor.preprocess(image).unsqueeze(0).to(
            self.predictor.device)
        tensor.requires_grad_(True)

        # Forward pass
        logit = self.model(tensor)

        # Backward pass for target class
        self.model.zero_grad()
        if target_class == 1:
            logit.backward()  # gradient w.r.t. failure logit
        else:
            (-logit).backward()  # gradient w.r.t. success

        # Compute GradCAM
        gradients = self._gradients  # (1, C, H', W')
        activations = self._activations  # (1, C, H', W')

        # Global average pool gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = torch.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        h, w = image.shape[:2]
        heatmap = cv2.resize(cam, (w, h))

        # Create overlay
        colormap = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, colormap, 0.4, 0)

        return heatmap, overlay

    def generate_batch(self, images: list) -> list:
        """Generate GradCAM for multiple images.

        Returns list of (heatmap, overlay) tuples.
        """
        return [self.generate(img) for img in images]
