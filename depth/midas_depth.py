import torch
import cv2
from config import Config

class DepthEstimator:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        self.model = torch.hub.load("intel-isl/MiDaS", Config.MIDAS_MODEL)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        # Select correct transform
        if "DPT" in Config.MIDAS_MODEL:
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    @torch.inference_mode()
    def get_depth(self, frame):
        # Convert to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform and send to device
        input_batch = self.transform(img).to(self.device)

        # Prediction
        prediction = self.model(input_batch)

        # Resize to original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize depth
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (1.0 - (depth - d_min) / (d_max - d_min)) * 100.0
        else:
            depth = depth * 0  # fallback

        return depth