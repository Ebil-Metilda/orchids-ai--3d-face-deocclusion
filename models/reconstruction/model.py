from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from pytorch3d.renderer import (
        AmbientLights,
        FoVPerspectiveCameras,
        MeshRasterizer,
        MeshRenderer,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import ico_sphere

    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False


class ConvAutoencoderPrior(nn.Module):
    """Fallback learned prior used when PyTorch3D is unavailable."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


class Simplified3DMMPrior(nn.Module):
    """Compact 3DMM-style prior using PyTorch3D mesh deformation + rendering."""

    def __init__(self, image_size: int = 128) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.coeff_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 8),
        )
        self.fallback = ConvAutoencoderPrior(channels=3)

    def _render_mesh(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        if not _PYTORCH3D_AVAILABLE:
            raise RuntimeError("PyTorch3D is not available")

        device = x.device
        batch_size = x.shape[0]

        base_mesh = ico_sphere(2, device)
        base_verts = base_mesh.verts_packed()
        base_faces = base_mesh.faces_packed()

        verts_list = []
        faces_list = []
        textures_list = []

        for idx in range(batch_size):
            c = coeffs[idx]
            verts = base_verts.clone()

            # Coefficients deform global shape and z-depth to mimic face geometry changes.
            scale = 0.85 + 0.15 * torch.sigmoid(c[0])
            x_stretch = 0.9 + 0.25 * torch.sigmoid(c[1])
            y_stretch = 0.9 + 0.25 * torch.sigmoid(c[2])
            jaw_shift = 0.05 * torch.tanh(c[3])
            forehead_shift = 0.05 * torch.tanh(c[4])

            verts = verts * scale
            verts[:, 0] = verts[:, 0] * x_stretch
            verts[:, 1] = verts[:, 1] * y_stretch

            jaw_mask = verts[:, 1] < -0.15
            forehead_mask = verts[:, 1] > 0.25
            verts[jaw_mask, 2] = verts[jaw_mask, 2] + jaw_shift
            verts[forehead_mask, 2] = verts[forehead_mask, 2] + forehead_shift

            color = torch.tensor([0.78, 0.64, 0.58], device=device)
            color = color + 0.08 * torch.tanh(c[5:8])
            color = torch.clamp(color, 0.0, 1.0)
            textures = color.view(1, 1, 3).repeat(1, verts.shape[0], 1)

            verts_list.append(verts)
            faces_list.append(base_faces)
            textures_list.append(textures)

        mesh = Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=TexturesVertex(verts_features=textures_list),
        )

        cameras = FoVPerspectiveCameras(device=device)
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = AmbientLights(device=device)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )

        rendered = renderer(mesh)[..., :3]
        rendered = rendered.permute(0, 3, 1, 2)
        if rendered.shape[-1] != x.shape[-1] or rendered.shape[-2] != x.shape[-2]:
            rendered = F.interpolate(
                rendered,
                size=(x.shape[-2], x.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        return torch.clamp(rendered, 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _PYTORCH3D_AVAILABLE:
            return self.fallback(x)

        coeffs = self.coeff_head(x)
        try:
            return self._render_mesh(x, coeffs)
        except Exception:
            return self.fallback(x)


class ReconstructionPriorNet(nn.Module):
    """Configurable reconstruction prior with optional PyTorch3D backend."""

    def __init__(
        self,
        channels: int = 3,
        use_pytorch3d: bool = False,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.use_pytorch3d = bool(use_pytorch3d)
        if self.use_pytorch3d:
            self.impl: nn.Module = Simplified3DMMPrior(image_size=image_size)
        else:
            self.impl = ConvAutoencoderPrior(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


def pytorch3d_is_available() -> bool:
    return _PYTORCH3D_AVAILABLE
