# mkdir -p data
# wget -P data https://dl.fbaipublicfiles.com/pytorch3d/data/teapot/teapot.obj

# Pytorch3d relevant issues
# https://github.com/facebookresearch/pytorch3d/issues/1087
# https://github.com/facebookresearch/pytorch3d/issues/1386

import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
import pytorch3d
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate,quaternion_to_matrix

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    SoftPhongShader, SoftGouraudShader
)
from renderer import LayeredShader


# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj("./data/teapot.obj")
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]*torch.tensor((0.5,0.5,0.5))  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=64,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100, 
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
# raster_settings = RasterizationSettings(
#     image_size=128, 
#     blur_radius=0.0, 
#     faces_per_pixel=10, 
#     bin_size=None
# )
sigma = 1e-4
raster_settings_soft = RasterizationSettings(
    image_size=64, 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
    faces_per_pixel=50, 
    perspective_correct=False,
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings_soft
    ),
    shader=LayeredShader(device=device, cameras=cameras, lights=lights)
)

# Select the viewpoint using spherical angles  
distance = 3   # distance from camera to the object
elevation = 50.0   # angle of elevation in degrees
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R = torch.eye(3, device=device).unsqueeze(dim=0)
T = torch.tensor([0,0,4], device=device).unsqueeze(dim=0)

# Render the teapot providing the values of R and T. 
silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)


silhouette = silhouette.cpu().numpy()
image_ref = image_ref.squeeze().permute(1, 2, 0)[:,:,:3].cpu().numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(silhouette.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
plt.grid(False)
plt.subplot(1, 2, 2)
plt.imshow(image_ref)
plt.grid(False)
plt.show()

class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the reference/target depth image
        image_ref = image_ref.clone()
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z translation of the mesh. 
        self.translation_params = nn.Parameter(
            torch.from_numpy(np.array([[1.0,0.5,0.5]], dtype=np.float32)).to(self.device))
                # Create an optimizable parameter for the x, y, z translation of the mesh. 
        self.rotation_params = nn.Parameter(
            torch.from_numpy(np.array([[1.0,0.2,0.2,0.2]], dtype=np.float32)).to(self.device))

        self.loss_fn = nn.HuberLoss()
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()

    def forward(self):
        trans = Translate(self.translation_params, device=self.device)
        rot = Rotate(quaternion_to_matrix(self.rotation_params), device=self.device)
        rverts = rot.transform_points(self.meshes.verts_list()[0])
        tverts = trans.transform_points(rverts)
        faces = self.meshes.faces_list()[0]
        
        self.tmesh = pytorch3d.structures.Meshes(
            verts=[tverts.to(self.device)],   
            faces=[faces.to(self.device)],
            textures=self.meshes.textures
        )

        # render mesh with static camera
        R = torch.eye(3, device=self.device).unsqueeze(dim=0)
        T = torch.tensor([0,0,4], device=self.device).unsqueeze(dim=0)
        fragments = self.renderer(meshes_world=self.tmesh, R=R, T=T)
        self.dimg = fragments.squeeze().permute(1, 2, 0)[:,:,:3]


        # loss = self.loss_fn(self.dimg, self.image_ref)
        loss = torch.sum((self.dimg - self.image_ref) ** 2)
        return loss, self.dimg


# We will save images periodically and compose them into a GIF.
filename_output = "./teapot_optimization_demo.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=teapot_mesh, renderer=phong_renderer, image_ref=torch.tensor(image_ref)).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


plt.figure(figsize=(10, 10))

_, image_init = model()
plt.subplot(1, 2, 1)
plt.imshow(model.dimg.detach().cpu().numpy())
plt.grid(False)
plt.title("Starting position")

plt.subplot(1, 2, 2)
plt.imshow(model.image_ref.cpu().numpy())
plt.grid(False)
plt.title("Reference silhouette")
plt.show()

loop = tqdm(range(100))
for i in loop:
    print('i: ', i)
    optimizer.zero_grad()
    loss, _ = model()
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    optimizer.step()
    with torch.no_grad():
        print(model.translation_params.grad)
        print(model.rotation_params.grad)
    
    loop.set_description('Optimizing (loss %.4f)' % loss.data)
    
    # if loss.item() < 200:
    #     break
    print('loss', loss.item())
    
    # Save outputs to create a GIF. 
R = torch.eye(3, device=device).unsqueeze(dim=0)
T = torch.tensor([0,0,4], device=device).unsqueeze(dim=0)
image = phong_renderer(meshes_world=model.tmesh.clone(), R=R, T=T)
image = image[0, ..., :3].detach().squeeze().cpu().numpy()
image = img_as_ubyte(image)
writer.append_data(image)


_, image_init = model()
plt.subplot(1, 2, 1)
plt.imshow(model.dimg.detach().squeeze().cpu().numpy())
plt.grid(False)
plt.title("End position")

plt.subplot(1, 2, 2)
plt.imshow(model.image_ref.cpu().numpy().squeeze())
plt.grid(False)
plt.title("Reference silhouette")
plt.show()

print(model.translation_params)
    
writer.close()