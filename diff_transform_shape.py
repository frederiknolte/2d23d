import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import torch.nn as nn
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
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    PointLights, TexturesVertex,
)
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from renderer import LayeredShader
import wandb
from argparse import ArgumentParser


# Set the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def load_object(path):
    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj(path)
    verts = verts.to(device)
    faces = faces_idx.verts_idx.to(device)

    # Scale and center vertices
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]*torch.tensor((0.5, 0.5, 0.5), device=device)  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    # Create a Meshes object for the target. Here we have only one mesh in the batch.
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )
    return mesh


def initialise_renderer(num_views, image_size, sigma, faces_per_pixel):
    R, T = look_at_view_transform(dist=4,
                                  elev=torch.zeros(num_views, device=device),
                                  azim=torch.tensor([360. * i / num_views for i in range(0, num_views)], device=device))

    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    camera = FoVPerspectiveCameras(device=device, R=R[None, 0, ...], T=T[None, 0, ...])

    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=False,
        bin_size=0
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device, location=cameras.get_camera_center())
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings_soft
        ),
        shader=LayeredShader(device=device, cameras=camera, lights=lights)
    )

    return phong_renderer, cameras


def render(renderer, mesh, cameras):
    # Render the mesh providing the values of R and T.
    image_ref = renderer(meshes_world=mesh.extend(len(cameras)), cameras=cameras)
    image_ref = image_ref.permute(0, 2, 3, 1)[:, :, :, :3]
    return image_ref


@torch.no_grad()
def render_nograd(renderer, mesh, cameras):
    return render(renderer, mesh, cameras)


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref, cameras, optimisation, loss_weights):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.cameras = cameras
        self.optimisation = optimisation
        self.loss_weights = loss_weights

        # Get the reference/target depth image
        image_ref = image_ref.clone()
        self.register_buffer('image_ref', image_ref)
        self.dimg, self.tmesh = None, None

        if self.optimisation == 'RT':
            # Create an optimizable parameter for the x, y, z translation of the mesh.
            self.translation_params = nn.Parameter(
                torch.from_numpy(np.array([[1.0,0.5,0.5]], dtype=np.float32)).to(self.device))
                    # Create an optimizable parameter for the x, y, z translation of the mesh.
            self.rotation_params = nn.Parameter(
                torch.from_numpy(np.array([[1.0,0.0,2.0,0.0]], dtype=np.float32)).to(self.device))
        else:
            self.deform_verts = nn.Parameter(
                torch.full(meshes.verts_packed().shape, 0.0, device=self.device)
            )

        self.loss_fn = nn.MSELoss()

    def forward(self):
        if self.optimisation == 'RT':
            self.RT_loss()
        else:
            self.shape_loss()

        # render mesh with static camera
        self.dimg = render(self.renderer, self.tmesh, self.cameras)

        rgb_loss = torch.sum((self.dimg - self.image_ref) ** 2, dim=[1, 2, 3]).mean()
        laplacian_loss = mesh_laplacian_smoothing(self.tmesh, method="uniform")
        normal_loss = mesh_normal_consistency(self.tmesh)
        edge_loss = mesh_edge_loss(self.tmesh)

        total_loss = rgb_loss + self.loss_weights['w_laplacian'] * laplacian_loss + self.loss_weights['w_normal'] * normal_loss + self.loss_weights['w_edge'] * edge_loss
        losses = {
            'loss/rgb_loss': rgb_loss,
            'loss/laplacian_loss': laplacian_loss,
            'loss/normal_loss': normal_loss,
            'loss/edge_loss': edge_loss,
        }
        return total_loss, self.dimg, losses

    def RT_loss(self):
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

    def shape_loss(self):
        self.tmesh = self.meshes.offset_verts(self.deform_verts)


def plot(control_target_image, control_source_image, optimisation_target_image, optimisation_source_image, show=False):
    max_num_views = max(control_target_image.shape[0], control_source_image.shape[0], optimisation_target_image.shape[0], optimisation_source_image.shape[0])
    fig, axs = plt.subplots(nrows=4, ncols=max_num_views, squeeze=False)
    fig.set_size_inches(max_num_views * 4, 8)
    axs[0, 0].set_ylabel("Control Target")
    axs[1, 0].set_ylabel("Control Source")
    axs[2, 0].set_ylabel("Optimisation Target")
    axs[3, 0].set_ylabel("Optimisation Source")

    def hide_axes(ax):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for i in range(control_target_image.shape[0]):
        axs[0, i].imshow(control_target_image[i])
        hide_axes(axs[0, i])

    for i in range(control_source_image.shape[0]):
        axs[1, i].imshow(control_source_image[i])
        hide_axes(axs[1, i])

    for i in range(optimisation_target_image.shape[0]):
        axs[2, i].imshow(optimisation_target_image[i])
        hide_axes(axs[2, i])

    for i in range(optimisation_source_image.shape[0]):
        axs[3, i].imshow(optimisation_source_image[i])
        hide_axes(axs[3, i])

    plt.tight_layout()
    if show:
        plt.show()
    else:
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        image = data.reshape((int(h), int(w), -1))
        return image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--w_laplacian', type=float, default=0.0)
    parser.add_argument('--w_normal', type=float, default=0.0)
    parser.add_argument('--w_edge', type=float, default=0.0)
    parser.add_argument('--sigma', type=float, default=1e-4)
    parse_config = parser.parse_args()

    # Setup
    config = {
        'source_object_path': './data/other_teapot_scaled.obj',
        'target_object_path': './data/teapot.obj',
        'optimisation': 'shape',
        'num_iter': 500,
        'learning_rate': 0.002,
        'num_views': 1,
        'num_control_views': 1,
        'image_size': 128,
        'control_image_size': 128,
        'faces_per_pixel': 150,
        'control_faces_per_pixel': 150,
        'sigma': parse_config.sigma,
        'control_sigma': 1e-4,
        'log_every': 10,
        'loss_weights': {
            'w_laplacian': parse_config.w_laplacian,
            'w_normal': parse_config.w_normal,
            'w_edge': parse_config.w_edge
        }
    }

    assert config['optimisation'] in ['RT', 'shape']
    assert config['num_control_views'] > 0

    # Initialise wandb
    wandb.init(
        project='2d23d',
        entity='frederiknolte',
        config=config,
        mode='online',
        settings=wandb.Settings(start_method="fork"),
    )

    if wandb.run.sweep_id is not None:
        os.makedirs(f'./output/{wandb.run.sweep_id}', exist_ok=True)
        filename_output = f'./output/{wandb.run.sweep_id}/{wandb.run.id}.mp4'
    else:
        filename_output = f'./output/{wandb.run.id}.mp4'

    # Load objects and initialise renderer
    source_mesh = load_object(config['source_object_path'])
    target_mesh = load_object(config['target_object_path'])

    renderer, cameras = initialise_renderer(num_views=config['num_views'],
                                            image_size=config['image_size'],
                                            sigma=config['sigma'],
                                            faces_per_pixel=config['faces_per_pixel'])
    control_renderer, control_cameras = initialise_renderer(num_views=config['num_control_views'],
                                                            image_size=config['control_image_size'],
                                                            sigma=config['control_sigma'],
                                                            faces_per_pixel=config['control_faces_per_pixel'])

    control_target_image = render_nograd(control_renderer, target_mesh, control_cameras).cpu().numpy()
    optimisation_target_image = render_nograd(renderer, target_mesh, cameras).cpu().numpy()

    # Initialize a model using the renderer, mesh and reference image
    model = Model(meshes=source_mesh,
                  renderer=renderer,
                  image_ref=torch.from_numpy(optimisation_target_image),
                  cameras=cameras,
                  optimisation=config["optimisation"],
                  loss_weights=config['loss_weights']).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    total_loss, dimg, _ = model()
    control_source_image = render_nograd(control_renderer, model.tmesh, control_cameras).cpu().numpy()
    plot(control_target_image, control_source_image, optimisation_target_image, dimg.detach().cpu().numpy(), show=True)

    # Optimisation loop
    loop = tqdm(range(config['num_iter']))
    flattened_target = np.concatenate(list(control_target_image * 255.), axis=1).astype(np.uint8)
    writer = imageio.get_writer(filename_output, format="mp4", mode="I", fps=10)
    total_time = 0
    for i in loop:
        step_start_time = time.time()
        optimizer.zero_grad()
        total_loss, dimg, losses = model()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_finish_time = time.time()
        step_time = step_finish_time - step_start_time
        total_time += step_time

        loop.set_description('Optimizing (loss %.4f)' % total_loss.data)

        # Save outputs to create an mp4.
        optimisation_source_image = dimg.detach().cpu().numpy()
        control_source_image = render_nograd(control_renderer, model.tmesh, control_cameras).cpu().numpy()
        image = plot(control_target_image, control_source_image, optimisation_target_image, optimisation_source_image)
        writer.append_data(image)

        if i % config['log_every'] == 0 or i == (config['num_iter'] - 1):
            log = losses
            log['loss/total_loss'] = total_loss.item()
            log['total_time'] = total_time
            log['step_time'] = step_time
            log['rendering'] = wandb.Image(image)
            log['step'] = i
            wandb.log(log)

    plot(control_target_image, control_source_image, optimisation_target_image, optimisation_source_image, show=True)

    writer.close()
    wandb.finish()
