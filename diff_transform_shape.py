import time
import torch
import numpy as np
from tqdm import tqdm
import imageio
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
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
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


def plot(source_image, target_image, num_views):
    # Plot objects from differen viewpoints
    fig, axs = plt.subplots(nrows=num_views, ncols=2, sharex=True, sharey=True, squeeze=False)
    fig.set_size_inches(8, num_views * 4)
    axs[0, 0].set_title("Source")
    axs[0, 1].set_title("Target")

    for i in range(num_views):
        axs[i, 0].imshow(source_image[i])
        axs[i, 1].imshow(target_image[i])
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


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

        loss = rgb_loss + self.loss_weights['w_laplacian'] * laplacian_loss + self.loss_weights['w_normal'] * normal_loss
        return loss, self.dimg, rgb_loss

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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--w_laplacian', type=float, default=0.0)
    parser.add_argument('--w_normal', type=float, default=0.0)
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
        'image_size': 128,
        'faces_per_pixel': 150,
        'sigma': parse_config.sigma,
        'log_every': 10,
        'loss_weights': {
            'w_laplacian': parse_config.w_laplacian,
            'w_normal': parse_config.w_normal,
        }
    }

    assert config['optimisation'] in ['RT', 'shape']

    # Initialise wandb
    import sys
    print(sys.path)
    wandb.init(
        project='2d23d',
        entity='frederiknolte',
        config=config,
        mode='online',
    )

    filename_output = f'./output/{wandb.run.id}.mp4'

    # Load objects and initialise renderer
    source_mesh = load_object(config['source_object_path'])
    target_mesh = load_object(config['target_object_path'])

    renderer, cameras = initialise_renderer(num_views=config['num_views'],
                                            image_size=config['image_size'],
                                            sigma=config['sigma'],
                                            faces_per_pixel=config['faces_per_pixel'])

    target_image_ref = render(renderer, target_mesh, cameras).cpu().numpy()
    source_image_ref = render(renderer, source_mesh, cameras).cpu().numpy()

    # Plot objects from differen viewpoints
    plot(source_image_ref, target_image_ref, config['num_views'])

    # Initialize a model using the renderer, mesh and reference image
    model = Model(meshes=source_mesh,
                  renderer=renderer,
                  image_ref=torch.tensor(target_image_ref),
                  cameras=cameras,
                  optimisation=config["optimisation"],
                  loss_weights=config['loss_weights']).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    loss, dimg, _ = model()
    plot(dimg.detach().cpu().numpy(), target_image_ref, config['num_views'])

    # Optimisation loop
    loop = tqdm(range(config['num_iter']))
    flattened_target = np.concatenate(list(target_image_ref * 255.), axis=1).astype(np.uint8)
    writer = imageio.get_writer(filename_output, format="mp4", mode="I", fps=10)
    total_time = 0
    for i in loop:
        step_start_time = time.time()
        optimizer.zero_grad()
        loss, dimg, rgb_loss = model()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_finish_time = time.time()
        step_time = step_finish_time - step_start_time
        total_time += step_time

        loop.set_description('Optimizing (loss %.4f)' % loss.data)

        # Save outputs to create an mp4.
        image = img_as_ubyte(dimg.detach().cpu().numpy())
        image = np.concatenate(list(image), axis=1)
        image = np.concatenate([flattened_target, image], axis=0)
        writer.append_data(image)

        if i % config['log_every'] == 0 or i == (config['num_iter'] - 1):
            wandb.log({
                'total_time': total_time,
                'step_time': step_time,
                'loss': loss.item(),
                'rgb_loss': rgb_loss.item(),
                'rendering': wandb.Image(image),
                'step': i
            })

    plot(dimg.detach().cpu().numpy(), target_image_ref, config['num_views'])

    writer.close()
    wandb.finish()
