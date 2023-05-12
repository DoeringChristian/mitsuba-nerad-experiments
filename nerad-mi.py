import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mitsuba.python.ad.integrators.common import ADIntegrator, mis_weight

from tqdm import tqdm

scene_dict = mi.cornell_box()
scene = mi.load_dict(scene_dict)
original_image = mi.render(scene, spp=16)

fig, axs = plt.subplots()
fig.patch.set_visible(False)
axs.axis("off")
fig.tight_layout()
axs.imshow(mi.util.convert_to_bitmap(original_image))
# plt.show()

scene_dict["glass"] = {"type": "conductor"}
small_box = scene_dict.pop("small-box")
small_box["bsdf"]["id"] = "glass"
scene_dict["small-box"] = small_box

scene: mi.Scene = mi.load_dict(scene_dict)
scene = mi.load_file("./data/scenes/cornell-box/scene.xml")
original_image = mi.render(scene, spp=16)

fig, axs = plt.subplots()
fig.patch.set_visible(False)
axs.axis("off")
fig.tight_layout()
axs.imshow(mi.util.convert_to_bitmap(original_image))
# plt.show()

m_area = []
for shape in scene.shapes():
    if not shape.is_emitter() and mi.has_flag(
        shape.bsdf().flags(), mi.BSDFFlags.Smooth
    ):
        m_area.append(shape.surface_area())
    else:
        m_area.append([0.0])

m_area = np.array(m_area)[:, 0]

if len(m_area):
    m_area /= m_area.sum()
else:
    raise Warning("No smooth shape. No need of neural network training.")

print(m_area, "\n")
print("Print discared surfaces:")
for i, area in enumerate(m_area):
    if area == 0:
        print("index: ", i)
        print("emitter?: ", scene.shapes()[i].is_emitter())
        print("bsdf: ", scene.shapes()[i].bsdf(), "\n")

shape_sampler = mi.DiscreteDistribution(m_area)


def sample_si(
    scene: mi.Scene,
    shape_sampler: mi.DiscreteDistribution,
    sample1,
    sample2,
    sample3,
    active=True,
):
    """Sample a batch of surface interactions with bsdfs.

    Args:
        scene (mitsuba.Scene): the underlying scene
        shape_sampler (mi.DiscreteDistribution): a source of random numbers for shape sampling
        sample1 (drjit.llvm.ad.Float): determines mesh surfaces
        sample2 (mitsuba.Point2f): determines positions on the meshes
        sample3 (mitsuba.Point2f): determines directions at the positions
        active (bool, optional): mask to specify active lanes. Defaults to True.

    Returns:
        tuple of (mitsuba.SurfaceInteraction3f, mitsuba.BSDF)
    """
    shape_indices = shape_sampler.sample(sample1, active)
    shape: mi.Shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_indices, active)

    ps = shape.sample_position(0.5, sample2, active)
    si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
    si.shape = shape
    bsdf = shape.bsdf()

    active_two_sided = mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
    si.wi = dr.select(
        active_two_sided,
        mi.warp.square_to_uniform_sphere(sample3),
        mi.warp.square_to_uniform_hemisphere(sample3),
    )

    return si, bsdf


l_sampler = mi.load_dict({"type": "independent", "sample_count": 1})
r_sampler = mi.load_dict({"type": "independent", "sample_count": 1})

batch_size = 2**14
M = 32
total_steps = 1000
lr = 5e-4
seed = 42


# helper functions for visualization
def world_to_ndc(scene, batch):
    """Transforms 3D world coordinates into normalized device coordinates (NDC) using the perspective transformation matrix.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.

    Returns:
        mi.Point3f: Array of 3D points in NDC.
    """
    sensor = mi.traverse(scene.sensors()[0])
    trafo = mi.Transform4f.perspective(
        fov=sensor["x_fov"], near=sensor["near_clip"], far=sensor["far_clip"]
    )
    pts = trafo @ sensor["to_world"].inverse() @ mi.Point3f(np.array(batch))
    return pts


def ndc_to_pixel(pts, h, w):
    """Converts points in NDC to pixel coordinates.

    Args:
        pts (mi.Point2f): Points in NDC.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given points.
    """
    hh, hw = h / 2, w / 2
    return mi.Point2f(dr.fma(pts.x, -hw, hw), dr.fma(pts.y, -hw, hh))  # not typo


def draw_multi_segments(starts, ends, color):
    """Draws multiple line segments on a plot.

    Args:
        starts (mi.Point2f): Starting points of the line segments.
        ends (mi.Point2f): Ending points of the line segments.
        color (str): Color of the line segments.
    """
    a = np.c_[starts.x, starts.y]
    b = np.c_[ends.x, ends.y]
    plt.plot(*np.c_[a, b, a * np.nan].reshape(-1, 2).T, color)


def pix_coord(scene, batch, h, w):
    """Calculates the pixel coordinates of the given 3D world coordinates.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given 3D world coordinates.
    """
    return ndc_to_pixel(world_to_ndc(scene, batch), h, w)


def draw_si(si, ax, original_image, marker=".", color="red"):
    """Draws the surface interaction on a plot.

    Args:
        si (mi.SurfaceInteraction): Surface interaction to be drawn.
        ax (matplotlib.axes.Axes): The axes object to plot the surface interaction on.
        original_image (np.ndarray): The original image to plot on the axes.
        marker (str): Marker style for the plot.
        color (str): Color for the plot.
    """
    h, w = original_image.shape[0], original_image.shape[1]
    x_pix = pix_coord(scene, si.p, h, w)
    wi_pix = pix_coord(scene, dr.fma(si.to_world(si.wi), 0.25, si.p), h, w)
    n_pix = pix_coord(scene, dr.fma(si.n, 0.25, si.p), h, w)

    # draw directions
    draw_multi_segments(x_pix, n_pix, "green")
    draw_multi_segments(x_pix, wi_pix, "magenta")

    ax.scatter(x_pix.x, x_pix.y, marker=marker, color=color)
    plt.scatter(n_pix.x, n_pix.y, marker=".", color="green")
    plt.scatter(wi_pix.x, wi_pix.y, marker="x", color="magenta")
    ax.imshow(np.clip(original_image ** (1.0 / 2.2), 0, 1))


def draw_moving_arrow(si, h, w, si_prev, ax):
    """Draws a moving arrow on a plot to represent the surface interaction movement.

    Args:
        si (mi.SurfaceInteraction): Surface interaction to be drawn.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.
        si_prev (mi.SurfaceInteraction): Previous surface interaction to connect to the current one.
        ax (matplotlib.axes.Axes): The axes object to plot the moving arrow on.
    """
    x_pix = pix_coord(scene, si.p, h, w)
    x_prev_pix = pix_coord(scene, si_prev.p, h, w)
    draw_multi_segments(x_prev_pix, x_pix, "green")


# sample only on the seventh shape, i.e., the glass box
tmp_area = [0] * len(m_area)
tmp_area[-1] = 1
tmp_shape_sampler = mi.DiscreteDistribution(tmp_area)

l_sampler = l_sampler.clone()
l_sampler.seed(seed, 5)  # sample five points

si, bsdf = sample_si(
    scene,
    tmp_shape_sampler,
    l_sampler.next_1d(),
    l_sampler.next_2d(),
    l_sampler.next_2d(),
)

fig, ax = plt.subplots()
fig.patch.set_visible(False)  # Hide the figure's background
ax.axis("off")  # Remove the axes from the image
fig.tight_layout()  # Remove any extra white spaces around the image
draw_si(si, ax, original_image, marker="o", color="cyan")
# plt.show()

from tinycudann import Encoding as NGPEncoding


class NRField(nn.Module):
    def __init__(self, bb_min, bb_max) -> None:
        """Initialize an instance of NRField.

        Args:
            bb_min (mi.ScalarBoundingBox3f): minimum point of the bounding box
            bb_max (mi.ScalarBoundingBox3f): maximum point of the bounding box
        """
        super().__init__()
        self.bb_min = bb_min
        self.bb_max = bb_max

        enc_config = {
            "base_resolution": 16,
            "n_levels": 8,
            "n_features_per_level": 4,
            "log2_hashmap_size": 22,
        }
        self.pos_enc = NGPEncoding(3, enc_config)

        in_features = 3 * 4 + self.pos_enc.n_output_dims
        n_neurons = 256
        layers = [  # two hidden layers
            torch.nn.Linear(in_features, n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, n_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_neurons, 3),
        ]
        self.rgb_net = torch.nn.Sequential(*layers)

    def forward(self, si: mi.SurfaceInteraction3f, bsdf: mi.BSDF):
        """Forward pass for NRField.

        Args:
            si (mitsuba.SurfaceInteraction3f): surface interaction
            bsdf (mitsuba.BSDF): bidirectional scattering distribution function

        Returns:
            torch.Tensor
        """
        with dr.suspend_grad():
            x = ((si.p - self.bb_min) / (self.bb_max - self.bb_min)).torch()
            wi = si.to_world(si.wi).torch()
            n = si.sh_frame.n.torch()
            f_d = bsdf.eval_diffuse_reflectance(si).torch()

        z_x = self.pos_enc(x)

        inp = torch.concat([x, wi, n, f_d, z_x], dim=1)
        out = self.rgb_net(inp)
        out = torch.abs(out)
        return out.to(torch.float32)


def get_camera_first_bounce(scene):
    cam_origin = mi.Point3f(0, 1, 3)
    cam_dir = dr.normalize(mi.Vector3f(0, -0.5, -1))
    cam_width = 2.0
    cam_height = 2.0
    image_res = [256, 256]

    x, y = dr.meshgrid(
        dr.linspace(mi.Float, -cam_width / 2, cam_width / 2, image_res[0]),
        dr.linspace(mi.Float, -cam_height / 2, cam_height / 2, image_res[1]),
    )
    ray_origin_local = mi.Vector3f(x, y, 0)
    ray_origin = mi.Frame3f(cam_dir).to_world(ray_origin_local) + cam_origin
    ray = mi.Ray3f(o=ray_origin, d=cam_dir)
    si = scene.ray_intersect(ray)

    return si, image_res


field = NRField(scene.bbox().min, scene.bbox().max).cuda()
si, image_res = get_camera_first_bounce(scene)
print(f"{field(si, si.bsdf())=}")


def render_lhs(scene, model, si, bsdf):
    """
    Renders the left-hand side of the rendering equation by calculating the emitter's radiance and
    the neural network output at the given surface interaction (si) position and direction (bsdf).

    Args:
        scene (mi.Scene): A Mitsuba scene object.
        model (torch.nn.Module): A neural network model that takes si and bsdf as input and returns
            a predicted radiance value.
        si (mi.SurfaceInteraction3f): A Mitsuba surface interaction object.
        bsdf (mi.BSDF): A Mitsuba BSDF object.

    Returns:
        tuple: A tuple containing four values:
            - L (mi.Spectrum): The total outgoing radiance value.
            - Le (mi.Spectrum): The emitter's radiance.
            - out (torch.Tensor): The neural network's predicted radiance value.
            - mask (torch.Tensor): A boolean tensor indicating which surface interactions are valid.
    """
    with dr.suspend_grad():
        Le = si.emitter(scene).eval(si)

        # discard the null bsdf backside
        null_face = ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (si.wi.z < 0)
        mask = si.is_valid() & ~null_face

        out = model(si, bsdf)
        L = Le + dr.select(mask, mi.Spectrum(out), 0)

    return L, Le, out, mask


def first_non_specular_or_null_si(scene, si, bsdf, sampler):
    """Find the first non-specular or null surface interaction.

    Args:
        scene (mi.Scene): Scene object.
        si (mi.SurfaceInteraction3f): Surface interaction.
        bsdf (mi.BSDF): BSDF object.
        sampler (mi.Sampler): Sampler object.

    Returns:
        tuple: A tuple containing four values:
            - si (mi.SurfaceInteraction3f): First non-specular or null surface interaction.
            - bsdf (mi.BSDF): Corresponding BSDF.
            - β (mi.Spectrum): The product of the weights of all previous BSDFs.
            - null_face (bool): A boolean mask indicating whether the surface is a null face or not.
    """
    # Instead of `bsdf.flags()`, based on `bsdf_sample.sampled_type`.
    with dr.suspend_grad():
        bsdf_ctx = mi.BSDFContext()

        depth = mi.UInt32(0)
        β = mi.Spectrum(1)
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        null_face = ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (si.wi.z < 0)
        active = si.is_valid() & ~null_face  # non-null surface
        active &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)  # Delta surface

        loop = mi.Loop(
            name="first_non_specular_or_null_si",
            state=lambda: (sampler, depth, β, active, null_face, si, bsdf),
        )
        loop.set_max_iterations(6)

        while loop(active):
            # loop invariant: si is located at non-null and Delta surface
            # if si is located at null or Smooth surface, end loop
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0)
            )
            bsdf = si.bsdf(ray)

            β *= bsdf_weight
            depth[si.is_valid()] += 1

            null_face &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (
                si.wi.z < 0
            )
            active &= si.is_valid() & ~null_face
            active &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

    # return si at the first non-specular bounce or null face
    return si, bsdf, β, null_face


class LHSIntegrator(ADIntegrator):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def sample(
        self,
        mode,
        scene,
        sampler,
        ray,
        depth,
        reparam,
        active,
        **kwargs,
    ):
        self.model.eval()
        with torch.no_grad():
            w, h = list(scene.sensors()[0].film().size())
            L = mi.Spectrum(0)

            ray = mi.Ray3f(dr.detach(ray))
            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0)
            )
            bsdf = si.bsdf(ray)

            # update si and bsdf with the first non-specular ones
            si, bsdf, β, _ = first_non_specular_or_null_si(scene, si, bsdf, sampler)
            L, _, _, _ = render_lhs(scene, self.model, si, bsdf)

        self.model.train()
        torch.cuda.empty_cache()
        return β * L, si.is_valid(), None


lhs_integrator = LHSIntegrator(field)
lhs_image = mi.render(scene, spp=4, integrator=lhs_integrator)

fig, ax = plt.subplots()
fig.patch.set_visible(False)  # Hide the figure's background
ax.axis("off")  # Remove the axes from the image
fig.tight_layout()  # Remove any extra white spaces around the image
ax.imshow(np.clip(lhs_image ** (1.0 / 2.2), 0, 1))


def render_rhs(scene, model, si, bsdf, sampler):
    with dr.suspend_grad():
        bsdf_ctx = mi.BSDFContext()

        depth = mi.UInt32(0)
        L = mi.Spectrum(0)
        β = mi.Spectrum(1)
        η = mi.Float(1)
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        Le = β * si.emitter(scene).eval(si)

        # emitter sampling
        active_next = si.is_valid()
        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )
        active_em &= dr.neq(ds.pdf, 0.0)

        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        Lr_dir = β * mis_em * bsdf_value_em * em_weight

        # bsdf sampling
        bsdf_sample, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next
        )

        # update
        L = L + Le + Lr_dir
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        η *= bsdf_sample.eta
        β *= bsdf_weight

        prev_si = dr.detach(si, True)
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True)
        bsdf = si.bsdf(ray)

        si, bsdf, β2, null_face = first_non_specular_or_null_si(
            scene, si, bsdf, sampler
        )
        β *= β2

        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        mis = mis_weight(
            prev_bsdf_pdf,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta),
        )

        L += β * mis * si.emitter(scene).eval(si)

        out = model(si, si.bsdf(ray))
        active_nr = (
            si.is_valid()
            & ~null_face
            & dr.eq(si.emitter(scene).eval(si), mi.Spectrum(0))
        )

        Le = L
        w_nr = β * mis
        L = Le + dr.select(active_nr, w_nr * mi.Spectrum(out), 0)

    return L, Le, out, w_nr, active_nr


class RHSIntegrator(ADIntegrator):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def sample(
        self,
        mode,
        scene,
        sampler,
        ray,
        depth,
        reparam,
        active,
        **kwargs,
    ):
        self.model.eval()
        with torch.no_grad():
            w, h = list(scene.sensors()[0].film().size())
            L = mi.Spectrum(0)

            ray = mi.Ray3f(dr.detach(ray))
            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0)
            )
            bsdf = si.bsdf(ray)

            # update si and bsdf with the first non-specular ones
            si, bsdf, β, _ = first_non_specular_or_null_si(scene, si, bsdf, sampler)
            L, _, _, _, _ = render_rhs(scene, self.model, si, bsdf, sampler)

        self.model.train()
        torch.cuda.empty_cache()
        return β * L, si.is_valid(), None


rhs_integrator = RHSIntegrator(field)
rhs_image = mi.render(scene, spp=M, integrator=rhs_integrator)

fig, ax = plt.subplots()
fig.patch.set_visible(False)  # Hide the figure's background
ax.axis("off")  # Remove the axes from the image
fig.tight_layout()  # Remove any extra white spaces around the image
ax.imshow(np.clip(rhs_image ** (1.0 / 2.2), 0, 1))

optimizer = torch.optim.Adam(field.parameters(), lr=lr)
train_losses = []
tqdm_iterator = tqdm(range(total_steps))

field.train()
for step in tqdm_iterator:
    optimizer.zero_grad()

    # detach the computation graph of samplers to avoid lengthy graph of dr.jit
    _l_sampler = l_sampler.clone()
    _l_sampler.seed(step, batch_size)
    _r_sampler = r_sampler.clone()
    _r_sampler.seed(step, batch_size * M // 2)

    si_lhs, bsdf_lhs = sample_si(
        scene,
        shape_sampler,
        _l_sampler.next_1d(),
        _l_sampler.next_2d(),
        _l_sampler.next_2d(),
    )

    # copy `si_lhs` M//2 times for RHS evaluation
    indices = dr.arange(mi.UInt, 0, batch_size)
    indices = dr.repeat(indices, M // 2)
    si_rhs = dr.gather(type(si_lhs), si_lhs, indices)
    bsdf_rhs = dr.gather(type(bsdf_lhs), bsdf_lhs, indices)

    # LHS and RHS evaluation
    _, Le_lhs, out_lhs, mask_lhs = render_lhs(scene, field, si_lhs, bsdf_lhs)
    _, Le_rhs, out_rhs, weight_rhs, mask_rhs = render_rhs(
        scene, field, si_rhs, bsdf_rhs, _r_sampler
    )
    weight_rhs = weight_rhs.torch() * mask_rhs.torch()

    lhs = Le_lhs.torch() + out_lhs * mask_lhs.torch().reshape(-1, 1)
    rhs = Le_rhs.torch() + out_rhs * weight_rhs
    rhs = rhs.reshape(batch_size, M // 2, 3).mean(dim=1)

    norm = 1
    # in our experiment, normalization makes rendering biased (dimmer)
    # norm = (lhs + rhs).detach()/2 + 1e-2

    loss = torch.nn.MSELoss()(lhs / norm, rhs / norm)
    loss.backward()
    optimizer.step()

    tqdm_iterator.set_description("Loss %.04f" % (loss.item()))
    train_losses.append(loss.item())
field.eval()

lhs_integrator = LHSIntegrator(field)
lhs_image = mi.render(scene, spp=16, integrator=lhs_integrator)

fig, ax = plt.subplots()
fig.patch.set_visible(False)  # Hide the figure's background
ax.axis("off")  # Remove the axes from the image
fig.tight_layout()  # Remove any extra white spaces around the image
ax.imshow(np.clip(lhs_image ** (1.0 / 2.2), 0, 1))

plt.show()
