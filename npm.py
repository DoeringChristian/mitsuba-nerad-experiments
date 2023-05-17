from typing import Union
import drjit as dr
import mitsuba as mi
import mitsuba

mi.set_variant("cuda_ad_rgb")

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mitsuba.python.ad.integrators.common import ADIntegrator, mis_weight

from tqdm import tqdm

# scene_dict = mi.cornell_box()
#
# scene_dict["glass"] = {"type": "conductor"}
# small_box = scene_dict.pop("small-box")
# small_box["bsdf"]["id"] = "glass"
# scene_dict["small-box"] = small_box
#
# scene: mi.Scene = mi.load_dict(scene_dict)
# scene = mi.load_file("./data/scenes/cornell-box/scene.xml")
scene = mi.load_dict(mi.cornell_box())
# scene = mi.load_file("./data/scenes/veach-ajar/scene.xml")


M = 32
batch_size = 2**14
total_steps = 1000
lr = 5e-4
seed = 42


from tinycudann import Encoding as NGPEncoding


class NRFieldOrig(nn.Module):
    def __init__(self, scene: mi.Scene, width=256, n_hidden=8) -> None:
        """Initialize an instance of NRField.

        Args:
            bb_min (mi.ScalarBoundingBox3f): minimum point of the bounding box
            bb_max (mi.ScalarBoundingBox3f): maximum point of the bounding box
        """
        super().__init__()
        self.bbox = scene.bbox()

        enc_config = {
            "otype": "Grid",
            "type": "Hash",
            "base_resolution": 16,
            "n_levels": 8,
            "n_features_per_level": 4,
            "log2_hashmap_size": 22,
        }
        self.pos_enc = NGPEncoding(3, enc_config)

        in_size = 3 * 4 + self.pos_enc.n_output_dims

        hidden_layers = []
        for _ in range(n_hidden):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(
            nn.Linear(in_size, width),
            nn.ReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, 3),
        ).to("cuda")

    def forward(self, si: mi.SurfaceInteraction3f, wo: mi.Vector3f):
        """Forward pass for NRField.

        Args:
            si (mitsuba.SurfaceInteraction3f): surface interaction
            bsdf (mitsuba.BSDF): bidirectional scattering distribution function

        Returns:
            torch.Tensor
        """
        with dr.suspend_grad():
            x = ((si.p - self.bbox.min) / (self.bbox.max - self.bbox.min)).torch()
            # wi = si.to_world(si.wi).torch()
            wo = si.to_world(wo).torch()
            n = si.sh_frame.n.torch()
            f_d = si.bsdf().eval_diffuse_reflectance(si).torch()

        z_x = self.pos_enc(x)

        inp = torch.concat([x, wo, n, f_d, z_x], dim=1)
        out = self.network(inp)
        out = torch.abs(out)
        return out.to(torch.float32)


class NeradIntegrator(mi.SamplingIntegrator):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(mi.Properties())
        self.model = model

        self.sampler: mi.Sampler = mi.load_dict(
            {"type": "independent", "sample_count": 1}
        )

    def sample_si(
        self,
        scene: mi.Scene,
        shape_sampler: mi.DiscreteDistribution,
        sample1,
        sample2,
        sample3,
        active=True,
    ) -> mi.SurfaceInteraction3f:
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
        shape_index = shape_sampler.sample(sample1, active)
        shape: mi.Shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_index, active)

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
        si.shape = shape

        return si

    def first_non_specular_or_null_si(
        self, scene: mi.Scene, si: mi.SurfaceInteraction3f, sampler: mi.Sampler
    ):
        """Find the first non-specular or null surface interaction.

        Args:
            scene (mi.Scene): Scene object.
            si (mi.SurfaceInteraction3f): Surface interaction.
            sampler (mi.Sampler): Sampler object.

        Returns:
            tuple: A tuple containing four values:
                - si (mi.SurfaceInteraction3f): First non-specular or null surface interaction.
                - β (mi.Spectrum): The product of the weights of all previous BSDFs.
                - null_face (bool): A boolean mask indicating whether the surface is a null face or not.
        """
        # Instead of `bsdf.flags()`, based on `bsdf_sample.sampled_type`.
        with dr.suspend_grad():
            bsdf_ctx = mi.BSDFContext()

            depth = mi.UInt32(0)
            β = mi.Spectrum(1)
            # prev_si = dr.zeros(mi.SurfaceInteraction3f)
            # prev_bsdf_pdf = mi.Float(1.0)
            # prev_bsdf_delta = mi.Bool(True)

            null_face = mi.Bool(True)
            active = mi.Bool(True)

            loop = mi.Loop(
                name="first_non_specular_or_null_si",
                state=lambda: (sampler, depth, β, active, null_face, si),
            )
            loop.set_max_iterations(6)

            while loop(active):
                # for i in range(6):
                # loop invariant: si is located at non-null and Delta surface
                # if si is located at null or Smooth surface, end loop

                bsdf: mi.BSDF = si.bsdf()

                bsdf_sample, bsdf_weight = bsdf.sample(
                    bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
                )
                null_face &= ~mi.has_flag(
                    bsdf_sample.sampled_type, mi.BSDFFlags.BackSide
                ) & (si.wi.z < 0)
                active &= si.is_valid() & ~null_face
                active &= mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Glossy)

                ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

                si[active] = scene.ray_intersect(
                    ray,
                    ray_flags=mi.RayFlags.All,
                    coherent=dr.eq(depth, 0),
                    active=active,
                )

                β[active] *= bsdf_weight
                depth[active] += 1

        # return si at the first non-specular bounce or null face
        return si, β, null_face

    def render_lhs(
        self, scene: mi.Scene, si: mi.SurfaceInteraction3f, mode: str = "drjit"
    ) -> mi.Color3f | torch.Tensor:
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
            null_face = ~mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide) & (
                si.wi.z < 0
            )
            mask = si.is_valid() & ~null_face

            out = self.model(si, si.wi)
            L = Le + dr.select(mask, mi.Spectrum(out), 0)

        if mode == "drjit":
            return Le + dr.select(mask, mi.Spectrum(out), 0)
        elif mode == "torch":
            return Le.torch() + out * mask.torch().reshape(-1, 1)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, mi.Bool, list[mi.Color3f]]:
        self.model.eval()
        with torch.no_grad():
            w, h = list(scene.sensors()[0].film().size())
            L = mi.Spectrum(0)

            ray = mi.Ray3f(dr.detach(ray))
            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True)
            bsdf = si.bsdf(ray)

            # update si and bsdf with the first non-specular ones
            si, β, _ = self.first_non_specular_or_null_si(scene, si, sampler)
            L = self.render_lhs(scene, si, mode="drjit")

        self.model.train()
        torch.cuda.empty_cache()
        return β * L, si.is_valid(), []

    def train(self, scene: mi.Scene):
        self.train_losses = []
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        tqdm_iterator = tqdm(range(total_steps))
        for step in tqdm_iterator:
            opt.zero_grad()

            sampler = self.sampler.clone()
            sampler.seed(step, batch_size)

            # L = mi.Color3f(0.0)
            ray, weight, emitter = scene.sample_emitter_ray(
                0.0, sampler.next_1d(), sampler.next_2d(), sampler.next_2d(), True
            )
            L = mi.Color3f(1.0)
            # print(f"{dr.max_nested(L)=}")
            # print(f"{L=}")

            # L += weight
            # L = mi.Color3f(1.0)
            active = mi.Bool(True)
            bsdf_ctx = mi.BSDFContext()

            local_loss = 0.0
            for i in range(3):
                si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
                active &= si.is_valid()
                bsdf: mi.BSDF = si.bsdf()

                bsdf_sample, bsdf_weight = bsdf.sample(
                    bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
                )

                L *= bsdf_weight
                # print(f"{dr.max_nested(bsdf_weight)=}")

                mask: torch.Tensor = dr.select(active, 1.0, 0.0).torch()
                mask = mask.reshape(-1, 1).repeat(1, 3)
                # print(f"{L.torch().shape=}")
                # print(f"{mask.shape=}")

                L_net = self.model(si, bsdf_sample.wo) * mask
                L_render = L.torch() * mask
                loss = torch.nn.MSELoss()(L_render, L_net)
                loss.backward()
                local_loss += loss.item()

                # L += si.emitter(scene, active).eval(si, active)

                ray = si.spawn_ray_to(si.to_world(bsdf_sample.wo))

            opt.step()

            self.train_losses.append(local_loss)
            tqdm_iterator.set_description(f"Loss {self.train_losses[-1]}")


# optimizer = torch.optim.Adam(field.parameters(), lr=lr)
# train_losses = []
# tqdm_iterator = tqdm(range(total_steps))

field = NRFieldOrig(scene, n_hidden=3)
integrator = NeradIntegrator(field)
integrator.train(scene)
image_orig = mi.render(scene, spp=1, integrator=integrator)
losses_orig = integrator.train_losses

# field = NRFieldSh(scene)
# integrator = NeradIntegrator(field)
# integrator.train()
# image_sh = mi.render(scene, spp=16, integrator=integrator)
# losses_sh = integrator.train_losses

# field = NRFieldSh(scene, wi_order=2)
# integrator = NeradIntegrator(field)
# integrator.train()
# image_sh_2 = mi.render(scene, spp=16, integrator=integrator)
# losses_sh_2 = integrator.train_losses

ref_image = mi.render(scene, spp=16)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.patch.set_visible(False)  # Hide the figure's background
ax[0][0].axis("off")  # Remove the axes from the image
ax[0][0].imshow(mi.util.convert_to_bitmap(image_orig))
# ax[0][1].axis("off")
# ax[0][1].imshow(mi.util.convert_to_bitmap(image_sh))
# ax[0][2].axis("off")
# ax[0][2].imshow(mi.util.convert_to_bitmap(image_sh_2))
ax[1][0].axis("off")
ax[1][0].imshow(mi.util.convert_to_bitmap(ref_image))
ax[1][1].plot(losses_orig, color="red")
# ax[1][1].plot(losses_sh, color="green")
# ax[1][1].plot(losses_sh_2, color="yellow")
fig.tight_layout()  # Remove any extra white spaces around the image

plt.show()
