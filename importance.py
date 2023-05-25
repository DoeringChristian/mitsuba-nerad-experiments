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


from tinycudann import Encoding as NGPEncoding


class NRField(nn.Module):
    def __init__(self, scene: mi.Scene, width=256, n_hidden=8) -> None:
        """Initialize an instance of NRField.

        Args:
            bb_min (mi.ScalarBoundingBox3f): minimum point of the bounding box
            bb_max (mi.ScalarBoundingBox3f): maximum point of the bounding box
        """
        super().__init__()
        self.bbox = scene.bbox()

        enc_config = {
            "otype": "HashGrid",
            # "type": "Hash",
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

    def forward(self, si: mi.SurfaceInteraction3f):
        """Forward pass for NRField.

        Args:
            si (mitsuba.SurfaceInteraction3f): surface interaction
            bsdf (mitsuba.BSDF): bidirectional scattering distribution function

        Returns:
            torch.Tensor
        """
        with dr.suspend_grad():
            x = ((si.p - self.bbox.min) / (self.bbox.max - self.bbox.min)).torch()
            wi = si.to_world(si.wi).torch()
            n = si.sh_frame.n.torch()
            f_d = si.bsdf().eval_diffuse_reflectance(si).torch()

        z_x = self.pos_enc(x)

        inp = torch.concat([x, wi, n, f_d, z_x], dim=1)
        out = self.network(inp)
        out = torch.abs(out)
        return out.to(torch.float32)


class NeradIntegrator(mi.SamplingIntegrator):
    def __init__(self, model) -> None:
        super().__init__(mi.Properties())
        self.model = model

        self.l_sampler = mi.load_dict({"type": "independent", "sample_count": 1})
        self.r_sampler = mi.load_dict({"type": "independent", "sample_count": 1})

        self.M = 32
        self.batch_size = 2**14
        self.total_steps = 100
        # self.lr = 5e-4
        self.lr = 5e-4
        self.seed = 42
        self.sample_mode = "lhs"

    def sample_si(
        self,
        scene: mi.Scene,
        shape_sampler: mi.DiscreteDistribution,
        sampler: mi.Sampler,
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
        sample_emitter = ~self.si.is_valid()

        ray, _, _ = scene.sample_emitter_ray(
            0.0, sampler.next_1d(), sampler.next_2d(), sampler.next_2d(), sample_emitter
        )
        bsdf: mi.BSDF = self.si.bsdf()
        bsdf_sample, _ = bsdf.sample(
            mi.BSDFContext(),
            self.si,
            sampler.next_1d(),
            sampler.next_2d(),
            ~sample_emitter,
        )
        ray[~sample_emitter] = self.si.spawn_ray(self.si.to_world(bsdf_sample.wo))

        self.si = scene.ray_intersect(ray)

        random_pos = sampler.next_1d() < 0.5
        shape_idx = shape_sampler.sample(sampler.next_1d(), random_pos)
        shape: mi.Shape = dr.gather(
            mi.ShapePtr, scene.shapes_dr(), shape_idx, random_pos
        )
        ps = shape.sample_position(0.0, sampler.next_2d(), random_pos)
        self.si[random_pos] = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
        self.si.shape[random_pos] = shape

        active_two_sided = mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
        self.si.wi = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere(sampler.next_2d()),
            mi.warp.square_to_uniform_hemisphere(sampler.next_2d()),
        )

        return self.si

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
        max_iterations = 6
        # Instead of `bsdf.flags()`, based on `bsdf_sample.sampled_type`.
        with dr.suspend_grad():
            bsdf_ctx = mi.BSDFContext()

            depth = mi.UInt32(0)
            β = mi.Spectrum(1)

            null_face = mi.Bool(True)
            active = mi.Bool(True)

            loop = mi.Loop(
                name="first_non_specular_or_null_si",
                state=lambda: (sampler, depth, β, active, null_face, si),
            )
            loop.set_max_iterations(max_iterations)

            while loop(active):
                # loop invariant: si is located at non-null and Delta surface
                # if si is located at null or Smooth surface, end loop

                bsdf: mi.BSDF = si.bsdf()

                bsdf_sample, bsdf_weight = bsdf.sample(
                    bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
                )

                null_face &= ~mi.has_flag(
                    bsdf_sample.sampled_type, mi.BSDFFlags.BackSide
                ) & (si.wi.z < 0)
                active &= si.is_valid() & ~null_face & (depth < max_iterations)
                active &= ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Smooth)

                ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

                si[active] = scene.ray_intersect(
                    ray,
                    ray_flags=mi.RayFlags.All,
                    coherent=False,
                    active=active,
                )

                β[active] *= bsdf_weight
                depth += 1

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

            out = self.model(si)

        if mode == "drjit":
            return Le + dr.select(mask, mi.Spectrum(out), 0)
        elif mode == "torch":
            # In torch mode returns only the network output
            # i.e. N(x, ω_o)
            return out * mask.torch().reshape(-1, 1)

    def render_rhs(
        self, scene: mi.Scene, si: mi.SurfaceInteraction3f, sampler, mode="drjit"
    ) -> mi.Color3d | torch.Tensor:
        with dr.suspend_grad():
            bsdf_ctx = mi.BSDFContext()

            depth = mi.UInt32(0)
            L = mi.Spectrum(0)
            β = mi.Spectrum(1)
            η = mi.Float(1)
            prev_si = dr.zeros(mi.SurfaceInteraction3f)
            prev_bsdf_pdf = mi.Float(1.0)
            prev_bsdf_delta = mi.Bool(True)

            # Do not use Le1 only render Transmittance
            # L = E + T = E(x, ω_o) + T(E(x', -ω_i) + N(x', -ω_i))
            # L(x, ω_o) = E(x, ω_o) + ∫ L(x'(x, ω_o), ω_i) f(x, ω_o, ω_i) dω_i^⊥ = E(x, ω_o) + N(x', ω_o)
            Le1 = β * si.emitter(scene).eval(si)

            bsdf = si.bsdf()

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
            # L = L + Le + Lr_dir
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True)
            bsdf = si.bsdf(ray)

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis_bsdf = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta),
            )

            si, β2, null_face = self.first_non_specular_or_null_si(scene, si, sampler)
            β *= β2

            # T(E)
            L = β * mis_bsdf * si.emitter(scene).eval(si) + Lr_dir

            out = self.model(si)
            active_nr = (
                si.is_valid()
                & ~null_face
                & dr.eq(si.emitter(scene).eval(si), mi.Spectrum(0))
            )

            w_nr = β * mis_bsdf

        if mode == "drjit":
            # E(x) + T(E(x')) + T(N(x'))
            return Le1 + L + dr.select(active_nr, w_nr * mi.Spectrum(out), 0)

        elif mode == "torch":
            # T(E) + T(N)
            return L.torch() + out * dr.select(active_nr, w_nr, 0).torch()

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

            # update si and bsdf with the first non-specular ones
            si, β, _ = self.first_non_specular_or_null_si(scene, si, sampler)
            if self.sample_mode == "rhs":
                L = self.render_rhs(scene, si, sampler, mode="drjit")
            elif self.sample_mode == "lhs":
                L = self.render_lhs(scene, si, mode="drjit")

        self.model.train()
        torch.cuda.empty_cache()
        return β * L, si.is_valid(), []

    def train(self):
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

        shape_sampler = mi.DiscreteDistribution(m_area)

        optimizer = torch.optim.Adam(field.parameters(), lr=self.lr)
        train_losses = []
        tqdm_iterator = tqdm(range(self.total_steps))

        self.model.train()
        for step in tqdm_iterator:
            optimizer.zero_grad()

            # detach the computation graph of samplers to avoid lengthy graph of dr.jit
            r_sampler = self.r_sampler.clone()
            l_sampler = self.l_sampler.clone()
            r_sampler.seed(step, self.batch_size * self.M // 2)
            l_sampler.seed(step, self.batch_size)
            self.si: mi.SurfaceInteraction3f = dr.zeros(
                mi.SurfaceInteraction3f, self.batch_size
            )

            si_lhs = self.sample_si(
                scene,
                shape_sampler,
                l_sampler,
            )

            # copy `si_lhs` M//2 times for RHS evaluation
            indices = dr.arange(mi.UInt, 0, self.batch_size)
            indices = dr.repeat(indices, self.M // 2)
            si_rhs = dr.gather(type(si_lhs), si_lhs, indices)
            # bsdf_rhs = dr.gather(type(bsdf_lhs), bsdf_lhs, indices)

            # LHS and RHS evaluation
            lhs = self.render_lhs(scene, si_lhs, mode="torch")
            rhs = self.render_rhs(scene, si_rhs, r_sampler, mode="torch")

            rhs = rhs.reshape(self.batch_size, self.M // 2, 3).mean(dim=1)

            norm = 1
            # in our experiment, normalization makes rendering biased (dimmer)
            # norm = (lhs + rhs).detach() / 2 + 1e-2

            loss = torch.nn.MSELoss()(lhs / norm, rhs / norm)
            loss.backward()
            optimizer.step()

            tqdm_iterator.set_description("Loss %.04f" % (loss.item()))
            train_losses.append(loss.item())
        self.model.eval()
        self.train_losses = train_losses


# optimizer = torch.optim.Adam(field.parameters(), lr=lr)
# train_losses = []
# tqdm_iterator = tqdm(range(total_steps))

if __name__ == "__main__":
    scene_dict = mi.cornell_box()

    scene_dict.pop("small-box")
    scene_dict.pop("large-box")
    # scene_dict["sphere"] = {
    #     "type": "sphere",
    #     "to_world": mi.ScalarTransform4f.translate([0.335, 0.0, -0.38]).scale(0.5),
    #     "bsdf": {"type": "plastic"},
    # }

    scene_dict["suzanne"] = {
        "type": "ply",
        "filename": "./data/meshes/suzanne.ply",
        "to_world": mi.ScalarTransform4f.translate([0.335, -0.3, 0.0])
        .rotate([1.0, 0.0, 0.0], 90.0)
        .scale(0.5),
        "bsdf": {"type": "dielectric"},
    }

    scene: mi.Scene = mi.load_dict(scene_dict)
    # scene = mi.load_file("./data/scenes/caustics/scene.xml")
    # scene = mi.load_file("./data/scenes/veach-door/scene.xml")

    field = NRField(scene, n_hidden=3, width=256)
    integrator = NeradIntegrator(field)
    integrator.train()
    image = mi.render(scene, spp=1, integrator=integrator)
    losses_orig = integrator.train_losses

    ref_image = mi.render(scene, spp=16)
    pt_image = mi.render(scene, spp=16, integrator=mi.load_dict({"type": "ptracer"}))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_visible(False)  # Hide the figure's background
    ax[0][0].axis("off")  # Remove the axes from the image
    ax[0][0].imshow(mi.util.convert_to_bitmap(image))
    ax[1][0].axis("off")
    ax[1][0].imshow(mi.util.convert_to_bitmap(ref_image))
    ax[0][1].axis("off")
    ax[0][1].imshow(mi.util.convert_to_bitmap(pt_image))
    ax[1][1].plot(losses_orig, color="red")
    fig.tight_layout()  # Remove any extra white spaces around the image

    plt.show()
