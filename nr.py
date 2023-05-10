import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tinycudann
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

import mypath


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


def n_sh_coeffs(order) -> int:
    return 2 * order + 1


class RadianceMLP(nn.Module):
    def __init__(self, width, n_hidden, bbox: mi.BoundingBox3f) -> None:
        super().__init__()
        self.bbox = bbox

        enc_config = {
            "base_resolution": 16,
            "n_levels": 8,
            "n_features_per_level": 4,
            "log2_hashmap_size": 22,
        }
        self.pos_enc = tinycudann.Encoding(3, enc_config)

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

        def init_weights(m: nn.Module):
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight)

        self.network.apply(init_weights)
        self.grad_activator = mi.Vector3f(0)

    def forward(self, si: mi.SurfaceInteraction3f):
        with dr.suspend_grad():
            x = ((si.p - self.bbox.min) / (self.bbox.max - self.bbox.min)).torch()
            wi = si.to_world(si.wi).torch()
            n = si.sh_frame.n.torch()
            f_d = si.bsdf().eval_diffuse_reflectance(si).torch()

        z_x = self.pos_enc(x)

        input = torch.concat([x, wi, n, f_d, z_x], dim=1)
        output = torch.abs(self.network(input))
        return output.to(torch.float32)

    def forward_mitsuba(self, si: mi.SurfaceInteraction3f):
        with dr.suspend_grad():
            x = (si.p - self.bbox.min) / (self.bbox.max - self.bbox.min)
            wi = si.to_world(si.wi)
            n = si.sh_frame.n
            f_d = si.bsdf().eval_diffuse_reflectance(si)

        def to_tensor(val):
            return mi.TensorXf(dr.ravel(val), [dr.shape(val)[1], dr.shape(val)[0]])

        x = to_tensor(x + self.grad_activator)
        wi = to_tensor(wi)
        n = to_tensor(n)
        f_d = to_tensor(si.bsdf().eval_diffuse_reflectance(si))

        @dr.wrap_ad("drjit", "torch")
        def internal(x, wi, n, f_d):
            z_x = self.pos_enc(x)
            input = torch.concat([x, wi, n, f_d, z_x], dim=1)
            output = torch.abs(self.network(input))
            return output.to(torch.float32)

        ret = dr.unravel(mi.Color3f, internal(x, wi, n, f_d))
        return ret

    def traverse(self, callback):
        callback.put_parameter(
            "grad_activator", self.grad_activator, mi.ParamFlags.Differentiable
        )


class PathIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth: int = props.get("max_depth", 8)
        self.rr_depth: int = props.get("rr_depth", 2)
        self.n = 0
        self.film_size: None | mi.Vector2u = None
        self.network: nn.Module = None

    def sample_si(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        shape_sampler: mi.DiscreteDistribution,
    ) -> tuple[mi.SurfaceInteraction3f, mi.Ray3f]:
        shape_idx = shape_sampler.sample(sampler.next_1d())
        shape: mi.Shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx, True)

        ps = shape.sample_position(0.0, sampler.next_2d(), True)
        si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
        si.shape = shape

        active_two_sided = mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide)
        sample3 = sampler.next_2d()
        si.wi = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere(sample3),
            mi.warp.square_to_uniform_hemisphere(sample3),
        )
        ray = mi.Ray3f(si.p + si.to_world(si.wi) * 0.001, -si.to_world(si.wi))
        return si, ray

    def train(self, scene: mi.Scene, sampler: mi.Sampler, wavefront_size: int, seed=0):
        bbox = scene.bbox()
        if self.network is None:
            self.network = RadianceMLP(256, 8, bbox)

        m_area = []
        for shape in scene.shapes():
            if not shape.is_emitter() and mi.has_flag(
                shape.bsdf().flags(), mi.BSDFFlags.Smooth
            ):
                m_area.append(shape.surface_area()[0])
            else:
                m_area.append(0.0)

        m_area = np.array(m_area)

        if len(m_area):
            m_area /= m_area.sum()
        else:
            raise Warning("No smooth shape. No need of neural network training!")

        sampler.seed(seed, wavefront_size)
        shape_sampler = mi.DiscreteDistribution(m_area)

        self.network.train()
        opt = torch.optim.Adam(self.network.parameters(), lr=5e-4)

        for i in tqdm(range(200)):
            opt.zero_grad()

            si, ray = self.sample_si(scene, sampler, shape_sampler)

            RHS, _, LHS = self.sample(scene, sampler, ray=ray)

            LHS = mi.Color3f(LHS[0], LHS[1], LHS[2])

            RHS: torch.Tensor = RHS.torch()
            LHS: torch.Tensor = LHS.torch()

            print(f"{RHS.grad=}")

            loss = torch.nn.MSELoss()(RHS, LHS)
            loss.backward()
            opt.step()

        ...

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, mi.Bool, list[mi.Color3f]]:
        # --------------------- Configure loop state ----------------------

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)
        depth = mi.UInt32(0)
        eta = mi.Float(1)
        # L = mi.Color3f(0.0)
        throughput = mi.Spectrum(1)
        valid_ray = dr.neq(scene.environment(), None)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        LHS = dr.select(active, self.network.forward_mitsuba(si), 0)

        # ---------------------- Direct emission ----------------------

        ds = mi.DirectionSample3f(scene, si, prev_si)
        em_pdf = mi.Float(0.0)

        em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

        mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

        E1 = throughput * ds.emitter.eval(si, prev_bsdf_pdf > 0.0) * mis_bsdf

        active_next = ((depth + 1) < self.max_depth) & si.is_valid()

        bsdf: mi.BSDF = si.bsdf(ray)

        # ---------------------- Emitter sampling ----------------------

        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )

        wo = si.to_local(ds.d)

        # ------ Evaluate BSDF * cos(theta) and sample direction -------

        sample1 = sampler.next_1d()
        sample2 = sampler.next_2d()

        bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
            bsdf_ctx, si, wo, sample1, sample2
        )

        # --------------- Emitter sampling contribution ----------------

        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

        mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

        # L = dr.fma(throughput, bsdf_val * em_weight * mis_em)

        # ---------------------- BSDF sampling ----------------------

        bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        eta *= bsdf_sample.eta
        valid_ray |= (
            active
            & si.is_valid()
            & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
        )

        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        depth[si.is_valid()] += 1

        # -------------------- Stopping criterion ---------------------

        # -------------------- RHS ---------------------

        si = scene.ray_intersect(ray)  # TODO: not necesarry in first interaction

        # ---------------------- Direct emission ----------------------

        ds = mi.DirectionSample3f(scene, si, prev_si)
        em_pdf = mi.Float(0.0)

        em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

        mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

        E2 = ds.emitter.eval(si, prev_bsdf_pdf > 0.0)

        active_next = ((depth + 1) < self.max_depth) & si.is_valid()

        bsdf: mi.BSDF = si.bsdf(ray)

        RHS = (
            self.network.forward_mitsuba(si) * mis_bsdf * throughput
            + E2 * mis_bsdf * throughput
            + bsdf_val * em_weight * mis_em
        )

        aov = dr.select(valid_ray, E1 + LHS, 0)
        rgb = dr.select(valid_ray, E1 + RHS, 0)

        return (
            rgb,
            valid_ray,
            [
                aov.x,
                aov.y,
                aov.z,
            ],
        )

    def aov_names(self):
        # warning: The below list must be in accordance with method process_nerad_output() and the outputs of the method sample() in this class
        return [
            "LHS.R",
            "LHS.G",
            "LHS.B",
        ]

    def traverse(self, callback):
        self.network.traverse(callback)


mi.register_integrator("path_test", lambda props: PathIntegrator(props))

if __name__ == "__main__":
    with dr.suspend_grad():
        scene: mi.Scene = mi.load_dict(mi.cornell_box())

        integrator: PathIntegrator = mi.load_dict({"type": "path_test"})
        integrator.network = RadianceMLP(256, 8, scene.bbox())

        print(f"{mi.traverse(integrator)=}")

        sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        integrator.train(scene, sampler, 1024)

        img = mi.render(scene, integrator=integrator, spp=1)

        plt.imshow(img[:, :, 3])
        plt.show()
