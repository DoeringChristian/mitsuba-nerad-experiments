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


class PathIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth: int = props.get("max_depth", 8)
        self.rr_depth: int = props.get("rr_depth", 2)
        self.n = 0

    def sample_model(self, scene, model, si, bsdf, sampler):
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
