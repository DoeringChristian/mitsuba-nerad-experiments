import mitsuba as mi
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

from nerad import NRField, NeradIntegrator

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
        "to_world": mi.ScalarTransform4f.translate([0.335, 0.0, 0.0])
        .rotate([1.0, 0.0, 0.0], 90.0)
        .scale(0.5),
        "bsdf": {"type": "dielectric"},
    }

    scene: mi.Scene = mi.load_dict(scene_dict)

    field = NRField(scene, n_hidden=3, width=256)
    integrator = NeradIntegrator(field)
    integrator.train(scene)
    image_independent = mi.render(scene, spp=1, integrator=integrator)
    losses_independent = integrator.train_losses

    l_sampler: mi.Sampler = mi.load_dict({"type": "multijitter", "sample_count": 1})
    r_sampler: mi.Sampler = mi.load_dict({"type": "multijitter", "sample_count": 1})

    field = NRField(scene, n_hidden=3, width=256)
    integrator = NeradIntegrator(field, l_sampler=l_sampler, r_sampler=r_sampler)
    integrator.train(scene)
    image_multi = mi.render(scene, spp=1, integrator=integrator)
    losses_multi = integrator.train_losses

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_visible(False)

    ax[0][0].axis("off")
    ax[0][0].imshow(mi.util.convert_bitmap(image_independent))
    ax[1][0].plot(losses_independent)

    ax[0][1].axis("off")
    ax[0][1].imshow(mi.util.convert_bitmap(image_independent))
    ax[1][1].plot(losses_independent)

    fig.tight_layout()
    plt.show()
