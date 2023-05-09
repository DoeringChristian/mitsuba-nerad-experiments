import mitsuba as mi
import drjit as dr


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

from nerad.integrator.nerad import Nerad

mi.register_integrator("nerad", lambda props: Nerad(props))

if __name__ == "__main__":
    scene = mi.load_dict(mi.cornell_box())
    integrator = mi.load_dict(
        {
            "type": "nerad",
        }
    )
