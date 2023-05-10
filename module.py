import mitsuba as mi
import drjit as dr


def matmul(a: mi.TensorXf, b: mi.TensorXf) -> mi.TensorXf:
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[0]

    N = a.shape[0]
    M = b.shape[1]
    K = b.shape[0]

    i, j = dr.arange(mi.UInt, N), dr.arange(mi.UInt, M)
    j, i = dr.meshgrid(j, i)

    print(f"{i=}")
    print(f"{j=}")

    k = mi.UInt(0)
    dst = mi.Float(0)

    loop = mi.Loop(name="matmul", state=lambda: (k, dst))
    loop.set_max_iterations(K)

    while loop(k < K):
        dst += dr.gather(mi.Float, a.array, i * a.shape[1] + k) * dr.gather(
            mi.Float, b.array, k * b.shape[1] + j
        )

        k += 1

    return mi.TensorXf(dst, shape=(N, M))


class Module:
    def __init__(self) -> None:
        ...

    def forward(self):
        ...


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.data = mi.TensorXf(mi.Float(0.0), [in_features, out_features])
