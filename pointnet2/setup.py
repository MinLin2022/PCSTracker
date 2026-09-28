from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


SOURCE_DIR = Path(__file__).resolve().parent / "src"


setup(
    name="pcstracker-pointnet2",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            "pointnet2_cuda",
            sources=[
                str(SOURCE_DIR / "pointnet2_api.cpp"),
                str(SOURCE_DIR / "group_points.cpp"),
                str(SOURCE_DIR / "group_points_gpu.cu"),
                str(SOURCE_DIR / "sampling.cpp"),
                str(SOURCE_DIR / "sampling_gpu.cu"),
            ],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
