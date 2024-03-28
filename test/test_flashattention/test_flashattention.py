"""Test FlashAttention-v2-FP32"""

from math import sqrt
import unittest
import os
import subprocess
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor


def _find_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        try:
            nvcc = subprocess.check_output(["which", "nvcc"]).decode().rstrip("\r\n")
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("NVCC Not Available") from exc
    return cuda_home


def _get_nvcc_info(cuda_home):
    nvcc = None
    if cuda_home is not None and os.path.isdir(cuda_home):
        try:
            nvcc = os.path.join(cuda_home, "bin/nvcc")
            subprocess.check_output(f"{nvcc} -V", shell=True)
        except subprocess.SubprocessError as exc:
            raise RuntimeError("NVCC Not Available") from exc
    return nvcc


ENV_INFO = {}
CUDA_HOME = _find_cuda_home()
ENV_INFO["cuda_home"] = CUDA_HOME
ENV_INFO["NVCC"] = _get_nvcc_info(CUDA_HOME)


def compile_kernel(kernel_name, **kwargs):
    """compile kernel and return so file path"""
    kernel_folder = Path(__file__).resolve().parent.parent / "_csrc" / "cuda"
    cuda_kernel_file = kernel_folder / f"{kernel_name}.cu"
    cuda_so_file = kernel_folder / f"{kernel_name}.so"
    if cuda_so_file.exists():
        return cuda_so_file

    flags = [
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]

    for key, value in kwargs.items():
        flags.append(f"-D{key}={value}")

    # Construct nvcc command-line arguments
    nvcc_args = [ENV_INFO["NVCC"], str(cuda_kernel_file), "-o", str(cuda_so_file)]

    nvcc_command = " ".join(nvcc_args + flags)
    # Execute nvcc compilation command
    print(nvcc_command)
    result = subprocess.run(
        nvcc_command,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        print("Compilation succeeded:", result.stdout.decode())
    else:
        error_message = result.stderr.decode()
        raise RuntimeError("Compilation failed:", error_message)

    return cuda_so_file


def load_flash_cuda_kernel(self, func_name, context_length):
    """load flash cuda kernel"""
    device_target = ms.get_context("device_target")
    if device_target != "GPU":
        raise RuntimeError("FlashAttention operator only support GPU currently.")

    so_path = compile_kernel(kernel_name="flash", Tmax=context_length)
    flash_op = ops.Custom(
        f"{str(so_path)}:{func_name}",
        out_shape=lambda q, k, v, l, m: q,
        out_dtype=lambda q, k, v, l, m: q,
        func_type="aot",
    )
    flash_op.add_prim_attr("primitive_target", device_target)
    return flash_op


def manual_attn(self, query, key, value):
    r"""
    manual attention
    """
    embed_size = query.shape[-1]
    scaling_factor = sqrt(sqrt(Tensor(embed_size, ms.float32)))
    query = query / scaling_factor
    attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)
    attn = ops.softmax(attn, -1)
    output = ops.matmul(attn, value)
    return output


def test_flashattention2_forward_FP32(self):
    r"""
    Unit test for flashattention forward.
    """
    # 加载flash cuda kernel
    op = self.load_flash_cuda_kernel("flash_forward", 512)

    ms.set_context(pynative_synchronize=True)
    profiler = ms.Profiler()

    # 定义输入数据
    Q = np.random.randn(16, 12, 64, 64).astype(np.float32)
    K = np.random.randn(16, 12, 64, 64).astype(np.float32)
    V = np.random.randn(16, 12, 64, 64).astype(np.float32)
    l = np.zeros((16, 12, 64), dtype=np.float32)
    m = np.ones((16, 12, 64), dtype=np.float32) * (-np.inf)
    print("=== profiling MindSpore manual-attention === ")
    output_manual = self.manual_attn(ms.Tensor(Q), ms.Tensor(K), ms.Tensor(V))
    # profiler.analyse()
    print("=== profiling MindSpore flash-attention === ")
    output = op(ms.Tensor(Q), ms.Tensor(K), ms.Tensor(V), ms.Tensor(l), ms.Tensor(m))
    profiler.analyse()
    # print(output)
    assert np.allclose(output[0].asnumpy(), output_manual[0].asnumpy(), atol=1e-03)
