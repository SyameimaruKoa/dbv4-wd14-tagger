# ncnn native diagnostics

`ncnn-respect-subgroup-option.patch` targets upstream ncnn revision
`9f9d4ec8150d840b570e735323499b12a354483a`.

The SPIR-V target in `src/gpu.cpp` depends on `opt.use_subgroup_ops`, but the
subgroup shader macros depend only on device capabilities. Disabling the option
therefore still emits subgroup operations while targeting SPIR-V 1.0. The AMD
diagnostic logged `subgroup op requires SPIR-V 1.3`, then a segmentation fault.
This patch gates those macros on the same option. It is a diagnostic dependency
patch; Vulkan probability parity remains to be verified.

Apply to the separate source checkout, then rebuild the `pyncnn` target:

```bash
git -C .dbv4/runtime/ncnn-candidate-source apply --unidiff-zero "$PWD/patches/ncnn-respect-subgroup-option.patch"
.dbv4/runtime/venv_ncnn_buildtools/bin/cmake --build .dbv4/runtime/ncnn-candidate-build --target pyncnn --parallel 2
```

The candidate uses Zig 0.16.0, CMake 4.4.3, Ninja 1.13.2, pybind11 3.1.0,
Vulkan enabled and OpenMP disabled. Python 3.13.16 headers are extracted into
`.dbv4/runtime/ncnn-python-headers`, without a system package installation.
Run the candidate by setting `PYTHONPATH` to
`.dbv4/runtime/ncnn-candidate-source/python`; do not overwrite the installed
wheel. Benchmark JSON records the loaded binding path and version.

Upstream source is licensed under [BSD 3-Clause](https://github.com/Tencent/ncnn/blob/9f9d4ec8150d840b570e735323499b12a354483a/LICENSE.txt).
