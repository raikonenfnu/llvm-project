# XDLOPS in MLIR

## _Intuition behind MFMA Ops_
### Broadcasting MFMA Ops
Coming soon!
### Reduction MFMA Ops
Coming Soon!
## _Building LLVM for XDLOPs_
```sh
git clone https://github.com/raikonenfnu/llvm-project
cd llvm-project
git checkout xdlops
export LLVM_BUILD_DIR=<your choice of build dir>
cmake -S llvm -B ${LLVM_BUILD_DIR} -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_TARGETS_TO_BUILD="X86;ARM;AArch64;NVPTX;AMDGPU" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DMLIR_ENABLE_CUDA_RUNNER=ON -DMLIR_ENABLE_ROCM_RUNNER=ON -DMLIR_GPU_TO_HSACO_PASS_ENABLE=ON -DLLVM_ENABLE_PROJECTS="mlir;lld"
cmake --build build ${LLVM_BUILD_DIR}
```

## _Running_ 
```sh
${LLVM_BUILD_DIR}/bin/mlir-opt mfma_f32x16x16x16f16.mlir \
  -gpu-kernel-outlining \
  -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-rocdl,gpu-to-hsaco{chip=gfx908})' \
  -gpu-to-llvm \
| ${LLVM_BUILD_DIR}/bin/mlir-cpu-runner \
  --shared-libs=${LLVM_BUILD_DIR}/lib/libmlir_rocm_runtime.so \
  --shared-libs=${LLVM_BUILD_DIR}/lib/libmlir_runner_utils.so \
  --entry-point-result=void \
```
### Expected Result
```sh
  Unranked Memref base@ = 0x796e400 rank = 1 offset = 0 sizes = [5] strides = [1] data =
  [2.19291e-34,  0,  0,  0,  4.48416e-44]
  Unranked Memref base@ = 0x796e400 rank = 1 offset = 0 sizes = [5] strides = [1] data =
  [65,  65,  65,  -nan,  4.48416e-44]
```