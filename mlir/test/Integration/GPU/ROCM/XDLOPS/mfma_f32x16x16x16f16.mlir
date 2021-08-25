// RUN: mlir-opt %s \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-rocdl,gpu-to-hsaco{chip=gfx908})' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  %cst16 = constant 3 : index
  %cst64 = constant 64 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %cst2 = memref.dim %arg1, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst16, %block_y = %cst, %block_z = %cst) {
    %cst0 = constant 0 : i32
    %v1 = constant dense<[10.0, 1.0, 2.0, 3.0]>: vector<4xf16>
    %v2 = constant dense<[3.0, 7.0, 5.0, 6.0]>: vector<4xf16>
    %v3 = constant dense<[0.0, 0.0, 0.0, 0.0]>: vector<4xf32>
    %ans = rocdl.mfma.f32.16x16x16f16 %v1, %v2, %v3, %cst0, %cst0, %cst0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %val0 = vector.extract %ans[0]: vector<4xf32>
    %val1 = vector.extract %ans[1]: vector<4xf32>
    %val2 = vector.extract %ans[2]: vector<4xf32>
    %val3 = vector.extract %ans[3]: vector<4xf32>
    memref.store %val0, %arg1[%c0] : memref<?xf32>
    memref.store %val1, %arg1[%c1] : memref<?xf32>
    memref.store %val2, %arg1[%c2] : memref<?xf32>
    memref.store %val3, %arg1[%c3] : memref<?xf32>
    gpu.terminator
  }
  return
}

func @main() {
  %arg0 = memref.alloc() : memref<5xf32>
  %21 = constant 5 : i32
  %22 = memref.cast %arg0 : memref<5xf32> to memref<?xf32>
  %cast = memref.cast %22 : memref<?xf32> to memref<*xf32>
  gpu.host_register %cast : memref<*xf32>
  %23 = memref.cast %22 : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  %24 = constant 1.0 : f32
  %25 = call @mgpuMemGetDeviceMemRef1dFloat(%22) : (memref<?xf32>) -> (memref<?xf32>)
  call @other_func(%24, %25) : (f32, memref<?xf32>) -> ()
  call @print_memref_f32(%23) : (memref<*xf32>) -> ()
  return
}

func private @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func private @print_memref_f32(%ptr : memref<*xf32>)
