//===- LowerGpuOpsToROCDLOps.cpp - MLIR GPU to ROCDL lowering passes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate ROCDLIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOROCDLOPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

enum class DppCtrl : unsigned {
  DppQuadPerm0000 = 0x000,
  DppQuadPerm1111 = 0x055,
  DppQuadPerm2222 = 0x0AA,
  DppQuadPerm3333 = 0x0FF,
  DppQuadPerm1032 = 0x0B1,
  DppQuadPerm2301 = 0x04E,
  DppQuadPerm0123 = 0x0E4,
  DppQuadPerm3210 = 0x01B,
  DppRowSl1 = 0x101,
  DppRowSl2 = 0x102,
  DppRowSl3 = 0x103,
  DppRowSl4 = 0x104,
  DppRowSl5 = 0x105,
  DppRowSl6 = 0x106,
  DppRowSl7 = 0x107,
  DppRowSl8 = 0x108,
  DppRowSl9 = 0x109,
  DppRowSl10 = 0x10A,
  DppRowSl11 = 0x10B,
  DppRowSl12 = 0x10C,
  DppRowSl13 = 0x10D,
  DppRowSl14 = 0x10E,
  DppRowSl15 = 0x10F,
  DppRowSr1 = 0x111,
  DppRowSr2 = 0x112,
  DppRowSr3 = 0x113,
  DppRowSr4 = 0x114,
  DppRowSr5 = 0x115,
  DppRowSr6 = 0x116,
  DppRowSr7 = 0x117,
  DppRowSr8 = 0x118,
  DppRowSr9 = 0x119,
  DppRowSr10 = 0x11A,
  DppRowSr11 = 0x11B,
  DppRowSr12 = 0x11C,
  DppRowSr13 = 0x11D,
  DppRowSr14 = 0x11E,
  DppRowSr15 = 0x11F,
  DppRowRr1 = 0x121,
  DppRowRr2 = 0x122,
  DppRowRr3 = 0x123,
  DppRowRr4 = 0x124,
  DppRowRr5 = 0x125,
  DppRowRr6 = 0x126,
  DppRowRr7 = 0x127,
  DppRowRr8 = 0x128,
  DppRowRr9 = 0x129,
  DppRowRr10 = 0x12A,
  DppRowRr11 = 0x12B,
  DppRowRr12 = 0x12C,
  DppRowRr13 = 0x12D,
  DppRowRr14 = 0x12E,
  DppRowRr15 = 0x12F,

  // WfSl and WfSr are not available on GFX10+.
  DppWfSl1 = 0x130,
  DppWfSr1 = 0x138,

  DppRowMirror = 0x140,
  DppRowHalfMirror = 0x141,

  // RowBcast modes are not available on GFX10+.
  DppRowBcast15 = 0x142,
  DppRowBcast31 = 0x143,

  // RowXmask and RowShare modes are only available on GFX10+.
  DppRowShare0 = 0x150,
  DppRowShare1 = 0x151,
  DppRowShare2 = 0x152,
  DppRowShare3 = 0x153,
  DppRowShare4 = 0x154,
  DppRowShare5 = 0x155,
  DppRowShare6 = 0x156,
  DppRowShare7 = 0x157,
  DppRowShare8 = 0x158,
  DppRowShare9 = 0x159,
  DppRowShare10 = 0x15A,
  DppRowShare11 = 0x15B,
  DppRowShare12 = 0x15C,
  DppRowShare13 = 0x15D,
  DppRowShare14 = 0x15E,
  DppRowShare15 = 0x15F,
  DppRowXmask0 = 0x160,
  DppRowXmask1 = 0x161,
  DppRowXmask2 = 0x162,
  DppRowXmask3 = 0x163,
  DppRowXmask4 = 0x164,
  DppRowXmask5 = 0x165,
  DppRowXmask6 = 0x166,
  DppRowXmask7 = 0x167,
  DppRowXmask8 = 0x168,
  DppRowXmask9 = 0x169,
  DppRowXmask10 = 0x16A,
  DppRowXmask11 = 0x16B,
  DppRowXmask12 = 0x16C,
  DppRowXmask13 = 0x16D,
  DppRowXmask14 = 0x16E,
  DppRowXmask15 = 0x16F,
};

/// Returns true if the given `gpu.func` can be safely called using the bare
/// pointer calling convention.
static bool canBeCalledWithBarePointers(gpu::GPUFuncOp func) {
  bool canBeBare = true;
  for (Type type : func.getArgumentTypes())
    if (auto memrefTy = dyn_cast<BaseMemRefType>(type))
      canBeBare &= LLVMTypeConverter::canConvertToBarePtr(memrefTy);
  return canBeBare;
}

Value getLaneId(RewriterBase &rewriter, Location loc,
                const unsigned indexBitwidth) {
  auto int32Type = IntegerType::get(rewriter.getContext(), 32);
  Value zero = rewriter.createOrFold<arith::ConstantIntOp>(loc, 0, 32);
  Value minus1 = rewriter.createOrFold<arith::ConstantIntOp>(loc, -1, 32);
  Value mbcntLo = rewriter.create<ROCDL::MbcntLoOp>(loc, int32Type,
                                                    ValueRange{minus1, zero});
  Value laneId = rewriter.create<ROCDL::MbcntHiOp>(loc, int32Type,
                                                   ValueRange{minus1, mbcntLo});
  return laneId;
}

namespace {
struct GPULaneIdOpToROCDL : ConvertOpToLLVMPattern<gpu::LaneIdOp> {
  using ConvertOpToLLVMPattern<gpu::LaneIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::LaneIdOp op, gpu::LaneIdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    // convert to:  %mlo = call @llvm.amdgcn.mbcnt.lo(-1, 0)
    // followed by: %lid = call @llvm.amdgcn.mbcnt.hi(-1, %mlo)

    Type intTy = IntegerType::get(context, 32);
    Value zero = rewriter.createOrFold<arith::ConstantIntOp>(loc, 0, 32);
    Value minus1 = rewriter.createOrFold<arith::ConstantIntOp>(loc, -1, 32);
    Value mbcntLo =
        rewriter.create<ROCDL::MbcntLoOp>(loc, intTy, ValueRange{minus1, zero});
    Value laneId = rewriter.create<ROCDL::MbcntHiOp>(
        loc, intTy, ValueRange{minus1, mbcntLo});
    // Truncate or extend the result depending on the index bitwidth specified
    // by the LLVMTypeConverter options.
    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();
    if (indexBitwidth > 32) {
      laneId = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), laneId);
    } else if (indexBitwidth < 32) {
      laneId = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), laneId);
    }
    rewriter.replaceOp(op, {laneId});
    return success();
  }
};

struct GPUShuffleOpLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  std::pair<Value, Value> createShuffle(RewriterBase &rewriter, Location loc,
                                        Value srcLaneId, Value dstLaneId,
                                        OpAdaptor adaptor) const {
    auto int32Type = rewriter.getIntegerType(32);
    auto boolType = rewriter.getIntegerType(1);
    auto width = adaptor.getWidth();

    Value zero = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 0);
    Value negwidth = rewriter.create<LLVM::SubOp>(loc, int32Type, zero, width);
    Value add = rewriter.create<LLVM::AddOp>(loc, int32Type, srcLaneId, width);
    Value widthOrZeroIfOutside =
        rewriter.create<LLVM::AndOp>(loc, int32Type, add, negwidth);
    Value isActiveSrcLane = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::slt, dstLaneId, widthOrZeroIfOutside);
    Value selectDstLane = rewriter.create<LLVM::SelectOp>(loc, isActiveSrcLane,
                                                          dstLaneId, srcLaneId);
    Value two = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 2);
    Value dwordAlignedDstLane =
        rewriter.create<LLVM::ShlOp>(loc, int32Type, selectDstLane, two);
    Value initShflValue = adaptor.getValue();
    if (adaptor.getValue().getType().isF32()) {
      initShflValue =
          rewriter.create<LLVM::BitcastOp>(loc, int32Type, initShflValue);
    }

    Value shflValue = rewriter.create<ROCDL::DsBpermuteOp>(
        loc, int32Type, dwordAlignedDstLane, initShflValue);
    if (adaptor.getValue().getType().isF32()) {
      shflValue = rewriter.create<LLVM::BitcastOp>(
          loc, adaptor.getValue().getType(), shflValue);
    }

    return std::make_pair(shflValue, isActiveSrcLane);
  }

  FailureOr<unsigned> getStaticMask(Value offset, Value width) const {
    auto widthOp = dyn_cast_or_null<LLVM::ConstantOp>(width.getDefiningOp());
    auto offsetOp = dyn_cast_or_null<LLVM::ConstantOp>(offset.getDefiningOp());
    if (!widthOp || !offsetOp)
      return failure();
    auto widthIntAttr = dyn_cast<IntegerAttr>(widthOp.getValue());
    auto offsetIntAttr = dyn_cast<IntegerAttr>(offsetOp.getValue());
    if (!offsetIntAttr || !widthIntAttr)
      return failure();
    int32_t offsetInt32 = offsetIntAttr.getInt();
    int32_t widthInt32 = widthIntAttr.getInt();
    // Width needs to be same size as warp.
    const int32_t kWarpSize = 32;
    if (widthInt32 != kWarpSize)
      return failure();
    // Do zext to unsigned to represent mask.
    return static_cast<unsigned>(offsetInt32);
  }

  std::pair<Value, Value> createShuffleXor(RewriterBase &rewriter, Location loc,
                                           OpAdaptor adaptor) const {
    auto int32Type = rewriter.getIntegerType(32);
    auto boolType = rewriter.getIntegerType(1);
    Value offset = adaptor.getOffset();
    Value width = adaptor.getWidth();

    bool canOptimize = false;
    DppCtrl dppCtrl = DppCtrl::DppQuadPerm0000;

    Value initShflValue = adaptor.getValue();
    if (adaptor.getValue().getType().isF32()) {
      initShflValue =
          rewriter.create<LLVM::BitcastOp>(loc, int32Type, initShflValue);
    }

    // Use dpp_mov if mask < 16
    // Use permlanex16 if support or ds_swizzle, if 16 <= mask < 32
    // Use ds_bpermute to handle more complex cases
    auto maybeMask = getStaticMask(adaptor.getOffset(), adaptor.getWidth());
    if (succeeded(maybeMask)) {
      unsigned mask = maybeMask.value();
      if (mask < 32) {
        bool canOptimize = true;
        switch (mask) {
        case 0:
          dppCtrl = DppCtrl::DppQuadPerm0123;
          break;
        case 1:
          dppCtrl = DppCtrl::DppQuadPerm1032;
          break;
        case 2:
          dppCtrl = DppCtrl::DppQuadPerm2301;
          break;
        case 3:
          dppCtrl = DppCtrl::DppQuadPerm3210;
          break;
        case 7:
          dppCtrl = DppCtrl::DppRowHalfMirror;
        case 8:
          dppCtrl = DppCtrl::DppRowRr8;
        case 15:
          dppCtrl = DppCtrl::DppRowMirror;
          break;
        default:
          canOptimize = false;
          break;
        }

        // TODO: Check if the backend supports DppRowXmask.
        // Check if we can use DppRowXmask.
        if (!canOptimize) {
          canOptimize = true;
          switch (mask) {
          case 4:
            dppCtrl = DppCtrl::DppRowXmask4;
            break;
          case 5:
            dppCtrl = DppCtrl::DppRowXmask5;
            break;
          case 6:
            dppCtrl = DppCtrl::DppRowXmask6;
            break;
          case 9:
            dppCtrl = DppCtrl::DppRowXmask9;
            break;
          case 10:
            dppCtrl = DppCtrl::DppRowXmask10;
            break;
          case 11:
            dppCtrl = DppCtrl::DppRowXmask11;
            break;
          case 12:
            dppCtrl = DppCtrl::DppRowXmask12;
            break;
          case 13:
            dppCtrl = DppCtrl::DppRowXmask13;
            break;
          case 14:
            dppCtrl = DppCtrl::DppRowXmask14;
            break;
          default:
            canOptimize = false;
            break;
          }
        }

        // TODO: Check if we support dppmov.
        if (canOptimize) {
          Value dppCtrlVal = rewriter.create<LLVM::ConstantOp>(
              loc, int32Type, (unsigned)dppCtrl);
          Value rowMask = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 15);
          Value bankMask =
              rewriter.create<LLVM::ConstantOp>(loc, int32Type, 15);
          Value boundCtrl =
              rewriter.create<LLVM::ConstantOp>(loc, boolType, true);
          // TODO: Add a builder that directly takes int32_t.
          Value shflValue = rewriter.create<ROCDL::MovDppOp>(
              loc, int32Type, initShflValue, dppCtrlVal, rowMask, bankMask,
              boundCtrl);
          if (adaptor.getValue().getType().isF32()) {
            shflValue = rewriter.create<LLVM::BitcastOp>(
                loc, adaptor.getValue().getType(), shflValue);
          }
          // dstLane is always inside since we assume width to be 32.
          Value isActiveSrcLane =
              rewriter.create<LLVM::ConstantOp>(loc, boolType, true);
          return std::make_pair(shflValue, isActiveSrcLane);
        }

        // TODO: Check if we support permlanedpp instructions.
        if (mask >= 16) {
          // For each mask from 16 to 31, this table contains the lane
          // selection bits for the permlanex16 instruction.
          static const unsigned laneSelBits[16][2] = {
              {0x76543210, 0xfedcba98}, {0x67452301, 0xefcdab89},
              {0x54761032, 0xdcfe98ba}, {0x45670123, 0xcdef89ab},
              {0x32107654, 0xba98fedc}, {0x23016745, 0xab89efcd},
              {0x10325476, 0x98badcfe}, {0x1234567, 0x89abcdef},
              {0xfedcba98, 0x76543210}, {0xefcdab89, 0x67452301},
              {0xdcfe98ba, 0x54761032}, {0xcdef89ab, 0x45670123},
              {0xba98fedc, 0x32107654}, {0xab89efcd, 0x23016745},
              {0x98badcfe, 0x10325476}, {0x89abcdef, 0x1234567}};
          unsigned laneSelS1 = laneSelBits[mask - 16][0];
          unsigned laneSelS2 = laneSelBits[mask - 16][1];

          Value old = rewriter.create<LLVM::UndefOp>(loc, int32Type);
          Value laneSelS1Value =
              rewriter.create<LLVM::ConstantOp>(loc, int32Type, laneSelS1);
          Value laneSelS2Value =
              rewriter.create<LLVM::ConstantOp>(loc, int32Type, laneSelS2);
          Value fi = rewriter.create<LLVM::ConstantOp>(loc, boolType, false);
          Value boundCtrl =
              rewriter.create<LLVM::ConstantOp>(loc, boolType, false);
          Value shflValue = rewriter.create<ROCDL::PermlaneX16Op>(
              loc, int32Type, old, initShflValue, laneSelS1Value,
              laneSelS2Value, fi, boundCtrl);
          if (adaptor.getValue().getType().isF32()) {
            shflValue = rewriter.create<LLVM::BitcastOp>(
                loc, adaptor.getValue().getType(), shflValue);
          }
          // dstLane is always inside since we assume width to be 32.
          Value isActiveSrcLane =
              rewriter.create<LLVM::ConstantOp>(loc, boolType, true);
          return std::make_pair(shflValue, isActiveSrcLane);
        }

        // TODO: Use ds_swizzle here if permlanex16 instructions are not
        // supported.
      }
    }

    // Use default path.
    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();
    Value srcLaneId = getLaneId(rewriter, loc, indexBitwidth);
    Value dstLaneId =
        rewriter.create<LLVM::XOrOp>(loc, int32Type, srcLaneId, offset);
    return createShuffle(rewriter, loc, srcLaneId, dstLaneId, adaptor);
  }

  /// Lowers a shuffle to the corresponding ROCDL ops.
  ///
  /// Use the `width` argument to see if src lane is participating.
  /// If not the dstLane would be itself.
  ///
  ///  Shuffle with DS Bpermute:
  ///   let shflMode = [xor, up, down, idx]
  ///   let width = 32(usually warpsize), step = [1, 2, 4, 8, 16, ... , width].
  ///   1. curLaneId = using mbcnt.lo + mbcnt.hi
  ///   2. widthOrZeroIfOutside = (curLaneId + width) & -width
  ///   3. dstLane = shflMode(curLaneId, step)
  ///   4. isActiveSrcLane = dstLane < isActiveSrcLane
  ///   5. dstLane = isActiveSrcLane ? dstLane : curLaneId
  ///   6. dwordAlignedDstLane = dstLane * 4 or dstLane << 2.
  ///   7. bpermute(dwordAlignedDstLane, shfl_value).
  ///
  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // TODO: Add support for non 32-bit shuffle values.
    if (adaptor.getValue().getType().getIntOrFloatBitWidth() != 32)
      return failure();

    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    Value offset = adaptor.getOffset();
    Value width = adaptor.getWidth();
    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();

    // TODO: Add support for gpu::ShuffleMode::UP and gpu::ShuffleMode::DOWN.
    Value shflValue, isActiveSrcLane;
    switch (op.getMode()) {
    case gpu::ShuffleMode::XOR: {
      std::tie(shflValue, isActiveSrcLane) =
          createShuffleXor(rewriter, loc, adaptor);
      break;
    }
    case gpu::ShuffleMode::IDX: {
      Value srcLaneId = getLaneId(rewriter, loc, indexBitwidth);
      std::tie(shflValue, isActiveSrcLane) =
          createShuffle(rewriter, loc, srcLaneId, offset, adaptor);
      break;
    }
    case gpu::ShuffleMode::UP:
    case gpu::ShuffleMode::DOWN:
      return failure();
    }

    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});

    return success();
  }
};

/// Import the GPU Ops to ROCDL Patterns.
#include "GPUToROCDL.cpp.inc"

// A pass that replaces all occurrences of GPU device operations with their
// corresponding ROCDL equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct LowerGpuOpsToROCDLOpsPass
    : public impl::ConvertGpuOpsToROCDLOpsBase<LowerGpuOpsToROCDLOpsPass> {
  LowerGpuOpsToROCDLOpsPass() = default;
  LowerGpuOpsToROCDLOpsPass(const std::string &chipset, unsigned indexBitwidth,
                            bool useBarePtrCallConv,
                            gpu::amd::Runtime runtime) {
    if (this->chipset.getNumOccurrences() == 0)
      this->chipset = chipset;
    if (this->indexBitwidth.getNumOccurrences() == 0)
      this->indexBitwidth = indexBitwidth;
    if (this->useBarePtrCallConv.getNumOccurrences() == 0)
      this->useBarePtrCallConv = useBarePtrCallConv;
    if (this->runtime.getNumOccurrences() == 0)
      this->runtime = runtime;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(ctx));
    }

    FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        ctx, DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.useOpaquePointers = useOpaquePointers;

    if (useBarePtrCallConv) {
      options.useBarePtrCallConv = true;
      WalkResult canUseBarePointers =
          m.walk([](gpu::GPUFuncOp func) -> WalkResult {
            if (canBeCalledWithBarePointers(func))
              return WalkResult::advance();
            return WalkResult::interrupt();
          });
      if (canUseBarePointers.wasInterrupted()) {
        emitError(UnknownLoc::get(ctx),
                  "bare pointer calling convention requires all memrefs to "
                  "have static shape and use the identity map");
        return signalPassFailure();
      }
    }

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(ctx);
      populateGpuRewritePatterns(patterns);
      arith::populateExpandBFloat16Patterns(patterns);
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }

    LLVMTypeConverter converter(ctx, options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) {
          switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 5;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });

    RewritePatternSet llvmPatterns(ctx);

    mlir::arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    populateAMDGPUToROCDLConversionPatterns(converter, llvmPatterns,
                                            *maybeChipset);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToROCDLConversionPatterns(converter, llvmPatterns, runtime);
    LLVMConversionTarget target(getContext());
    configureGpuToROCDLConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();

    // Manually rewrite known block size attributes so the LLVMIR translation
    // infrastructure can pick them up.
    m.walk([ctx](LLVM::LLVMFuncOp op) {
      if (auto blockSizes = dyn_cast_or_null<DenseI32ArrayAttr>(
              op->removeAttr(gpu::GPUFuncOp::getKnownBlockSizeAttrName()))) {
        op->setAttr(ROCDL::ROCDLDialect::getReqdWorkGroupSizeAttrName(),
                    blockSizes);
        // Also set up the rocdl.flat_work_group_size attribute to prevent
        // conflicting metadata.
        uint32_t flatSize = 1;
        for (uint32_t size : blockSizes.asArrayRef()) {
          flatSize *= size;
        }
        StringAttr flatSizeAttr =
            StringAttr::get(ctx, Twine(flatSize) + "," + Twine(flatSize));
        op->setAttr(ROCDL::ROCDLDialect::getFlatWorkGroupSizeAttrName(),
                    flatSizeAttr);
      }
    });
  }
};

} // namespace

void mlir::configureGpuToROCDLConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<ROCDL::ROCDLDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FRemOp, LLVM::LogOp,
                      LLVM::Log10Op, LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp,
                      LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

template <typename OpTy>
static void populateOpPatterns(LLVMTypeConverter &converter,
                               RewritePatternSet &patterns, StringRef f32Func,
                               StringRef f64Func) {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func);
}

void mlir::populateGpuToROCDLConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    mlir::gpu::amd::Runtime runtime) {
  using mlir::gpu::amd::Runtime;

  populateWithGenerated(patterns);
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                       ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>>(
          converter, gpu::GPUFuncOp::getKnownBlockSizeAttrName());
  patterns.add<GPUIndexIntrinsicOpLowering<
      gpu::BlockIdOp, ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>>(
      converter, gpu::GPUFuncOp::getKnownGridSizeAttrName());
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp,
                                       ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp,
                                       ROCDL::GridDimYOp, ROCDL::GridDimZOp>,
           GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      /*allocaAddrSpace=*/ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace,
      /*workgroupAddrSpace=*/ROCDL::ROCDLDialect::kSharedMemoryAddressSpace,
      StringAttr::get(&converter.getContext(),
                      ROCDL::ROCDLDialect::getKernelFuncAttrName()));
  if (Runtime::HIP == runtime) {
    patterns.add<GPUPrintfOpToHIPLowering>(converter);
  } else if (Runtime::OpenCL == runtime) {
    // Use address space = 4 to match the OpenCL definition of printf()
    patterns.add<GPUPrintfOpToLLVMCallLowering>(converter, /*addressSpace=*/4);
  }

  patterns.add<GPUShuffleOpLowering, GPULaneIdOpToROCDL>(converter);

  populateOpPatterns<math::AbsFOp>(converter, patterns, "__ocml_fabs_f32",
                                   "__ocml_fabs_f64");
  populateOpPatterns<math::AtanOp>(converter, patterns, "__ocml_atan_f32",
                                   "__ocml_atan_f64");
  populateOpPatterns<math::Atan2Op>(converter, patterns, "__ocml_atan2_f32",
                                    "__ocml_atan2_f64");
  populateOpPatterns<math::CbrtOp>(converter, patterns, "__ocml_cbrt_f32",
                                   "__ocml_cbrt_f64");
  populateOpPatterns<math::CeilOp>(converter, patterns, "__ocml_ceil_f32",
                                   "__ocml_ceil_f64");
  populateOpPatterns<math::CosOp>(converter, patterns, "__ocml_cos_f32",
                                  "__ocml_cos_f64");
  populateOpPatterns<math::ExpOp>(converter, patterns, "__ocml_exp_f32",
                                  "__ocml_exp_f64");
  populateOpPatterns<math::Exp2Op>(converter, patterns, "__ocml_exp2_f32",
                                   "__ocml_exp2_f64");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, "__ocml_expm1_f32",
                                    "__ocml_expm1_f64");
  populateOpPatterns<math::FloorOp>(converter, patterns, "__ocml_floor_f32",
                                    "__ocml_floor_f64");
  populateOpPatterns<arith::RemFOp>(converter, patterns, "__ocml_fmod_f32",
                                    "__ocml_fmod_f64");
  populateOpPatterns<math::LogOp>(converter, patterns, "__ocml_log_f32",
                                  "__ocml_log_f64");
  populateOpPatterns<math::Log10Op>(converter, patterns, "__ocml_log10_f32",
                                    "__ocml_log10_f64");
  populateOpPatterns<math::Log1pOp>(converter, patterns, "__ocml_log1p_f32",
                                    "__ocml_log1p_f64");
  populateOpPatterns<math::Log2Op>(converter, patterns, "__ocml_log2_f32",
                                   "__ocml_log2_f64");
  populateOpPatterns<math::PowFOp>(converter, patterns, "__ocml_pow_f32",
                                   "__ocml_pow_f64");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, "__ocml_rsqrt_f32",
                                    "__ocml_rsqrt_f64");
  populateOpPatterns<math::SinOp>(converter, patterns, "__ocml_sin_f32",
                                  "__ocml_sin_f64");
  populateOpPatterns<math::SqrtOp>(converter, patterns, "__ocml_sqrt_f32",
                                   "__ocml_sqrt_f64");
  populateOpPatterns<math::TanhOp>(converter, patterns, "__ocml_tanh_f32",
                                   "__ocml_tanh_f64");
  populateOpPatterns<math::TanOp>(converter, patterns, "__ocml_tan_f32",
                                  "__ocml_tan_f64");
  populateOpPatterns<math::ErfOp>(converter, patterns, "__ocml_erf_f32",
                                  "__ocml_erf_f64");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass(const std::string &chipset,
                                      unsigned indexBitwidth,
                                      bool useBarePtrCallConv,
                                      gpu::amd::Runtime runtime) {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>(
      chipset, indexBitwidth, useBarePtrCallConv, runtime);
}
