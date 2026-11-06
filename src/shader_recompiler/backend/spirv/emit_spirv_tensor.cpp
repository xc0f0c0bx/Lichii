// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <span>

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "common/common_types.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Backend::SPIRV {
namespace {
// Cooperative matrix HMMA: KHR opcodes and operands Sirit has no instructions for.
constexpr u32 OP_TYPE_COOPERATIVE_MATRIX_KHR = 4456;
constexpr u32 OP_COOPERATIVE_MATRIX_LOAD_KHR = 4457;
constexpr u32 OP_COOPERATIVE_MATRIX_STORE_KHR = 4458;
constexpr u32 OP_COOPERATIVE_MATRIX_MUL_ADD_KHR = 4459;
constexpr u32 CAPABILITY_COOPERATIVE_MATRIX_KHR = 6022;
constexpr u32 OP_TYPE_ARRAY = 28;
constexpr u32 OP_EXT_INST = 12;
constexpr u32 MARKER_LOAD = 1;
constexpr u32 MARKER_STORE = 2;
constexpr u32 MARKER_MUL_ADD = 3;
constexpr u32 USE_MATRIX_A = 0;
constexpr u32 USE_MATRIX_B = 1;
constexpr u32 USE_ACCUMULATOR = 2;
constexpr u32 LAYOUT_ROW_MAJOR = 0;
constexpr u32 LAYOUT_COLUMN_MAJOR = 1;
constexpr u32 MARKER_SET = 0x434F4F50; // "COOP"
constexpr u32 TYPE_MAGIC = 0x434F0000;

// Per-subgroup staging in shared memory, in 32-bit words holding half pairs: A (16 rows x 16),
// B transposed (8 rows x 16) and C/D (16 rows x 8). The fp32-accumulator shape adds C/D as
// floats (16 rows x 8).
constexpr u32 SCRATCH_A = 0;
constexpr u32 SCRATCH_BT = 128;
constexpr u32 SCRATCH_C = 192;
constexpr u32 SCRATCH_C32 = 256;
constexpr u32 SHARED_MEMORY_LIMIT = 32 * 1024;

u32 ScratchWords(CooperativeMatrixShape shape) {
    return shape == CooperativeMatrixShape::M8N8K16 ? 384 : 256;
}

bool UsesTensor(const IR::Program& program) {
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::TensorMulAdd && inst.Flags<u32>() != 0) {
                return true;
            }
        }
    }
    return false;
}

u32 SubgroupsPerWorkgroup(const IR::Program& program) {
    const u32 threads =
        program.workgroup_size[0] * program.workgroup_size[1] * program.workgroup_size[2];
    return (threads + 31) / 32;
}

bool UsesCooperativeMatrix(const EmitContext& ctx, const IR::Program& program) {
    if (ctx.profile.cooperative_matrix_shape == CooperativeMatrixShape::None ||
        program.stage != Stage::Compute || !ctx.profile.SupportsSubgroupStage(program.stage)) {
        return false;
    }
    return program.shared_memory_size + SubgroupsPerWorkgroup(program) *
                                            ScratchWords(ctx.profile.cooperative_matrix_shape) *
                                            4 <=
           SHARED_MEMORY_LIMIT;
}

Id CooperativeMatrixType(EmitContext& ctx, Id component, u32 rows, u32 columns, u32 use) {
    // The constants come first so the rewritten type can refer to them.
    const Id scope = ctx.Const(static_cast<u32>(spv::Scope::Subgroup));
    const Id rows_id = ctx.Const(rows);
    const Id columns_id = ctx.Const(columns);
    const Id use_id = ctx.Const(use);
    // A placeholder array whose length names the shape; PatchCooperativeMatrix rewrites it.
    const Id type = ctx.TypeArray(component, ctx.Const(TYPE_MAGIC | rows << 8 | columns << 4 | use));
    if (std::ranges::none_of(ctx.cooperative_matrix_types,
                             [&](const auto& entry) { return entry.type.value == type.value; })) {
        ctx.cooperative_matrix_types.push_back({type, scope, rows_id, columns_id, use_id});
    }
    return type;
}

void DefineCooperativeMatrixFunction(EmitContext& ctx, const IR::Program& program) {
    const CooperativeMatrixShape shape = ctx.profile.cooperative_matrix_shape;
    const bool transposed = shape == CooperativeMatrixShape::M8N16K16;
    const bool halves = shape == CooperativeMatrixShape::M8N8K16;
    const u32 scratch_words = ScratchWords(shape);
    ctx.AddCapability(spv::Capability::GroupNonUniform);
    ctx.AddCapability(static_cast<spv::Capability>(CAPABILITY_COOPERATIVE_MATRIX_KHR));
    ctx.AddExtension("SPV_KHR_cooperative_matrix");
    const Id u32ty = ctx.U32[1];
    Id f16 = ctx.F16[1];
    if (!Sirit::ValidId(f16)) {
        ctx.AddCapability(spv::Capability::Float16);
        f16 = ctx.TypeFloat(16);
    }
    ctx.cooperative_matrix_marker = ctx.Const(MARKER_SET);
    const auto imm = [&](u32 v) { return ctx.Const(v); };
    const auto add = [&](Id a, Id b) { return ctx.OpIAdd(u32ty, a, b); };
    const auto addi = [&](Id a, u32 b) { return ctx.OpIAdd(u32ty, a, imm(b)); };

    // Scratch for every subgroup of the workgroup, and the subgroup's index.
    const Id scratch_type = ctx.TypeArray(u32ty, imm(SubgroupsPerWorkgroup(program) * scratch_words));
    const Id scratch = ctx.AddGlobalVariable(
        ctx.TypePointer(spv::StorageClass::Workgroup, scratch_type), spv::StorageClass::Workgroup);
    ctx.interfaces.push_back(scratch);
    const Id word_pointer = ctx.TypePointer(spv::StorageClass::Workgroup, u32ty);
    const Id subgroup_id = ctx.AddGlobalVariable(
        ctx.TypePointer(spv::StorageClass::Input, u32ty), spv::StorageClass::Input);
    ctx.Decorate(subgroup_id, spv::Decoration::BuiltIn, spv::BuiltIn::SubgroupId);
    ctx.interfaces.push_back(subgroup_id);

    // Direct: D(16x8) = A(16x16) B(16x8) + C. Transposed: D^T(8x16) = B^T(8x16) A^T(16x16) + C^T,
    // which the transposed layouts give for free from the same scratch.
    // Halves: D_top(8x8) = A_top(8x16) B(16x8) + C_top, and the same for the bottom rows, with
    // the accumulator in fp32.
    Id type_a;
    Id type_b;
    Id type_c;
    if (halves) {
        type_a = CooperativeMatrixType(ctx, f16, 8, 16, USE_MATRIX_A);
        type_b = CooperativeMatrixType(ctx, f16, 16, 8, USE_MATRIX_B);
        type_c = CooperativeMatrixType(ctx, ctx.F32[1], 8, 8, USE_ACCUMULATOR);
    } else if (transposed) {
        type_a = CooperativeMatrixType(ctx, f16, 8, 16, USE_MATRIX_A);
        type_b = CooperativeMatrixType(ctx, f16, 16, 16, USE_MATRIX_B);
        type_c = CooperativeMatrixType(ctx, f16, 8, 16, USE_ACCUMULATOR);
    } else {
        type_a = CooperativeMatrixType(ctx, f16, 16, 16, USE_MATRIX_A);
        type_b = CooperativeMatrixType(ctx, f16, 16, 8, USE_MATRIX_B);
        type_c = CooperativeMatrixType(ctx, f16, 16, 8, USE_ACCUMULATOR);
    }
    const Id row_major = imm(LAYOUT_ROW_MAJOR);
    const Id column_major = imm(LAYOUT_COLUMN_MAJOR);

    const Id type = ctx.TypeFunction(ctx.U32[2], ctx.U32[4], ctx.U32[2], ctx.U32[2]);
    const Id fn = ctx.OpFunction(ctx.U32[2], spv::FunctionControlMask::DontInline, type);
    ctx.cooperative_matrix_function = fn;
    const Id matrix_a = ctx.OpFunctionParameter(ctx.U32[4]);
    const Id matrix_b = ctx.OpFunctionParameter(ctx.U32[2]);
    const Id matrix_c = ctx.OpFunctionParameter(ctx.U32[2]);
    ctx.AddLabel();

    const Id subgroup_scope = imm(static_cast<u32>(spv::Scope::Subgroup));
    const Id semantics = imm(static_cast<u32>(spv::MemorySemanticsMask::WorkgroupMemory) |
                             static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease));
    const auto barrier = [&] { ctx.OpControlBarrier(subgroup_scope, subgroup_scope, semantics); };
    const auto at = [&](Id offset) { return ctx.OpAccessChain(word_pointer, scratch, offset); };
    const auto store = [&](Id offset, Id value) { ctx.OpStore(at(offset), value); };
    const auto load = [&](Id offset) { return ctx.OpLoad(u32ty, at(offset)); };
    const auto marker = [&](u32 op, Id result_type, std::initializer_list<Id> operands) {
        return ctx.OpExtInst(result_type, ctx.cooperative_matrix_marker, op,
                             std::span<const Id>(operands.begin(), operands.size()));
    };

    // The guest's m16n8k16 fragments: lane g*4+t holds A rows g and g+8 at k = 2t, 2t+1 (+8),
    // B column g at k = 2t, 2t+1 (+8), and C/D rows g and g+8 at n = 2t, 2t+1.
    const Id lane = ctx.OpBitwiseAnd(u32ty, ctx.OpLoad(u32ty, ctx.subgroup_local_invocation_id),
                                     imm(31));
    const Id g = ctx.OpShiftRightLogical(u32ty, lane, imm(2));
    const Id t = ctx.OpBitwiseAnd(u32ty, lane, imm(3));
    const Id base = ctx.OpIMul(u32ty, ctx.OpLoad(u32ty, subgroup_id), imm(scratch_words));
    const Id a_base = addi(base, SCRATCH_A);
    const Id bt_base = addi(base, SCRATCH_BT);
    const Id c_base = addi(base, SCRATCH_C);
    const Id g8t = add(ctx.OpShiftLeftLogical(u32ty, g, imm(3)), t);
    const Id g4t = add(ctx.OpShiftLeftLogical(u32ty, g, imm(2)), t);
    const Id a_row = add(a_base, g8t);
    const Id bt_row = add(bt_base, g8t);
    const Id c_row = add(c_base, g4t);
    const auto extract = [&](Id composite, u32 index) {
        return ctx.OpCompositeExtract(u32ty, composite, index);
    };
    store(a_row, extract(matrix_a, 0));
    store(addi(a_row, 64), extract(matrix_a, 1));
    store(addi(a_row, 4), extract(matrix_a, 2));
    store(addi(a_row, 68), extract(matrix_a, 3));
    store(bt_row, extract(matrix_b, 0));
    store(addi(bt_row, 4), extract(matrix_b, 1));
    if (halves) {
        // C rows g and g+8 at n = 2t, 2t+1, widened to floats.
        const Id f32ty = ctx.F32[1];
        const Id c32_base = addi(base, SCRATCH_C32);
        const Id c32_row = add(add(c32_base, ctx.OpShiftLeftLogical(u32ty, g, imm(3))),
                               ctx.OpShiftLeftLogical(u32ty, t, imm(1)));
        for (u32 row = 0; row < 2; ++row) {
            const Id pair = ctx.OpUnpackHalf2x16(ctx.F32[2], extract(matrix_c, row));
            for (u32 col = 0; col < 2; ++col) {
                store(addi(c32_row, row * 64 + col),
                      ctx.OpBitcast(u32ty, ctx.OpCompositeExtract(f32ty, pair, col)));
            }
        }
        barrier();
        const Id b = marker(MARKER_LOAD, type_b, {at(bt_base), column_major, imm(8)});
        for (u32 half = 0; half < 2; ++half) {
            const Id a = marker(MARKER_LOAD, type_a, {at(addi(a_base, half * 64)), row_major, imm(8)});
            const Id c_at = at(addi(c32_base, half * 64));
            const Id c = marker(MARKER_LOAD, type_c, {c_at, row_major, imm(8)});
            const Id d = marker(MARKER_MUL_ADD, type_c, {a, b, c});
            marker(MARKER_STORE, u32ty, {c_at, d, row_major, imm(8)});
        }
        barrier();
        std::array<Id, 2> packed;
        for (u32 row = 0; row < 2; ++row) {
            const Id lo = ctx.OpBitcast(f32ty, load(addi(c32_row, row * 64)));
            const Id hi = ctx.OpBitcast(f32ty, load(addi(c32_row, row * 64 + 1)));
            packed[row] = ctx.OpPackHalf2x16(u32ty, ctx.OpCompositeConstruct(ctx.F32[2], lo, hi));
        }
        // The next evaluation's stores wait for every lane's reads.
        barrier();
        ctx.OpReturnValue(ctx.OpCompositeConstruct(ctx.U32[2], packed[0], packed[1]));
        ctx.OpFunctionEnd();
        return;
    }
    store(c_row, extract(matrix_c, 0));
    store(addi(c_row, 32), extract(matrix_c, 1));
    barrier();
    if (transposed) {
        const Id a = marker(MARKER_LOAD, type_a, {at(bt_base), row_major, imm(8)});
        const Id b = marker(MARKER_LOAD, type_b, {at(a_base), column_major, imm(8)});
        const Id c = marker(MARKER_LOAD, type_c, {at(c_base), column_major, imm(4)});
        const Id d = marker(MARKER_MUL_ADD, type_c, {a, b, c});
        marker(MARKER_STORE, u32ty, {at(c_base), d, column_major, imm(4)});
    } else {
        const Id a = marker(MARKER_LOAD, type_a, {at(a_base), row_major, imm(8)});
        const Id b = marker(MARKER_LOAD, type_b, {at(bt_base), column_major, imm(8)});
        const Id c = marker(MARKER_LOAD, type_c, {at(c_base), row_major, imm(4)});
        const Id d = marker(MARKER_MUL_ADD, type_c, {a, b, c});
        marker(MARKER_STORE, u32ty, {at(c_base), d, row_major, imm(4)});
    }
    barrier();
    const Id d0 = load(c_row);
    const Id d1 = load(addi(c_row, 32));
    // The next evaluation's stores wait for every lane's reads.
    barrier();
    ctx.OpReturnValue(ctx.OpCompositeConstruct(ctx.U32[2], d0, d1));
    ctx.OpFunctionEnd();
}
} // Anonymous namespace

void DefineSoftwareMatrixFunctions(EmitContext& ctx, const IR::Program& program) {
    if (UsesTensor(program) && UsesCooperativeMatrix(ctx, program)) {
        DefineCooperativeMatrixFunction(ctx, program);
        return;
    }
    std::array<bool, 2> needed{};
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::TensorMulAdd || inst.Flags<u32>() == 0) continue;
            const u32 depth = inst.Flags<u32>();
            if (depth != 8 && depth != 16) throw LogicError("Invalid software HMMA depth");
            needed[depth == 16] = true;
        }
    }
    for (u32 variant = 0; variant < 2; ++variant) {
        if (!needed[variant]) continue;
        ctx.AddCapability(spv::Capability::GroupNonUniform);
        ctx.AddCapability(spv::Capability::GroupNonUniformShuffle);
        if (!ctx.profile.SupportsSubgroupStage(program.stage) ||
            ctx.profile.has_broken_spirv_subgroup_shuffle)
            throw NotImplementedException("Software HMMA requires subgroup shuffle support");
        const Id u32ty = ctx.U32[1], f32ty = ctx.F32[1];
        const auto imm = [&](u32 v) { return ctx.Const(v); };
        const auto add = [&](Id a, Id b) { return ctx.OpIAdd(u32ty, a, b); };
        const auto and_ = [&](Id a, u32 b) { return ctx.OpBitwiseAnd(u32ty, a, imm(b)); };
        const auto unpack = [&](Id v) { return ctx.OpUnpackHalf2x16(ctx.F32[2], v); };
        const auto round_half = [&](Id value) {
            // Do not express the intermediate rounding as unpack(pack(value)): drivers may
            // eliminate that round trip. Round normal significands explicitly, ties to even;
            // half subnormals lie on the exactly representable 2^-24 grid.
            const Id bits = ctx.OpBitcast(u32ty, value);
            const Id magnitude = and_(bits, 0x7fffffffU);
            const Id tie = and_(ctx.OpShiftRightLogical(u32ty, magnitude, imm(13)), 1);
            const Id rounded = and_(add(add(magnitude, imm(0xfff)), tie), ~0x1fffU);
            const Id normal = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, rounded, imm(0x47800000)), imm(0x7f800000), rounded);
            const Id scaled = ctx.OpFMul(f32ty, ctx.OpBitcast(f32ty, magnitude), ctx.Const(16777216.0f));
            const Id subnormal = ctx.OpBitcast(u32ty,
                ctx.OpFMul(f32ty, ctx.OpRoundEven(f32ty, scaled), ctx.Const(0x1p-24f)));
            const Id finite = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, magnitude, imm(0x38800000)), normal, subnormal);
            const Id result = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, magnitude, imm(0x7f800000)), magnitude, finite);
            return ctx.OpBitcast(f32ty, ctx.OpBitwiseOr(u32ty, result, and_(bits, 0x80000000)));
        };
        const Id type = ctx.TypeFunction(ctx.U32[2], ctx.U32[4], ctx.U32[2], ctx.U32[2]);
        const Id fn = ctx.OpFunction(ctx.U32[2], spv::FunctionControlMask::DontInline, type);
        ctx.software_matrix_functions[variant] = fn;
        const Id matrix_a = ctx.OpFunctionParameter(ctx.U32[4]);
        const Id matrix_b = ctx.OpFunctionParameter(ctx.U32[2]);
        const Id matrix_c = ctx.OpFunctionParameter(ctx.U32[2]);
        ctx.AddLabel();
        const Id counter = ctx.AddLocalVariable(ctx.TypePointer(spv::StorageClass::Function, u32ty),
                                           spv::StorageClass::Function);
        std::array<Id, 4> accumulators;
        for (auto& acc : accumulators)
            acc = ctx.AddLocalVariable(ctx.TypePointer(spv::StorageClass::Function, f32ty),
                                 spv::StorageClass::Function);
        std::array<Id, 8> inputs;
        for (u32 i = 0; i < 4; ++i)
            inputs[i] = ctx.OpCompositeExtract(u32ty, matrix_a, i);
        for (u32 i = 0; i < 2; ++i) {
            inputs[4 + i] = ctx.OpCompositeExtract(u32ty, matrix_b, i);
            inputs[6 + i] = ctx.OpCompositeExtract(u32ty, matrix_c, i);
        }
        for (u32 row = 0; row < 2; ++row) {
            const Id c = unpack(inputs[6 + row]);
            for (u32 col = 0; col < 2; ++col)
                ctx.OpStore(accumulators[row * 2 + col], ctx.OpCompositeExtract(f32ty, c, col));
        }
        ctx.OpStore(counter, imm(0));
        const Id lane = ctx.OpLoad(u32ty, ctx.subgroup_local_invocation_id);
        const Id warp_base = and_(lane, ~31U);
        const Id row_base = and_(lane, 28);
        const Id col_base = ctx.OpShiftLeftLogical(u32ty, and_(lane, 3), imm(3));
        const Id header = ctx.OpLabel(), body = ctx.OpLabel(), next = ctx.OpLabel(), end = ctx.OpLabel();
        ctx.OpBranch(header);
        ctx.AddLabel(header);
        const Id pair = ctx.OpLoad(u32ty, counter);
        const Id more = ctx.OpULessThan(ctx.U1, pair, imm(variant ? 8 : 4));
        ctx.OpLoopMerge(end, next, spv::LoopControlMask::DontUnroll);
        ctx.OpBranchConditional(more, body, end);
        ctx.AddLabel(body);
        const Id upper = ctx.OpUGreaterThanEqual(ctx.U1, pair, imm(4));
        const Id component = and_(pair, 3);
        const Id a_lane = add(warp_base, add(row_base, component));
        const Id b_lane = add(warp_base, add(col_base, component));
        const Id b = ctx.OpSelect(u32ty, upper, inputs[5], inputs[4]);
        std::array<Id, 2> bs;
        for (u32 col = 0; col < 2; ++col)
            bs[col] = unpack(ctx.OpGroupNonUniformShuffle(u32ty, imm(3), b,
                                                          add(b_lane, imm(col * 4))));
        const Id round_now = ctx.OpIEqual(ctx.U1, component, imm(3));
        for (u32 row = 0; row < 2; ++row) {
            const Id a = ctx.OpSelect(u32ty, upper, inputs[row + 2], inputs[row]);
            const Id as = unpack(ctx.OpGroupNonUniformShuffle(u32ty, imm(3), a, a_lane));
            for (u32 col = 0; col < 2; ++col) {
                Id value = ctx.OpLoad(f32ty, accumulators[row * 2 + col]);
                for (u32 half = 0; half < 2; ++half) {
                    value = ctx.OpFma(f32ty, ctx.OpCompositeExtract(f32ty, as, half),
                                      ctx.OpCompositeExtract(f32ty, bs[col], half), value);
                    ctx.Decorate(value, spv::Decoration::NoContraction);
                }
                ctx.OpStore(accumulators[row * 2 + col], value);
            }
        }
        // HMMA.F16 rounds after each K=8 group, including the midpoint of K=16.
        // OpSelect evaluates both operands: keep the expensive half conversion in
        // a branch so the other three pairs do no rounding work. The counter is
        // identical in every participating lane and all shuffles precede this branch.
        const Id round_block = ctx.OpLabel(), round_merge = ctx.OpLabel();
        ctx.OpSelectionMerge(round_merge, spv::SelectionControlMask::DontFlatten);
        ctx.OpBranchConditional(round_now, round_block, round_merge);
        ctx.AddLabel(round_block);
        for (const Id accumulator : accumulators) {
            ctx.OpStore(accumulator, round_half(ctx.OpLoad(f32ty, accumulator)));
        }
        ctx.OpBranch(round_merge);
        ctx.AddLabel(round_merge);
        ctx.OpBranch(next);
        ctx.AddLabel(next);
        ctx.OpStore(counter, add(pair, imm(1)));
        ctx.OpBranch(header);
        ctx.AddLabel(end);
        std::array<Id, 2> packed;
        for (u32 row = 0; row < 2; ++row)
            packed[row] = ctx.OpPackHalf2x16(u32ty, ctx.OpCompositeConstruct(ctx.F32[2],
                ctx.OpLoad(f32ty, accumulators[row * 2]), ctx.OpLoad(f32ty, accumulators[row * 2 + 1])));
        ctx.OpReturnValue(ctx.OpCompositeConstruct(ctx.U32[2], packed[0], packed[1]));
        ctx.OpFunctionEnd();
    }
}

Id EmitTensorMulAdd(EmitContext& ctx, IR::Inst* inst, Id a, Id b, Id c) {
    const u32 depth = inst->Flags<u32>();
    if (depth != 8 && depth != 16) {
        throw NotImplementedException("Unsupported matrix multiply depth");
    }
    if (Sirit::ValidId(ctx.cooperative_matrix_function)) {
        // The translator zeroes the K 8..15 registers of a depth-8 product, so one K=16 shape
        // serves both depths.
        return ctx.OpFunctionCall(ctx.U32[2], ctx.cooperative_matrix_function, a, b, c);
    }
    return ctx.OpFunctionCall(ctx.U32[2], ctx.software_matrix_functions[depth == 16], a, b, c);
}

// Rewrites the cooperative matrix markers into the KHR instructions: the placeholder array
// types into OpTypeCooperativeMatrixKHR, and the OpExtInst markers into the load, store and
// multiply-add. Instruction lengths change, so the module is rebuilt.
void PatchCooperativeMatrix(std::vector<u32>& code, const EmitContext& ctx) {
    if (!Sirit::ValidId(ctx.cooperative_matrix_marker)) {
        return;
    }
    const u32 marker = ctx.cooperative_matrix_marker.value;
    std::vector<u32> result;
    result.reserve(code.size() + 32);
    result.insert(result.end(), code.begin(), code.begin() + 5);
    const auto header = [](u32 words, u32 opcode) { return words << 16 | opcode; };
    size_t offset = 5;
    while (offset < code.size()) {
        const u32 word_count = code[offset] >> 16;
        const u32 opcode = code[offset] & 0xFFFFu;
        if (word_count == 0 || offset + word_count > code.size()) {
            throw LogicError("Malformed SPIR-V while patching cooperative matrices");
        }
        const std::span<const u32> inst(code.data() + offset, word_count);
        offset += word_count;
        if (opcode == OP_TYPE_ARRAY) {
            const auto it = std::ranges::find(ctx.cooperative_matrix_types, inst[1],
                                              [](const auto& entry) { return entry.type.value; });
            if (it != ctx.cooperative_matrix_types.end()) {
                result.insert(result.end(), {header(7, OP_TYPE_COOPERATIVE_MATRIX_KHR), inst[1],
                                             inst[2], it->scope.value, it->rows.value,
                                             it->columns.value, it->use.value});
                continue;
            }
        } else if (opcode == OP_EXT_INST && inst[3] == marker) {
            switch (inst[4]) {
            case MARKER_LOAD:
                result.insert(result.end(), {header(6, OP_COOPERATIVE_MATRIX_LOAD_KHR), inst[1],
                                             inst[2], inst[5], inst[6], inst[7]});
                continue;
            case MARKER_STORE:
                result.insert(result.end(), {header(5, OP_COOPERATIVE_MATRIX_STORE_KHR), inst[5],
                                             inst[6], inst[7], inst[8]});
                continue;
            case MARKER_MUL_ADD:
                result.insert(result.end(), {header(6, OP_COOPERATIVE_MATRIX_MUL_ADD_KHR),
                                             inst[1], inst[2], inst[5], inst[6], inst[7]});
                continue;
            default:
                throw LogicError("Unknown cooperative matrix marker");
            }
        }
        result.insert(result.end(), inst.begin(), inst.end());
    }
    code = std::move(result);
}

} // namespace Shader::Backend::SPIRV
