// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Intel XeSS super resolution in place of a game's DLSS evaluation.

#include <array>
#include <cstdlib>

#include "video_core/vulkan_common/vulkan.h"

#include <xess/xess_vk.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_xess_upscaler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/xess.h"

namespace Vulkan {
namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;

int EnvInt(const char* name, int fallback) {
    const char* const value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

f32 EnvFloat(const char* name, f32 fallback) {
    const char* const value = std::getenv(name);
    return value ? static_cast<f32>(std::strtod(value, nullptr)) : fallback;
}

constexpr std::array QUALITIES{
    XESS_QUALITY_SETTING_AA,          XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS,
    XESS_QUALITY_SETTING_ULTRA_QUALITY, XESS_QUALITY_SETTING_QUALITY,
    XESS_QUALITY_SETTING_BALANCED,    XESS_QUALITY_SETTING_PERFORMANCE,
    XESS_QUALITY_SETTING_ULTRA_PERFORMANCE,
};

// XeSS validates the input size against the quality setting's range: pick the setting whose
// range holds this render size, else the one whose optimal size is nearest.
xess_quality_settings_t QualityFor(xess_context_handle_t context, VkExtent2D render,
                                   VkExtent2D upscale) {
    const Xess::Api& api = Xess::GetApi();
    const xess_2d_t output{upscale.width, upscale.height};
    xess_quality_settings_t nearest = XESS_QUALITY_SETTING_BALANCED;
    u32 nearest_distance = ~0u;
    for (const xess_quality_settings_t quality : QUALITIES) {
        xess_2d_t optimal{};
        xess_2d_t min{};
        xess_2d_t max{};
        if (api.get_optimal_input_resolution(context, &output, quality, &optimal, &min, &max) !=
            XESS_RESULT_SUCCESS) {
            continue;
        }
        if (render.width >= min.x && render.width <= max.x && render.height >= min.y &&
            render.height <= max.y) {
            return quality;
        }
        const u32 distance = static_cast<u32>(std::abs(static_cast<s32>(render.width) -
                                                       static_cast<s32>(optimal.x)));
        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest = quality;
        }
    }
    return nearest;
}

VkImageMemoryBarrier Barrier(const XessUpscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

xess_vk_image_view_info ViewInfo(const XessUpscaler::Image& image, VkExtent2D size) {
    return xess_vk_image_view_info{
        .imageView = image.view,
        .image = image.image,
        .subresourceRange = image.range,
        .format = image.format,
        .width = size.width,
        .height = size.height,
    };
}

} // Anonymous namespace

XessUpscaler::XessUpscaler(const Device& device_, Scheduler& scheduler_,
                           MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!Xess::IsAvailable()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

XessUpscaler::~XessUpscaler() {
    // Not scheduler.Finish(): this runs when the renderer goes away, after the game's channels are
    // released, and a submit would touch their state. Nothing more gets recorded by then.
    void(device.GetLogical().WaitIdle());
}

void XessUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
}

bool XessUpscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    bool reset = in.reset;
    bool init = false;
    if (!output_image || in.render_size.width != render_extent.width ||
        in.render_size.height != render_extent.height ||
        in.upscale_size.width != upscale_extent.width ||
        in.upscale_size.height != upscale_extent.height) {
        CreateOutput(in.upscale_size);
        render_extent = in.render_size;
        upscale_extent = in.upscale_size;
        init = true;
        reset = true;
    }
    // XeSS wants motion in render pixels from this frame to the previous one, which is what the
    // game's own DLSS gets. EDEN_NX2_XESS_MV_SCALE and EDEN_NX2_XESS_JITTER_SCALE (default 1)
    // multiply the motion vectors and the jitter for A/B runs; -1 flips a convention.
    static const f32 mv_scale = EnvFloat("EDEN_NX2_XESS_MV_SCALE", 1.0f);
    static const f32 jitter_scale = EnvFloat("EDEN_NX2_XESS_JITTER_SCALE", 1.0f);
    const std::array<f32, 2> wanted_velocity_scale{in.motion_vector_scale[0] * mv_scale,
                                                   in.motion_vector_scale[1] * mv_scale};
    const bool velocity_changed = wanted_velocity_scale != velocity_scale;
    velocity_scale = wanted_velocity_scale;

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, reset, init, velocity_changed,
                      velocity = wanted_velocity_scale, output_image_handle = *output_image,
                      output_view_handle = *output_view](vk::CommandBuffer cmdbuf) {
        const Xess::Api& api = Xess::GetApi();
        const xess_context_handle_t context = Xess::Context();
        // The caller draws the output this frame even when XeSS writes nothing to it: clear it
        // then, rather than show undefined contents (FSR 3.1 takes over from the next frame).
        const auto clear_output = [&] {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = output_image_handle,
                    .subresourceRange = range,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                   barrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT));
            cmdbuf.ClearColorImage(output_image_handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VkClearColorValue{}, range);
            cmdbuf.PipelineBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
        };
        const auto fail = [&](const char* stage, xess_result_t result) {
            if (!failed.exchange(true)) {
                LOG_ERROR(Render_Vulkan, "Intel XeSS {} failed ({}); using FSR 3.1", stage,
                          Xess::ResultName(result));
            }
            clear_output();
        };

        if (init) {
            // The game's own setup (Fortune's Weave: render-size motion vectors, reversed depth,
            // linear colour above 1 with no exposure passed), so XeSS gets the HDR path with its
            // own auto exposure. EDEN_NX2_XESS_LDR=1 / EDEN_NX2_XESS_AUTO_EXPOSURE=0 change that.
            uint32_t flags = XESS_INIT_FLAG_INVERTED_DEPTH;
            if (EnvInt("EDEN_NX2_XESS_AUTO_EXPOSURE", 1) != 0) {
                flags |= XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;
            }
            if (EnvInt("EDEN_NX2_XESS_LDR", 0) != 0) {
                flags |= XESS_INIT_FLAG_LDR_INPUT_COLOR;
            }
            const xess_quality_settings_t quality =
                QualityFor(context, in.render_size, in.upscale_size);
            const xess_vk_init_params_t params{
                .outputResolution = {in.upscale_size.width, in.upscale_size.height},
                .qualitySetting = quality,
                .initFlags = flags,
                .creationNodeMask = 0,
                .visibleNodeMask = 0,
                .tempBufferHeap = VK_NULL_HANDLE,
                .bufferHeapOffset = 0,
                .tempTextureHeap = VK_NULL_HANDLE,
                .textureHeapOffset = 0,
                .pipelineCache = VK_NULL_HANDLE,
            };
            if (const xess_result_t result = api.init(context, &params);
                result != XESS_RESULT_SUCCESS) {
                fail("initialization", result);
                return;
            }
            api.set_jitter_scale(context, jitter_scale, jitter_scale);
            LOG_INFO(Render_Vulkan, "Intel XeSS: {}x{} -> {}x{}, quality setting {}, flags {:#x}",
                     in.render_size.width, in.render_size.height, in.upscale_size.width,
                     in.upscale_size.height, static_cast<int>(quality), flags);
        }
        if (init || velocity_changed) {
            api.set_velocity_scale(context, velocity[0], velocity[1]);
        }

        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // XeSS wants them in SHADER_READ_ONLY_OPTIMAL and the output in GENERAL.
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };
        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };

        xess_vk_execute_params_t params{};
        params.colorTexture = ViewInfo(in.color, in.render_size);
        params.velocityTexture = ViewInfo(in.motion_vectors, in.render_size);
        params.depthTexture = ViewInfo(in.depth, in.render_size);
        params.outputTexture = xess_vk_image_view_info{
            .imageView = output_view_handle,
            .image = output_image_handle,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .format = OUTPUT_FORMAT,
            .width = in.upscale_size.width,
            .height = in.upscale_size.height,
        };
        params.jitterOffsetX = in.jitter[0];
        params.jitterOffsetY = in.jitter[1];
        params.exposureScale = 1.0f;
        params.resetHistory = reset ? 1u : 0u;
        params.inputWidth = in.render_size.width;
        params.inputHeight = in.render_size.height;

        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);
        const xess_result_t result = api.execute(context, *cmdbuf, &params);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                               to_general);
        if (result != XESS_RESULT_SUCCESS) {
            fail("evaluation", result);
        }
    });
    // XeSS binds its own pipelines and descriptors on the command buffer.
    scheduler.InvalidateState();
    return true;
}

} // namespace Vulkan
