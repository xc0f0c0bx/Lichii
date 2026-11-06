// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Intel XeSS super resolution in place of a game's DLSS. Only the SDK's headers are compiled in:
// libxess.dll (redistributable, copied next to the executable by the build) is loaded at run time
// and resolved by name, so a build without it still runs and falls back to FSR 3.1.

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// xess_vk.h includes vulkan/vulkan.h itself; our wrapper must go first so the platform
// (VK_USE_PLATFORM_*) sections are part of that first inclusion.
#include "video_core/vulkan_common/vulkan.h"

#include <xess/xess.h>
#include <xess/xess_vk.h>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Xess {

/// libxess.dll's entry points, resolved when it is loaded.
struct Api {
    decltype(&xessGetVersion) get_version{};
    decltype(&xessVKGetRequiredInstanceExtensions) get_instance_extensions{};
    decltype(&xessVKGetRequiredDeviceExtensions) get_device_extensions{};
    decltype(&xessVKGetRequiredDeviceFeatures) get_device_features{};
    decltype(&xessVKCreateContext) create_context{};
    decltype(&xessVKInit) init{};
    decltype(&xessVKExecute) execute{};
    decltype(&xessDestroyContext) destroy_context{};
    decltype(&xessGetOptimalInputResolution) get_optimal_input_resolution{};
    decltype(&xessSetJitterScale) set_jitter_scale{};
    decltype(&xessSetVelocityScale) set_velocity_scale{};
    decltype(&xessSetLoggingCallback) set_logging_callback{};
    decltype(&xessIsOptimalDriver) is_optimal_driver{};
    decltype(&xessGetProperties) get_properties{};
};

/// Where libxess.dll was found, or empty.
[[nodiscard]] std::filesystem::path FindLibrary();

/// When the Intel XeSS setting is on, loads libxess.dll and asks it what the instance needs.
/// Call before creating the renderer's instance.
bool Start();

/// Whether XeSS is loaded and waiting for the renderer's device.
[[nodiscard]] bool WantsDevice();

/// Instance extensions and the API version XeSS asks for (empty / 0 when not started).
[[nodiscard]] std::vector<std::string> InstanceExtensions();
[[nodiscard]] u32 MinApiVersion();

/// Device extensions XeSS asks for on this GPU; nullopt when it can't run there.
[[nodiscard]] std::optional<std::vector<std::string>> DeviceExtensions(VkInstance instance,
                                                                       VkPhysicalDevice physical);

/// Sets the features XeSS needs in the chain the device is created with (writing existing
/// structures and appending its own). Returns false when the GPU lacks them.
bool AddDeviceFeatures(VkInstance instance, VkPhysicalDevice physical,
                       VkPhysicalDeviceFeatures2& features2);

/// Creates XeSS's context on the renderer's device. Stops XeSS and returns false when it can't run.
bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device);

/// Whether XeSS can run on the attached device.
[[nodiscard]] bool IsAvailable();

/// The context created by AttachDevice.
[[nodiscard]] xess_context_handle_t Context();

/// Destroys the context; call before the device is destroyed.
void Stop();

[[nodiscard]] const Api& GetApi();

/// A result code's name, for logs.
[[nodiscard]] const char* ResultName(xess_result_t result);

} // namespace Vulkan::Xess
