// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Intel XeSS super resolution in place of a game's DLSS: loading libxess.dll and its Vulkan setup.

#include <cstdlib>

#include "common/dynamic_library.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/vulkan_common/xess.h"

namespace Vulkan::Xess {
namespace {

#ifdef _WIN32
constexpr const char* LIBRARY_NAME = "libxess.dll";
#else
constexpr const char* LIBRARY_NAME = "libxess.so";
#endif

struct State {
    Common::DynamicLibrary library;
    Api api;
    std::filesystem::path path;
    xess_context_handle_t context{};
    std::vector<std::string> instance_extensions;
    u32 min_api_version{};
    bool started{};
    bool attached{};
};

State& GetState() {
    static State state;
    return state;
}

std::filesystem::path ExeDirectory() {
#ifdef _WIN32
    return Common::FS::GetExeDirectory();
#else
    std::error_code error;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::filesystem::path{} : exe.parent_path();
#endif
}

/// EDEN_XESS_DIR, then next to the executable (where the build puts the DLL), then an "xess"
/// folder next to it.
std::vector<std::filesystem::path> SearchDirs() {
    std::vector<std::filesystem::path> dirs;
    if (const char* const dir = std::getenv("EDEN_XESS_DIR"); dir && *dir) {
        dirs.emplace_back(dir);
    }
    const std::filesystem::path exe_dir = ExeDirectory();
    if (!exe_dir.empty()) {
        dirs.push_back(exe_dir);
        dirs.push_back(exe_dir / "xess");
    }
    return dirs;
}

void LogFromXess(const char* message, xess_logging_level_t level) {
    switch (level) {
    case XESS_LOGGING_LEVEL_ERROR:
        LOG_ERROR(Render_Vulkan, "XeSS: {}", message);
        break;
    case XESS_LOGGING_LEVEL_WARNING:
        LOG_WARNING(Render_Vulkan, "XeSS: {}", message);
        break;
    case XESS_LOGGING_LEVEL_INFO:
        LOG_INFO(Render_Vulkan, "XeSS: {}", message);
        break;
    default:
        LOG_DEBUG(Render_Vulkan, "XeSS: {}", message);
        break;
    }
}

bool Load(State& state, const std::filesystem::path& path) {
    if (!state.library.Open(Common::FS::PathToUTF8String(path).c_str())) {
        LOG_ERROR(Render_Vulkan, "Could not load {}", Common::FS::PathToUTF8String(path));
        return false;
    }
    Api& api = state.api;
    const bool resolved =
        state.library.GetSymbol("xessGetVersion", &api.get_version) &&
        state.library.GetSymbol("xessVKGetRequiredInstanceExtensions",
                                &api.get_instance_extensions) &&
        state.library.GetSymbol("xessVKGetRequiredDeviceExtensions",
                                &api.get_device_extensions) &&
        state.library.GetSymbol("xessVKGetRequiredDeviceFeatures", &api.get_device_features) &&
        state.library.GetSymbol("xessVKCreateContext", &api.create_context) &&
        state.library.GetSymbol("xessVKInit", &api.init) &&
        state.library.GetSymbol("xessVKExecute", &api.execute) &&
        state.library.GetSymbol("xessDestroyContext", &api.destroy_context) &&
        state.library.GetSymbol("xessGetOptimalInputResolution",
                                &api.get_optimal_input_resolution) &&
        state.library.GetSymbol("xessSetJitterScale", &api.set_jitter_scale) &&
        state.library.GetSymbol("xessSetVelocityScale", &api.set_velocity_scale) &&
        state.library.GetSymbol("xessSetLoggingCallback", &api.set_logging_callback) &&
        state.library.GetSymbol("xessIsOptimalDriver", &api.is_optimal_driver) &&
        state.library.GetSymbol("xessGetProperties", &api.get_properties);
    if (!resolved) {
        LOG_ERROR(Render_Vulkan, "{} is not a usable XeSS library (missing entry points)",
                  Common::FS::PathToUTF8String(path));
        state.library.Close();
        return false;
    }
    state.path = path;
    return true;
}

} // Anonymous namespace

std::filesystem::path FindLibrary() {
    for (const std::filesystem::path& dir : SearchDirs()) {
        const std::filesystem::path candidate = dir / LIBRARY_NAME;
        if (Common::FS::IsFile(candidate)) {
            return candidate;
        }
    }
    return {};
}

bool Start() {
    State& state = GetState();
    if (state.started) {
        return true;
    }
    if (!Settings::values.intel_xess.GetValue()) {
        return false;
    }
    if (!state.library.IsOpen()) {
        const std::filesystem::path path = FindLibrary();
        if (path.empty()) {
            LOG_WARNING(Render_Vulkan,
                        "Intel XeSS is on, but {} is not next to the executable; using FSR 3.1",
                        LIBRARY_NAME);
            return false;
        }
        if (!Load(state, path)) {
            return false;
        }
    }
    const Api& api = state.api;
    xess_version_t version{};
    api.get_version(&version);

    uint32_t count{};
    const char* const* names{};
    uint32_t min_api_version{};
    if (const xess_result_t result = api.get_instance_extensions(&count, &names, &min_api_version);
        result != XESS_RESULT_SUCCESS) {
        LOG_WARNING(Render_Vulkan, "XeSS can't run with this Vulkan ({}); using FSR 3.1",
                    ResultName(result));
        return false;
    }
    state.instance_extensions.assign(names, names + count);
    state.min_api_version = min_api_version;
    LOG_INFO(Render_Vulkan, "Intel XeSS {}.{}.{} loaded from {}", version.major, version.minor,
             version.patch, Common::FS::PathToUTF8String(state.path));
    state.started = true;
    return true;
}

bool WantsDevice() {
    const State& state = GetState();
    return state.started && !state.attached;
}

std::vector<std::string> InstanceExtensions() {
    const State& state = GetState();
    return state.started ? state.instance_extensions : std::vector<std::string>{};
}

u32 MinApiVersion() {
    const State& state = GetState();
    return state.started ? state.min_api_version : 0;
}

std::optional<std::vector<std::string>> DeviceExtensions(VkInstance instance,
                                                         VkPhysicalDevice physical) {
    const State& state = GetState();
    if (!state.started) {
        return std::nullopt;
    }
    uint32_t count{};
    const char* const* names{};
    if (const xess_result_t result =
            state.api.get_device_extensions(instance, physical, &count, &names);
        result != XESS_RESULT_SUCCESS) {
        LOG_WARNING(Render_Vulkan, "XeSS is not supported on this GPU or driver ({})",
                    ResultName(result));
        return std::nullopt;
    }
    return std::vector<std::string>(names, names + count);
}

bool AddDeviceFeatures(VkInstance instance, VkPhysicalDevice physical,
                       VkPhysicalDeviceFeatures2& features2) {
    const State& state = GetState();
    if (!state.started) {
        return false;
    }
    void* chain = &features2;
    if (const xess_result_t result = state.api.get_device_features(instance, physical, &chain);
        result != XESS_RESULT_SUCCESS) {
        LOG_WARNING(Render_Vulkan, "This GPU lacks the Vulkan features XeSS needs ({})",
                    ResultName(result));
        return false;
    }
    return true;
}

bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device) {
    State& state = GetState();
    if (!WantsDevice()) {
        return false;
    }
    const Api& api = state.api;
    xess_context_handle_t context{};
    if (const xess_result_t result = api.create_context(instance, physical, device, &context);
        result != XESS_RESULT_SUCCESS || !context) {
        LOG_WARNING(Render_Vulkan,
                    "XeSS is not supported on this GPU or driver ({}); using FSR 3.1",
                    ResultName(result));
        Stop();
        return false;
    }
    state.context = context;
    state.attached = true;
    api.set_logging_callback(context,
                             Settings::values.renderer_debug.GetValue() ? XESS_LOGGING_LEVEL_DEBUG
                                                                        : XESS_LOGGING_LEVEL_WARNING,
                             LogFromXess);
    if (const xess_result_t result = api.is_optimal_driver(context);
        result == XESS_RESULT_WARNING_OLD_DRIVER) {
        LOG_WARNING(Render_Vulkan, "The graphics driver is older than XeSS recommends");
    }
    xess_properties_t properties{};
    if (api.get_properties(context, nullptr, &properties) == XESS_RESULT_SUCCESS) {
        LOG_INFO(Render_Vulkan, "Intel XeSS is available ({} MiB of scratch memory)",
                 (properties.tempBufferHeapSize + properties.tempTextureHeapSize) >> 20);
    } else {
        LOG_INFO(Render_Vulkan, "Intel XeSS is available");
    }
    return true;
}

bool IsAvailable() {
    const State& state = GetState();
    return state.started && state.attached;
}

xess_context_handle_t Context() {
    return GetState().context;
}

void Stop() {
    State& state = GetState();
    if (state.context) {
        state.api.destroy_context(state.context);
        state.context = nullptr;
    }
    if (state.started) {
        LOG_INFO(Render_Vulkan, "Intel XeSS stopped");
    }
    state.started = false;
    state.attached = false;
}

const Api& GetApi() {
    return GetState().api;
}

const char* ResultName(xess_result_t result) {
    switch (result) {
    case XESS_RESULT_WARNING_NONEXISTING_FOLDER:
        return "warning: nonexisting folder";
    case XESS_RESULT_WARNING_OLD_DRIVER:
        return "warning: old driver";
    case XESS_RESULT_SUCCESS:
        return "success";
    case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE:
        return "unsupported device";
    case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER:
        return "unsupported driver";
    case XESS_RESULT_ERROR_UNINITIALIZED:
        return "uninitialized";
    case XESS_RESULT_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY:
        return "device out of memory";
    case XESS_RESULT_ERROR_DEVICE:
        return "device error";
    case XESS_RESULT_ERROR_NOT_IMPLEMENTED:
        return "not implemented";
    case XESS_RESULT_ERROR_INVALID_CONTEXT:
        return "invalid context";
    case XESS_RESULT_ERROR_OPERATION_IN_PROGRESS:
        return "operation in progress";
    case XESS_RESULT_ERROR_UNSUPPORTED:
        return "unsupported";
    case XESS_RESULT_ERROR_CANT_LOAD_LIBRARY:
        return "can't load library";
    case XESS_RESULT_ERROR_WRONG_CALL_ORDER:
        return "wrong call order";
    case XESS_RESULT_ERROR_UNKNOWN:
        break;
    }
    return "unknown error";
}

} // namespace Vulkan::Xess
