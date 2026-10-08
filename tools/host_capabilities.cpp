// Read-only host Vulkan capability inventory. Does not create a device or dispatch.
#include <vulkan/vulkan.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void check(VkResult result) { if(result!=VK_SUCCESS) throw std::runtime_error("Vulkan enumeration failed"); }
static void quote(const char* value) {
    std::cout << '"';
    for(;*value;++value) {
        if(*value=='"' || *value=='\\') std::cout << '\\';
        if(static_cast<unsigned char>(*value)<32) throw std::runtime_error("invalid device name");
        std::cout << *value;
    }
    std::cout << '"';
}
int main() {
    VkInstance instance{};
    try {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; info.pApplicationInfo=&app;
        check(vkCreateInstance(&info,nullptr,&instance));
        uint32_t count=0; check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
        std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
        std::cout << "{\"schema\":1,\"devices\":[";
        bool first=true;
        for(auto device:devices) {
            uint32_t n=0; check(vkEnumerateDeviceExtensionProperties(device,nullptr,&n,nullptr));
            std::vector<VkExtensionProperties> extensions(n);
            check(vkEnumerateDeviceExtensionProperties(device,nullptr,&n,extensions.data()));
            bool extension=false;
            for(const auto& e:extensions) extension|=std::strcmp(e.extensionName,VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)==0;
            VkPhysicalDeviceProperties plain{}; vkGetPhysicalDeviceProperties(device,&plain);
            bool control=extension || plain.apiVersion>=VK_API_VERSION_1_3;
            VkPhysicalDeviceSubgroupSizeControlProperties size{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
            VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            subgroup.pNext=control?&size:nullptr;
            VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props.pNext=&subgroup;
            vkGetPhysicalDeviceProperties2(device,&props);
            VkPhysicalDeviceSubgroupSizeControlFeatures sf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
            VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            f12.pNext=control?&sf:nullptr;
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext=&f12;
            vkGetPhysicalDeviceFeatures2(device,&features);
            bool wave32=control && sf.subgroupSizeControl && size.minSubgroupSize<=32 && size.maxSubgroupSize>=32 &&
                (size.requiredSubgroupSizeStages&VK_SHADER_STAGE_COMPUTE_BIT);
            if(!first) std::cout << ','; first=false;
            std::cout << "{\"name\":"; quote(plain.deviceName);
            std::cout << ",\"api_version\":" << plain.apiVersion << ",\"driver_version\":" << plain.driverVersion
                << ",\"subgroup_size\":" << subgroup.subgroupSize << ",\"subgroup_stages\":" << subgroup.supportedStages
                << ",\"subgroup_operations\":" << subgroup.supportedOperations
                << ",\"subgroup_size_control\":" << sf.subgroupSizeControl
                << ",\"min_subgroup_size\":" << size.minSubgroupSize << ",\"max_subgroup_size\":" << size.maxSubgroupSize
                << ",\"compute_wave32_requestable\":" << (wave32?"true":"false")
                << ",\"shader_float16\":" << f12.shaderFloat16 << ",\"shader_int64\":" << features.features.shaderInt64
                << ",\"storage_image_extended_formats\":" << features.features.shaderStorageImageExtendedFormats
                << ",\"max_compute_shared_memory\":" << plain.limits.maxComputeSharedMemorySize
                << ",\"max_compute_invocations\":" << plain.limits.maxComputeWorkGroupInvocations
                << ",\"max_storage_buffer_range\":" << plain.limits.maxStorageBufferRange << ",\"formats\":[";
            bool firstFormat=true;
            for(auto format:{VK_FORMAT_R16_SFLOAT,VK_FORMAT_R16G16_SFLOAT,VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_R32_SFLOAT}) {
                VkFormatProperties p{}; vkGetPhysicalDeviceFormatProperties(device,format,&p);
                if(!firstFormat) std::cout << ','; firstFormat=false;
                std::cout << "{\"vk_format\":" << format << ",\"optimal_features\":" << p.optimalTilingFeatures << '}';
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        vkDestroyInstance(instance,nullptr); return 0;
    } catch(const std::exception& error) {
        if(instance) vkDestroyInstance(instance,nullptr);
        std::cerr << error.what() << '\n'; return 1;
    }
}
