// Offline Vulkan buffer runner: set 0 binding 0 input, binding 1 output, entry main.
#include <vulkan/vulkan.h>
#include <array>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static void check(VkResult result, const char* call) {
    if(result != VK_SUCCESS) throw std::runtime_error(std::string(call)+": "+std::to_string(result));
}
static uint32_t number(const char* text) {
    uint32_t n=0; const char* end=text+std::strlen(text);
    auto result=std::from_chars(text,end,n);
    if(result.ec!=std::errc{} || result.ptr!=end || n==0) throw std::runtime_error("expected positive integer");
    return n;
}
static std::vector<uint32_t> read(const char* path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    auto size=f.tellg();
    if(!f || size<=0 || size>256*1024*1024 || size%4) throw std::runtime_error("invalid input file size");
    std::vector<uint32_t> data(size/4); f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(data.data()),size)) throw std::runtime_error("input read failed");
    return data;
}
struct Context {
    VkInstance instance{}; VkDevice device{}; VkQueue queue{};
    VkDescriptorSetLayout setLayout{}; VkDescriptorPool descriptors{};
    VkPipelineLayout pipelineLayout{}; VkShaderModule shader{}; VkPipeline pipeline{};
    VkCommandPool commands{}; VkFence fence{};
    struct Buffer { VkBuffer handle{}; VkDeviceMemory memory{}; void* mapped{}; };
    std::array<Buffer,2> buffers{};
    ~Context() {
        if(device) {
            vkDeviceWaitIdle(device);
            if(fence) vkDestroyFence(device,fence,nullptr);
            if(commands) vkDestroyCommandPool(device,commands,nullptr);
            if(pipeline) vkDestroyPipeline(device,pipeline,nullptr);
            if(shader) vkDestroyShaderModule(device,shader,nullptr);
            if(pipelineLayout) vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
            if(descriptors) vkDestroyDescriptorPool(device,descriptors,nullptr);
            if(setLayout) vkDestroyDescriptorSetLayout(device,setLayout,nullptr);
            for(auto& b:buffers) {
                if(b.mapped) vkUnmapMemory(device,b.memory);
                if(b.handle) vkDestroyBuffer(device,b.handle,nullptr);
                if(b.memory) vkFreeMemory(device,b.memory,nullptr);
            }
            vkDestroyDevice(device,nullptr);
        }
        if(instance) vkDestroyInstance(instance,nullptr);
    }
};
int main(int argc,char** argv) {
    try {
        if(argc!=6 && argc!=7) throw std::runtime_error("usage: helixsr_host_compute shader.spv input.bin output.bin output_bytes groups_x [--wave32]");
        const bool wave32=argc==7;
        if(wave32 && std::strcmp(argv[6],"--wave32")) throw std::runtime_error("unknown runner option");
        auto code=read(argv[1]), input=read(argv[2]);
        if(code.size()<5 || code[0]!=0x07230203) throw std::runtime_error("invalid SPIR-V header");
        bool derivativeLinear=false;
        for(size_t i=5;i<code.size();) {
            const uint32_t words=code[i]>>16,opcode=code[i]&0xffffu;
            if(!words || words>code.size()-i) throw std::runtime_error("invalid SPIR-V instruction length");
            if(opcode==17 && words==2 && code[i+1]==5350) derivativeLinear=true;
            i+=words;
        }
        const uint32_t outputBytes=number(argv[4]), groups=number(argv[5]);
        if(outputBytes>256*1024*1024 || outputBytes%4) throw std::runtime_error("invalid output length");
        Context c;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; instance.pApplicationInfo=&app;
        check(vkCreateInstance(&instance,nullptr,&c.instance),"vkCreateInstance");
        uint32_t count=0; check(vkEnumeratePhysicalDevices(c.instance,&count,nullptr),"enumerate device count");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(c.instance,&count,devices.data()),"enumerate devices");
        VkPhysicalDevice physical{}; uint32_t family=0;
        for(auto device:devices) {
            uint32_t families=0; vkGetPhysicalDeviceQueueFamilyProperties(device,&families,nullptr);
            std::vector<VkQueueFamilyProperties> queues(families);
            vkGetPhysicalDeviceQueueFamilyProperties(device,&families,queues.data());
            for(uint32_t i=0;i<families;++i) if(queues[i].queueCount && (queues[i].queueFlags&VK_QUEUE_COMPUTE_BIT)) {
                physical=device; family=i; break;
            }
            if(physical) break;
        }
        if(!physical) throw std::runtime_error("no host compute device");
        VkPhysicalDeviceProperties properties{}; vkGetPhysicalDeviceProperties(physical,&properties);
        if(properties.apiVersion<VK_API_VERSION_1_2 || groups>properties.limits.maxComputeWorkGroupCount[0] ||
           outputBytes>properties.limits.maxStorageBufferRange || input.size()*4>properties.limits.maxStorageBufferRange)
            throw std::runtime_error("host device limits exceeded");
        std::cerr << "host_device=" << properties.deviceName << " api=" << properties.apiVersion << '\n';
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroupFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlProperties subgroupSize{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivativeFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
        const char* subgroupExtension=VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
        std::vector<const char*> requiredExtensions;
        if(wave32) requiredExtensions.push_back(subgroupExtension);
        if(derivativeLinear) requiredExtensions.push_back(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
        if(!requiredExtensions.empty()) {
            uint32_t n=0; check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&n,nullptr),"extension count");
            std::vector<VkExtensionProperties> extensions(n);
            check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&n,extensions.data()),"extensions");
            for(const char* required:requiredExtensions) {
                bool found=false;
                for(const auto& e:extensions) found|=std::strcmp(e.extensionName,required)==0;
                if(!found) throw std::runtime_error(std::string("required extension unavailable: ")+required);
            }
        }
        if(wave32) {
            VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props.pNext=&subgroupSize;
            vkGetPhysicalDeviceProperties2(physical,&props);
            f12.pNext=&subgroupFeatures;
        }
        if(derivativeLinear) {
            derivativeFeatures.pNext=f12.pNext; f12.pNext=&derivativeFeatures;
        }
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.pNext=&f12;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext=&f11;
        vkGetPhysicalDeviceFeatures2(physical,&features);
        if(derivativeLinear && !derivativeFeatures.computeDerivativeGroupLinear)
            throw std::runtime_error("compute derivative linear groups unavailable");
        if(wave32 && (!subgroupFeatures.subgroupSizeControl || subgroupSize.minSubgroupSize>32 ||
           subgroupSize.maxSubgroupSize<32 || !(subgroupSize.requiredSubgroupSizeStages&VK_SHADER_STAGE_COMPUTE_BIT)))
            throw std::runtime_error("wave32 subgroup size unavailable");
        VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        enable12.shaderFloat16=f12.shaderFloat16; enable12.shaderInt8=f12.shaderInt8;
        enable12.storageBuffer8BitAccess=f12.storageBuffer8BitAccess;
        VkPhysicalDeviceSubgroupSizeControlFeatures enableSubgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        if(wave32) { enableSubgroup.subgroupSizeControl=VK_TRUE; enable12.pNext=&enableSubgroup; }
        VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR enableDerivative{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
        if(derivativeLinear) {
            enableDerivative.computeDerivativeGroupLinear=VK_TRUE;
            enableDerivative.pNext=enable12.pNext; enable12.pNext=&enableDerivative;
        }
        VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        enable11.storageBuffer16BitAccess=f11.storageBuffer16BitAccess; enable11.pNext=&enable12;
        VkPhysicalDeviceFeatures enable{}; enable.shaderInt64=features.features.shaderInt64;
        const float priority=1;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex=family; queue.queueCount=1; queue.pQueuePriorities=&priority;
        VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device.pNext=&enable11; device.queueCreateInfoCount=1; device.pQueueCreateInfos=&queue; device.pEnabledFeatures=&enable;
        device.enabledExtensionCount=static_cast<uint32_t>(requiredExtensions.size());
        device.ppEnabledExtensionNames=requiredExtensions.data();
        check(vkCreateDevice(physical,&device,nullptr,&c.device),"vkCreateDevice");
        vkGetDeviceQueue(c.device,family,0,&c.queue);
        VkPhysicalDeviceMemoryProperties memory{}; vkGetPhysicalDeviceMemoryProperties(physical,&memory);
        const VkDeviceSize sizes[]={input.size()*4,outputBytes};
        for(unsigned i=0;i<2;++i) {
            auto& b=c.buffers[i];
            VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            buffer.size=sizes[i]+256; buffer.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            check(vkCreateBuffer(c.device,&buffer,nullptr,&b.handle),"vkCreateBuffer");
            VkMemoryRequirements requirements{}; vkGetBufferMemoryRequirements(c.device,b.handle,&requirements);
            uint32_t type=memory.memoryTypeCount;
            for(uint32_t j=0;j<memory.memoryTypeCount;++j)
                if((requirements.memoryTypeBits&(1u<<j)) &&
                   (memory.memoryTypes[j].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==
                     (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {type=j;break;}
            if(type==memory.memoryTypeCount) throw std::runtime_error("host-coherent buffer memory unavailable");
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize=requirements.size; allocation.memoryTypeIndex=type;
            check(vkAllocateMemory(c.device,&allocation,nullptr,&b.memory),"vkAllocateMemory");
            check(vkBindBufferMemory(c.device,b.handle,b.memory,0),"vkBindBufferMemory");
            check(vkMapMemory(c.device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"vkMapMemory");
            std::memset(b.mapped,0xcd,sizes[i]);
            std::memset(static_cast<char*>(b.mapped)+sizes[i],0xa5,256);
        }
        std::memcpy(c.buffers[0].mapped,input.data(),sizes[0]);
        VkDescriptorSetLayoutBinding bindings[2]{};
        for(unsigned i=0;i<2;++i) bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount=2; layout.pBindings=bindings;
        check(vkCreateDescriptorSetLayout(c.device,&layout,nullptr,&c.setLayout),"descriptor layout");
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets=1; pool.poolSizeCount=1; pool.pPoolSizes=&poolSize;
        check(vkCreateDescriptorPool(c.device,&pool,nullptr,&c.descriptors),"descriptor pool");
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool=c.descriptors; allocate.descriptorSetCount=1; allocate.pSetLayouts=&c.setLayout;
        VkDescriptorSet set{}; check(vkAllocateDescriptorSets(c.device,&allocate,&set),"descriptor allocation");
        VkDescriptorBufferInfo infos[2]{}; VkWriteDescriptorSet writes[2]{};
        for(unsigned i=0;i<2;++i) {
            infos[i]={c.buffers[i].handle,0,sizes[i]};
            writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet=set; writes[i].dstBinding=i; writes[i].descriptorCount=1;
            writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo=&infos[i];
        }
        vkUpdateDescriptorSets(c.device,2,writes,0,nullptr);
        VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayout.setLayoutCount=1; pipelineLayout.pSetLayouts=&c.setLayout;
        check(vkCreatePipelineLayout(c.device,&pipelineLayout,nullptr,&c.pipelineLayout),"pipeline layout");
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize=code.size()*4; module.pCode=code.data();
        check(vkCreateShaderModule(c.device,&module,nullptr,&c.shader),"shader module");
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.layout=c.pipelineLayout; pipeline.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; pipeline.stage.module=c.shader; pipeline.stage.pName="main";
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSize{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        requiredSize.requiredSubgroupSize=32;
        if(wave32) pipeline.stage.pNext=&requiredSize;
        check(vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&c.pipeline),"compute pipeline");
        VkCommandPoolCreateInfo commands{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; commands.queueFamilyIndex=family;
        check(vkCreateCommandPool(c.device,&commands,nullptr,&c.commands),"command pool");
        VkCommandBufferAllocateInfo command{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command.commandPool=c.commands; command.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; command.commandBufferCount=1;
        VkCommandBuffer cmd{}; check(vkAllocateCommandBuffers(c.device,&command,&cmd),"command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmd,&begin),"begin commands");
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipelineLayout,0,1,&set,0,nullptr);
        vkCmdDispatch(cmd,groups,1,1);
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        check(vkEndCommandBuffer(cmd),"end commands");
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check(vkCreateFence(c.device,&fence,nullptr,&c.fence),"fence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&cmd;
        check(vkQueueSubmit(c.queue,1,&submit,c.fence),"queue submit");
        check(vkWaitForFences(c.device,1,&c.fence,VK_TRUE,60000000000ull),"host computation timeout");
        for(unsigned i=0;i<2;++i) for(unsigned j=0;j<256;++j)
            if(static_cast<unsigned char*>(c.buffers[i].mapped)[sizes[i]+j]!=0xa5) throw std::runtime_error("buffer guard changed");
        if(std::memcmp(c.buffers[0].mapped,input.data(),sizes[0])) throw std::runtime_error("readonly input changed");
        std::ofstream output(argv[3],std::ios::binary|std::ios::trunc);
        if(!output.write(static_cast<const char*>(c.buffers[1].mapped),outputBytes)) throw std::runtime_error("output write failed");
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
