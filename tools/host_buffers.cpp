// Offline Vulkan runner for dense binding-ordered uniform/storage buffers.
#include <vulkan/vulkan.h>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(VkResult r,const char* what) { if(r!=VK_SUCCESS) throw std::runtime_error(std::string(what)+": "+std::to_string(r)); }
static uint32_t number(const char* text) {
    uint32_t n=0;const char* end=text+std::strlen(text);auto r=std::from_chars(text,end,n);
    if(r.ec!=std::errc{} || r.ptr!=end || !n) throw std::runtime_error("expected positive integer");return n;
}
static std::vector<uint8_t> read(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);auto size=f.tellg();
    if(!f || size<=0 || size>256*1024*1024 || size%4) throw std::runtime_error("invalid buffer file: "+path);
    std::vector<uint8_t> data(size);f.seekg(0);if(!f.read(reinterpret_cast<char*>(data.data()),size)) throw std::runtime_error("read failed");
    return data;
}
struct Buffer {
    VkBuffer handle{};VkDeviceMemory memory{};void* mapped{};VkDeviceSize size{};
    VkDescriptorType type{};std::vector<uint8_t> initial;std::string output;
};
struct Context {
    VkInstance instance{};VkDevice device{};VkDescriptorSetLayout setLayout{};VkDescriptorPool pool{};
    VkPipelineLayout pipelineLayout{};VkShaderModule shader{};VkPipeline pipeline{};VkCommandPool commands{};VkFence fence{};
    std::vector<Buffer> buffers;
    ~Context(){
        if(device){vkDeviceWaitIdle(device);if(fence)vkDestroyFence(device,fence,nullptr);if(commands)vkDestroyCommandPool(device,commands,nullptr);
          if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);if(shader)vkDestroyShaderModule(device,shader,nullptr);
          if(pipelineLayout)vkDestroyPipelineLayout(device,pipelineLayout,nullptr);if(pool)vkDestroyDescriptorPool(device,pool,nullptr);
          if(setLayout)vkDestroyDescriptorSetLayout(device,setLayout,nullptr);
          for(auto& b:buffers){if(b.mapped)vkUnmapMemory(device,b.memory);if(b.handle)vkDestroyBuffer(device,b.handle,nullptr);if(b.memory)vkFreeMemory(device,b.memory,nullptr);}
          vkDestroyDevice(device,nullptr);}if(instance)vkDestroyInstance(instance,nullptr);
    }
};
int main(int argc,char** argv){
 try{
    if(argc<5)throw std::runtime_error("usage: helixsr_host_buffers shader.spv groups_x u:file|r:file|w:file:bytes ...");
    auto codeBytes=read(argv[1]);if(codeBytes.size()<20 || codeBytes.size()%4 || *reinterpret_cast<const uint32_t*>(codeBytes.data())!=0x07230203)
      throw std::runtime_error("invalid SPIR-V");
    uint32_t groups=number(argv[2]);Context c;c.buffers.resize(argc-3);
    for(int i=3;i<argc;++i){
      std::string spec=argv[i];if(spec.size()<3 || spec[1]!=':')throw std::runtime_error("invalid buffer specification");
      auto& b=c.buffers[i-3];char mode=spec[0];std::string value=spec.substr(2);
      b.type=mode=='u'?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      if(mode=='u' || mode=='r'){b.initial=read(value);b.size=b.initial.size();}
      else if(mode=='w'){
        auto split=value.rfind(':');if(split==std::string::npos)throw std::runtime_error("write buffer requires path:bytes");
        b.output=value.substr(0,split);b.size=number(value.c_str()+split+1);
        if(b.size%4 || b.size>256*1024*1024)throw std::runtime_error("invalid output size");
      }else throw std::runtime_error("buffer mode must be u, r or w");
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;check(vkCreateInstance(&ci,nullptr,&c.instance),"instance");
    uint32_t count=0;check(vkEnumeratePhysicalDevices(c.instance,&count,nullptr),"device count");std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(c.instance,&count,devices.data()),"devices");VkPhysicalDevice physical{};uint32_t family=0;
    for(auto d:devices){uint32_t n=0;vkGetPhysicalDeviceQueueFamilyProperties(d,&n,nullptr);std::vector<VkQueueFamilyProperties> q(n);
      vkGetPhysicalDeviceQueueFamilyProperties(d,&n,q.data());for(uint32_t i=0;i<n;++i)if(q[i].queueCount&&(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT)){physical=d;family=i;break;}if(physical)break;}
    if(!physical)throw std::runtime_error("no compute device");
    VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(physical,&properties);
    if(groups>properties.limits.maxComputeWorkGroupCount[0])throw std::runtime_error("dispatch exceeds device limit");
    for(const auto& b:c.buffers)if((b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && b.size>properties.limits.maxUniformBufferRange)||
      (b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && b.size>properties.limits.maxStorageBufferRange))throw std::runtime_error("descriptor range exceeds device limit");
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};f11.pNext=&f12;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};features.pNext=&f11;vkGetPhysicalDeviceFeatures2(physical,&features);
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};e12.shaderFloat16=f12.shaderFloat16;e12.shaderInt8=f12.shaderInt8;e12.storageBuffer8BitAccess=f12.storageBuffer8BitAccess;
    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};e11.storageBuffer16BitAccess=f11.storageBuffer16BitAccess;e11.pNext=&e12;
    VkPhysicalDeviceFeatures enabled{};enabled.shaderInt64=features.features.shaderInt64;float priority=1;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};queueInfo.queueFamilyIndex=family;queueInfo.queueCount=1;queueInfo.pQueuePriorities=&priority;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};deviceInfo.pNext=&e11;deviceInfo.queueCreateInfoCount=1;deviceInfo.pQueueCreateInfos=&queueInfo;deviceInfo.pEnabledFeatures=&enabled;
    check(vkCreateDevice(physical,&deviceInfo,nullptr,&c.device),"device");VkQueue queue{};vkGetDeviceQueue(c.device,family,0,&queue);
    VkPhysicalDeviceMemoryProperties memory{};vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    for(auto& b:c.buffers){VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=b.size+256;bi.usage=b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER?VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      check(vkCreateBuffer(c.device,&bi,nullptr,&b.handle),"buffer");VkMemoryRequirements req{};vkGetBufferMemoryRequirements(c.device,b.handle,&req);uint32_t type=memory.memoryTypeCount;
      for(uint32_t i=0;i<memory.memoryTypeCount;++i)if((req.memoryTypeBits&(1u<<i))&&
        (memory.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
      if(type==memory.memoryTypeCount)throw std::runtime_error("coherent memory unavailable");VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=type;
      check(vkAllocateMemory(c.device,&ai,nullptr,&b.memory),"memory");check(vkBindBufferMemory(c.device,b.handle,b.memory,0),"bind memory");check(vkMapMemory(c.device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"map");
      std::memset(b.mapped,0xcd,b.size);if(!b.initial.empty())std::memcpy(b.mapped,b.initial.data(),b.size);std::memset(static_cast<uint8_t*>(b.mapped)+b.size,0xa5,256);}
    std::vector<VkDescriptorSetLayoutBinding> layouts(c.buffers.size());uint32_t uniforms=0,storage=0;
    for(uint32_t i=0;i<c.buffers.size();++i){layouts[i]={i,c.buffers[i].type,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};(c.buffers[i].type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER?uniforms:storage)++;}
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=layouts.size();li.pBindings=layouts.data();check(vkCreateDescriptorSetLayout(c.device,&li,nullptr,&c.setLayout),"layout");
    std::vector<VkDescriptorPoolSize> poolSizes;if(uniforms)poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,uniforms});if(storage)poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,storage});
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pi.maxSets=1;pi.poolSizeCount=poolSizes.size();pi.pPoolSizes=poolSizes.data();check(vkCreateDescriptorPool(c.device,&pi,nullptr,&c.pool),"pool");
    VkDescriptorSetAllocateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};si.descriptorPool=c.pool;si.descriptorSetCount=1;si.pSetLayouts=&c.setLayout;VkDescriptorSet set{};check(vkAllocateDescriptorSets(c.device,&si,&set),"set");
    std::vector<VkDescriptorBufferInfo> infos(c.buffers.size());std::vector<VkWriteDescriptorSet> writes(c.buffers.size());
    for(uint32_t i=0;i<c.buffers.size();++i){infos[i]={c.buffers[i].handle,0,c.buffers[i].size};writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=c.buffers[i].type;writes[i].pBufferInfo=&infos[i];}
    vkUpdateDescriptorSets(c.device,writes.size(),writes.data(),0,nullptr);VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&c.setLayout;check(vkCreatePipelineLayout(c.device,&pli,nullptr,&c.pipelineLayout),"pipeline layout");
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=codeBytes.size();mi.pCode=reinterpret_cast<const uint32_t*>(codeBytes.data());check(vkCreateShaderModule(c.device,&mi,nullptr,&c.shader),"shader");
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pci.layout=c.pipelineLayout;pci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};pci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pci.stage.module=c.shader;pci.stage.pName="main";check(vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&pci,nullptr,&c.pipeline),"pipeline");
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=family;check(vkCreateCommandPool(c.device,&cpi,nullptr,&c.commands),"command pool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=c.commands;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;VkCommandBuffer cmd{};check(vkAllocateCommandBuffers(c.device,&cai,&cmd),"command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(cmd,&begin),"begin");VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipelineLayout,0,1,&set,0,nullptr);vkCmdDispatch(cmd,groups,1,1);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);check(vkEndCommandBuffer(cmd),"end");
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(c.device,&fi,nullptr,&c.fence),"fence");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;check(vkQueueSubmit(queue,1,&submit,c.fence),"submit");check(vkWaitForFences(c.device,1,&c.fence,VK_TRUE,60000000000ull),"wait");
    for(auto& b:c.buffers){for(unsigned i=0;i<256;++i)if(static_cast<uint8_t*>(b.mapped)[b.size+i]!=0xa5)throw std::runtime_error("guard changed");
      if(!b.initial.empty()&&std::memcmp(b.mapped,b.initial.data(),b.size))throw std::runtime_error("readonly buffer changed");
      if(!b.output.empty()){std::ofstream f(b.output,std::ios::binary|std::ios::trunc);if(!f.write(static_cast<char*>(b.mapped),b.size))throw std::runtime_error("output write failed");}}
    std::cerr<<"host_device="<<properties.deviceName<<" api="<<properties.apiVersion<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
