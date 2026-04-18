// vkrun.c — minimal Vulkan compute harness for SQTT/RGP capture via MESA_VK_TRACE
//
// Usage:  vkrun <shader.spv> [gx [gy [gz]]]
// Build:  gcc -O2 vkrun.c -lvulkan -o vkrun
//
// Picks the first AMD physical device, creates one SSBO (binding=0, 64 KiB),
// dispatches <gx,gy,gz> workgroups, and reads back the first 8 floats so the
// kernel's stores can't be DCE'd by the driver.
//
// The RGP capture itself is enabled by the RADV env vars set around the invocation:
//   MESA_VK_TRACE=rgp
//   MESA_VK_TRACE_TRIGGER=/tmp/rgp_trigger   (touch this file before submit)

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  fprintf(stderr, "Vulkan error %d at %s:%d (%s)\n", _r, __FILE__, __LINE__, #x); exit(1); } } while(0)

static uint32_t* load_spv(const char* path, size_t* out_bytes) {
  FILE* f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  uint32_t* buf = malloc(n);
  if (fread(buf, 1, n, f) != (size_t)n) { perror("read"); exit(1); }
  fclose(f); *out_bytes = n; return buf;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <shader.spv> [gx [gy [gz]]]\n", argv[0]); return 1; }
  const char* spv_path = argv[1];
  uint32_t gx = argc > 2 ? (uint32_t)atoi(argv[2]) : 1;
  uint32_t gy = argc > 3 ? (uint32_t)atoi(argv[3]) : 1;
  uint32_t gz = argc > 4 ? (uint32_t)atoi(argv[4]) : 1;

  VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "sqtt_probe", .apiVersion = VK_API_VERSION_1_2};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
  VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));

  uint32_t pd_count; vkEnumeratePhysicalDevices(inst, &pd_count, NULL);
  VkPhysicalDevice* pds = malloc(sizeof(*pds) * pd_count);
  vkEnumeratePhysicalDevices(inst, &pd_count, pds);
  VkPhysicalDevice pd = VK_NULL_HANDLE;
  for (uint32_t i = 0; i < pd_count; i++) {
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pds[i], &props);
    if (props.vendorID == 0x1002) { pd = pds[i]; fprintf(stderr, "Device: %s\n", props.deviceName); break; }
  }
  if (pd == VK_NULL_HANDLE) { fprintf(stderr, "No AMD device found.\n"); return 1; }

  uint32_t qf_count; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, NULL);
  VkQueueFamilyProperties* qfs = malloc(sizeof(*qfs) * qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qfs);
  uint32_t qf = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
  if (qf == UINT32_MAX) { fprintf(stderr, "No compute queue family.\n"); return 1; }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio};
  VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
  VkDevice dev; CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

  const size_t BSIZE = 64 * 1024;
  VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = BSIZE, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VkBuffer buf; CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
  VkMemoryRequirements mreq; vkGetBufferMemoryRequirements(dev, buf, &mreq);
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  uint32_t mem_idx = UINT32_MAX;
  VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((mreq.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & need) == need) { mem_idx = i; break; }
  if (mem_idx == UINT32_MAX) { fprintf(stderr, "No host-visible memory type.\n"); return 1; }
  VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = mreq.size, .memoryTypeIndex = mem_idx};
  VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
  vkBindBufferMemory(dev, buf, mem, 0);
  void* pmap; vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &pmap);
  memset(pmap, 0, BSIZE);
  vkUnmapMemory(dev, mem);

  VkDescriptorSetLayoutBinding dslb = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dslci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 1, .pBindings = &dslb};
  VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
  VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1, .pSetLayouts = &dsl};
  VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

  size_t spv_bytes; uint32_t* spv = load_spv(spv_path, &spv_bytes);
  VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = spv_bytes, .pCode = spv};
  VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
  free(spv);

  VkPipelineShaderStageCreateInfo ssci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"};
  VkComputePipelineCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = ssci, .layout = pl};
  VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

  VkDescriptorPoolSize dps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1};
  VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps};
  VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
  VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl};
  VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo dbi = {.buffer = buf, .offset = 0, .range = VK_WHOLE_SIZE};
  VkWriteDescriptorSet wds = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi};
  vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

  VkCommandPoolCreateInfo cpoolci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .queueFamilyIndex = qf};
  VkCommandPool cpool; CHECK(vkCreateCommandPool(dev, &cpoolci, NULL, &cpool));
  VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
  VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &cbbi);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
  vkCmdDispatch(cb, gx, gy, gz);
  vkEndCommandBuffer(cb);

  VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
  CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(queue));

  vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &pmap);
  fprintf(stderr, "out[0..7]:");
  float* fp = (float*)pmap;
  for (int i = 0; i < 8; i++) fprintf(stderr, " %.3f", fp[i]);
  fprintf(stderr, "\n");
  vkUnmapMemory(dev, mem);

  vkDeviceWaitIdle(dev);
  return 0;
}
