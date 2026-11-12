/**
 * Screen-space ambient occlusion on the resolved main scene.
 *
 * The scene is found from its resolves: a raw depth copy followed by a color
 * copy from the same EDRAM surface into same-sized textures of at least
 * 1024x576 (with predicated tiling there is one pair per tile). Once the GPU
 * draws to a different surface, the occlusion is computed from the resolved
 * depth texture and multiplied into the resolved color texture, so it lands
 * before any post-processing that samples the scene, and after all tiles.
 * Works on the resolution-scaled resolve buffer only.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <rex/cvar.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/graphics/d3d12/render_target_cache.h>
#include <rex/graphics/d3d12/texture_cache.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/ui/d3d12/d3d12_util.h>

REXCVAR_DEFINE_DOUBLE(ssao_strength, 0.0, "GPU",
                      "Screen-space ambient occlusion on the 3D scene, 0 = off. Needs a "
                      "render scale above 1x")
    .range(0.0, 1.0);
REXCVAR_DEFINE_DOUBLE(ssao_radius, 10.0, "GPU",
                      "Ambient occlusion radius, in units of the camera's near plane")
    .range(0.01, 100.0);
REXCVAR_DEFINE_BOOL(ssao_debug, false, "GPU", "Show the ambient occlusion term alone");

namespace rex::graphics::d3d12 {

namespace shaders {
#include "../shaders/bytecode/d3d12_5_1/ssao_ao_cs.h"
#include "../shaders/bytecode/d3d12_5_1/ssao_apply_cs.h"
}  // namespace shaders

namespace {
constexpr uint32_t kSsaoMinWidth = 1024, kSsaoMinHeight = 576;
}  // namespace

void D3D12RenderTargetCache::SsaoNoteResolve(const draw_util::ResolveInfo& resolve_info) {
  if (REXCVAR_GET(ssao_strength) <= 0.0 || !IsDrawResolutionScaled() ||
      GetPath() != Path::kHostRenderTargets || !resolve_info.copy_dest_extent_length) {
    return;
  }
  const RegisterFile& regs = register_file();
  auto dest_pitch = regs.Get<reg::RB_COPY_DEST_PITCH>();
  bool scene_sized = dest_pitch.copy_dest_pitch >= kSsaoMinWidth &&
                     dest_pitch.copy_dest_height >= kSsaoMinHeight;
  // Depth copies always write 32bpp k_24_8(_FLOAT). shortcut: color only from
  // 8888 (what DOA4 resolves to), add 2_10_10_10 / float if another game needs them.
  bool dest_8888 =
      resolve_info.copy_dest_info.copy_dest_format == xenos::ColorFormat::k_8_8_8_8;
  // RB_COLOR_INFO / RB_DEPTH_INFO get repointed per copy, the surface stays.
  uint32_t surface_info = regs[XE_GPU_REG_RB_SURFACE_INFO];
  bool same_scene = ssao_.depth && ssao_.surface_info == surface_info &&
                    ssao_.dest_pitch == dest_pitch.value;
  if (resolve_info.IsCopyingDepth()) {
    if (!scene_sized ||
        resolve_info.rb_copy_control.copy_command != xenos::CopyCommand::kRaw) {
      return;
    }
    if (!same_scene) {
      ssao_ = {};
      ssao_.depth = true;
      ssao_.surface_info = surface_info;
      ssao_.dest_pitch = dest_pitch.value;
      ssao_.depth_base = resolve_info.copy_dest_base;
      ssao_.depth_endian = uint32_t(resolve_info.copy_dest_info.copy_dest_endian);
      ssao_.depth_float24 = regs.Get<reg::RB_DEPTH_INFO>().depth_format ==
                            xenos::DepthRenderTargetFormat::kD24FS8;
      auto depth_control = regs.Get<reg::RB_DEPTHCONTROL>();
      ssao_.depth_reversed = depth_control.z_enable &&
                             (depth_control.zfunc == xenos::CompareFunction::kGreater ||
                              depth_control.zfunc == xenos::CompareFunction::kGreaterEqual);
    }
    // Tiles resolve into parts of one texture; the lowest base is its origin.
    ssao_.depth_base = std::min(ssao_.depth_base, resolve_info.copy_dest_base);
    return;
  }
  if (!same_scene || !dest_8888) {
    return;
  }
  ssao_.color_base = ssao_.color ? std::min(ssao_.color_base, resolve_info.copy_dest_base)
                                 : resolve_info.copy_dest_base;
  ssao_.color_endian = uint32_t(resolve_info.copy_dest_info.copy_dest_endian);
  ssao_.color = true;
}

void D3D12RenderTargetCache::SsaoFlushIfLeavingSurface(D3D12TextureCache& texture_cache) {
  if (!ssao_.depth) {
    return;
  }
  if (register_file()[XE_GPU_REG_RB_SURFACE_INFO] == ssao_.surface_info) {
    return;
  }
  if (ssao_.color) {
    SsaoRun(texture_cache);
  }
  ssao_ = {};
}

void D3D12RenderTargetCache::SsaoRun(D3D12TextureCache& texture_cache) {
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  reg::RB_COPY_DEST_PITCH dest_pitch;
  dest_pitch.value = ssao_.dest_pitch;
  uint32_t scale_x = draw_resolution_scale_x(), scale_y = draw_resolution_scale_y();
  uint32_t pitch_aligned = (dest_pitch.copy_dest_pitch + 31) & ~uint32_t(31);
  uint32_t height_aligned = (dest_pitch.copy_dest_height + 31) & ~uint32_t(31);
  uint32_t extent = pitch_aligned * height_aligned * 4;
  uint32_t width = dest_pitch.copy_dest_pitch * scale_x;
  uint32_t height = dest_pitch.copy_dest_height * scale_y;

  if (!ssao_ao_pipeline_) {
    ssao_ao_pipeline_ = ui::d3d12::util::CreateComputePipeline(
        device, shaders::ssao_ao_cs, sizeof(shaders::ssao_ao_cs), resolve_copy_root_signature_);
    ssao_apply_pipeline_ = ui::d3d12::util::CreateComputePipeline(
        device, shaders::ssao_apply_cs, sizeof(shaders::ssao_apply_cs),
        resolve_copy_root_signature_);
    if (!ssao_ao_pipeline_ || !ssao_apply_pipeline_) {
      REXGPU_ERROR("D3D12RenderTargetCache: Failed to create the SSAO pipelines");
      return;
    }
  }
  if (width != ssao_texture_width_ || height != ssao_texture_height_) {
    if (ssao_texture_) {
      // shortcut: the scene size is fixed per game, so a change only happens on
      // the first frames; the old texture may still be in flight, so keep it.
      return;
    }
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16G16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ssao_texture_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesDefault,
            command_processor_.GetD3D12Provider().GetHeapFlagCreateNotZeroed(), &desc,
            ssao_texture_state_, nullptr, IID_PPV_ARGS(&ssao_texture_)))) {
      REXGPU_ERROR("D3D12RenderTargetCache: Failed to create the SSAO texture");
      return;
    }
    ssao_texture_width_ = width;
    ssao_texture_height_ = height;
  }

  uint32_t constants[4];
  constants[0] = width | (height << 16);
  constants[1] = (pitch_aligned >> 5) | (scale_x << 8) | (scale_y << 12) |
                 (uint32_t(ssao_.depth_float24) << 20) | (uint32_t(ssao_.depth_reversed) << 21) |
                 (uint32_t(REXCVAR_GET(ssao_debug)) << 22);
  float radius = float(REXCVAR_GET(ssao_radius)), strength = float(REXCVAR_GET(ssao_strength));
  std::memcpy(&constants[2], &radius, sizeof(float));
  std::memcpy(&constants[3], &strength, sizeof(float));
  static_assert(sizeof(draw_util::ResolveCopyShaderConstants::DestRelative) >= sizeof(constants),
                "The SSAO constants must fit in the resolve copy root constants");
  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  UINT groups_x = (width + 7) / 8, groups_y = (height + 7) / 8;

  // Pass 1: occlusion and log2 depth from the depth texture.
  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors[2];
  if (!texture_cache.MakeScaledResolveRangeCurrent(ssao_.depth_base, extent) ||
      !command_processor_.RequestOneUseSingleViewDescriptors(2, descriptors)) {
    return;
  }
  texture_cache.CreateCurrentScaledResolveRangeUintPow2SRV(descriptors[0].first, 2);
  texture_cache.TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
  uav_desc.Format = DXGI_FORMAT_R16G16_FLOAT;
  uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(ssao_texture_, nullptr, &uav_desc, descriptors[1].first);
  command_processor_.PushTransitionBarrier(ssao_texture_, ssao_texture_state_,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ssao_texture_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  constants[1] = (constants[1] & ~(uint32_t(3) << 16)) | (ssao_.depth_endian << 16);
  command_list.D3DSetComputeRootSignature(resolve_copy_root_signature_);
  command_list.D3DSetComputeRoot32BitConstants(0, 4, constants, 0);
  command_list.D3DSetComputeRootDescriptorTable(1, descriptors[1].second);
  command_list.D3DSetComputeRootDescriptorTable(2, descriptors[0].second);
  command_processor_.SetExternalPipeline(ssao_ao_pipeline_);
  command_processor_.SubmitBarriers();
  command_list.D3DDispatch(groups_x, groups_y, 1);

  // Pass 2: depth-aware blur, multiplied into the color texture in place.
  if (!texture_cache.MakeScaledResolveRangeCurrent(ssao_.color_base, extent) ||
      !command_processor_.RequestOneUseSingleViewDescriptors(2, descriptors)) {
    return;
  }
  texture_cache.CreateCurrentScaledResolveRangeUintPow2UAV(descriptors[0].first, 2);
  texture_cache.TransitionCurrentScaledResolveRange(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
  srv_desc.Format = DXGI_FORMAT_R16G16_FLOAT;
  srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc.Texture2D.MipLevels = 1;
  device->CreateShaderResourceView(ssao_texture_, &srv_desc, descriptors[1].first);
  command_processor_.PushTransitionBarrier(ssao_texture_, ssao_texture_state_,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  ssao_texture_state_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  constants[1] = (constants[1] & ~(uint32_t(3) << 16)) | (ssao_.color_endian << 16);
  command_list.D3DSetComputeRoot32BitConstants(0, 4, constants, 0);
  command_list.D3DSetComputeRootDescriptorTable(1, descriptors[0].second);
  command_list.D3DSetComputeRootDescriptorTable(2, descriptors[1].second);
  command_processor_.SetExternalPipeline(ssao_apply_pipeline_);
  // The resolve copies wrote this range as a UAV too, and a UAV -> UAV
  // transition emits no barrier, so order the writes explicitly.
  command_processor_.PushUAVBarrier(texture_cache.GetCurrentScaledResolveBufferResource());
  command_processor_.SubmitBarriers();
  command_list.D3DDispatch(groups_x, groups_y, 1);
  texture_cache.MarkCurrentScaledResolveRangeUAVWritesCommitNeeded();
  // Textures created from the color range must be reloaded with the occlusion.
  texture_cache.MarkRangeAsResolved(ssao_.color_base, extent);
}

void D3D12RenderTargetCache::SsaoShutdown() {
  ui::d3d12::util::ReleaseAndNull(ssao_texture_);
  ui::d3d12::util::ReleaseAndNull(ssao_apply_pipeline_);
  ui::d3d12::util::ReleaseAndNull(ssao_ao_pipeline_);
  ssao_texture_width_ = ssao_texture_height_ = 0;
  ssao_ = {};
}

}  // namespace rex::graphics::d3d12
