/**
 * High-resolution projected shadows (shadow_quality).
 *
 * DOA4 draws its fighters (a full-screen background triangle, then the
 * casters, all textureless) into a 256x256 8888 + depth surface from the light
 * and resolves it; blurs that texture twice with full-screen passes, each
 * resolved back to the same texture; then copies the surface to a second
 * texture, which the stage projects with one bilinear fetch. Here the caster
 * pass (textureless draws into a small single-sample 8888 + depth surface) is
 * drawn a second time into a private target at factor x the guest resolution,
 * with the host viewport derived for that scale. Its resolve is remembered; the
 * resolve of the blurred surface to a new texture (directly, or as a copy after
 * blurs in place) is accompanied by a Gaussian blur of the private copy, and
 * fetches of that texture's guest address sample the high-resolution result
 * instead. Only the caster silhouette and the blur are touched: the stage, the
 * projection and the shading stay the game's.
 */

#include <algorithm>
#include <cmath>
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

REXCVAR_DEFINE_INT32(shadow_quality, 0, "GPU",
                     "Fighter shadow resolution: 0 = original, 1 = 4x, 2 = 8x")
    .range(0, 2);
REXCVAR_DEFINE_DOUBLE(shadow_softness, 0.5, "GPU",
                      "Fighter shadow edge softness with shadow_quality, 1 = about as soft "
                      "as the original")
    .range(0.0, 2.0);

namespace rex::graphics::d3d12 {
namespace shaders {
#include "../shaders/bytecode/d3d12_5_1/shadow_blur_cs.h"
}  // namespace shaders

namespace {
// Widest surface treated as a shadow caster pass, in guest pixels.
constexpr uint32_t kShadowMaxSurfacePitch = 512;
// Width of the game's own blur, as a Gaussian sigma in guest texels: four
// bilinear taps a texel apart, run across and then down.
constexpr float kShadowGuestBlurSigma = 1.0f;
}  // namespace

bool D3D12RenderTargetCache::IsShadowEnabled() const {
  return REXCVAR_GET(shadow_quality) > 0 && GetPath() == Path::kHostRenderTargets;
}

void D3D12RenderTargetCache::ShadowTransition(ShadowTexture& texture,
                                              D3D12_RESOURCE_STATES state) {
  command_processor_.PushTransitionBarrier(texture.resource, texture.state, state);
  texture.state = state;
}

bool D3D12RenderTargetCache::ShadowEnsureResources(uint32_t surface_pitch,
                                                   DXGI_FORMAT color_format,
                                                   xenos::DepthRenderTargetFormat depth_format) {
  uint32_t factor = REXCVAR_GET(shadow_quality) >= 2 ? 8 : 4;
  uint32_t size = surface_pitch * factor;
  if (shadow_.factor) {
    // shortcut: one caster surface per game; a different one (or a changed
    // quality) keeps the first set of resources until restart.
    return shadow_.factor == factor && shadow_.size >= size &&
           shadow_.color_format == color_format && shadow_.depth_format == depth_format;
  }
  const ui::d3d12::D3D12Provider& provider = command_processor_.GetD3D12Provider();
  ID3D12Device* device = provider.GetDevice();
  auto create = [&](ShadowTexture& texture, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                    D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = size;
    desc.Height = size;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    texture.state = state;
    return SUCCEEDED(device->CreateCommittedResource(&ui::d3d12::util::kHeapPropertiesDefault,
                                                     provider.GetHeapFlagCreateNotZeroed(), &desc,
                                                     state, nullptr,
                                                     IID_PPV_ARGS(&texture.resource)));
  };
  D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
  heap_desc.NumDescriptors = 1;
  heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  bool ok = create(shadow_.color, color_format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                   D3D12_RESOURCE_STATE_RENDER_TARGET) &&
            create(shadow_.depth, GetDepthResourceDXGIFormat(depth_format),
                   D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE) &&
            create(shadow_.blur_temp, color_format, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS) &&
            SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&shadow_.rtv_heap)));
  heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
  ok = ok && SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&shadow_.dsv_heap)));
  if (ok) {
    shadow_.blur_pipeline = ui::d3d12::util::CreateComputePipeline(
        device, shaders::shadow_blur_cs, sizeof(shaders::shadow_blur_cs),
        resolve_copy_root_signature_);
    ok = shadow_.blur_pipeline != nullptr;
  }
  if (!ok) {
    REXGPU_ERROR("D3D12RenderTargetCache: Failed to create the high-resolution shadow resources");
    ShadowShutdown();
    shadow_.factor = UINT32_MAX;  // don't retry
    return false;
  }
  D3D12_RENDER_TARGET_VIEW_DESC rtv_desc = {};
  rtv_desc.Format = color_format;
  rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
  device->CreateRenderTargetView(shadow_.color.resource, &rtv_desc,
                                 shadow_.rtv_heap->GetCPUDescriptorHandleForHeapStart());
  D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
  dsv_desc.Format = GetDepthDSVDXGIFormat(depth_format);
  dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
  device->CreateDepthStencilView(shadow_.depth.resource, &dsv_desc,
                                 shadow_.dsv_heap->GetCPUDescriptorHandleForHeapStart());
  shadow_.factor = factor;
  shadow_.size = size;
  shadow_.color_format = color_format;
  shadow_.depth_format = depth_format;
  REXGPU_INFO("High-resolution shadows: {}x caster target, {}x{}", factor, size, size);
  return true;
}

uint32_t D3D12RenderTargetCache::ShadowMirrorDraw(uint32_t bound_render_targets,
                                                  const uint32_t* bound_formats,
                                                  const draw_util::Scissor& scissor_scaled,
                                                  const uint32_t* sampled_bases,
                                                  uint32_t sampled_count) {
  if (!IsShadowEnabled() || shadow_.factor == UINT32_MAX) {
    return 0;
  }
  const RegisterFile& regs = register_file();
  if (sampled_count) {
    // Drawing into the caster surface from the resolved casters is the game's blur.
    if (shadow_.holds_base && regs[XE_GPU_REG_RB_SURFACE_INFO] == shadow_.pass_surface_info &&
        regs[XE_GPU_REG_RB_COLOR_INFO] == shadow_.pass_color_info &&
        regs[XE_GPU_REG_RB_DEPTH_INFO] == shadow_.pass_depth_info) {
      for (uint32_t i = 0; i < sampled_count; ++i) {
        if (sampled_bases[i] == shadow_.holds_base) {
          shadow_.pass_blurs = true;
        }
      }
    }
    return 0;
  }
  auto surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  // Depth and color 0 only, so the pipeline matches the private target.
  if (surface_info.msaa_samples != xenos::MsaaSamples::k1X ||
      surface_info.surface_pitch > kShadowMaxSurfacePitch || bound_render_targets != 0b11 ||
      xenos::ColorRenderTargetFormat(bound_formats[1]) !=
          xenos::ColorRenderTargetFormat::k_8_8_8_8) {
    return 0;
  }
  auto depth_format = xenos::DepthRenderTargetFormat(bound_formats[0]);
  DXGI_FORMAT color_format = GetColorDrawDXGIFormat(xenos::ColorRenderTargetFormat::k_8_8_8_8);
  if (!ShadowEnsureResources(surface_info.surface_pitch, color_format, depth_format)) {
    return 0;
  }
  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  D3D12_CPU_DESCRIPTOR_HANDLE rtv = shadow_.rtv_heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_CPU_DESCRIPTOR_HANDLE dsv = shadow_.dsv_heap->GetCPUDescriptorHandleForHeapStart();
  ShadowTransition(shadow_.color, D3D12_RESOURCE_STATE_RENDER_TARGET);
  ShadowTransition(shadow_.depth, D3D12_RESOURCE_STATE_DEPTH_WRITE);
  command_processor_.SubmitBarriers();
  uint32_t color_info = regs[XE_GPU_REG_RB_COLOR_INFO];
  uint32_t depth_info = regs[XE_GPU_REG_RB_DEPTH_INFO];
  if (!shadow_.pass_draws || shadow_.pass_surface_info != surface_info.value ||
      shadow_.pass_color_info != color_info || shadow_.pass_depth_info != depth_info) {
    // A new caster pass: start from the surface's clear values, as the guest
    // target was cleared by its last resolve.
    shadow_.pass_surface_info = surface_info.value;
    shadow_.pass_color_info = color_info;
    shadow_.pass_depth_info = depth_info;
    shadow_.pass_draws = 0;
    shadow_.holds_base = 0;
    shadow_.pass_blurs = false;
    shadow_.holds_blurred = false;
    uint32_t clear_color = shadow_.clear_known ? shadow_.clear_color
                                               : regs[XE_GPU_REG_RB_COLOR_CLEAR];
    uint32_t clear_depth = shadow_.clear_known ? shadow_.clear_depth
                                               : regs[XE_GPU_REG_RB_DEPTH_CLEAR];
    float color[4];
    for (uint32_t i = 0; i < 4; ++i) {
      color[i] = ((clear_color >> (i * 8)) & 0xFF) * (1.0f / 0xFF);
    }
    uint32_t depth_guest = (clear_depth >> 8) & 0xFFFFFF;
    float depth = depth_format == xenos::DepthRenderTargetFormat::kD24FS8
                      ? xenos::Float20e4To32(depth_guest) * 0.5f
                      : xenos::UNorm24To32(depth_guest);
    command_list.D3DClearRenderTargetView(rtv, color, 0, nullptr);
    command_list.D3DClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                                          depth, UINT8(clear_depth & 0xFF), 0, nullptr);
  }
  ++shadow_.pass_draws;
  command_list.D3DOMSetRenderTargets(1, &rtv, FALSE, &dsv);
  // The next guest draw binds the guest render targets again.
  are_current_command_list_render_targets_valid_ = false;
  // Scissor values are whole guest pixels times the draw resolution scale.
  D3D12_RECT scissor;
  scissor.left = LONG(std::min(
      scissor_scaled.offset[0] / draw_resolution_scale_x() * shadow_.factor, shadow_.size));
  scissor.top = LONG(std::min(
      scissor_scaled.offset[1] / draw_resolution_scale_y() * shadow_.factor, shadow_.size));
  scissor.right = LONG(std::min((scissor_scaled.offset[0] + scissor_scaled.extent[0]) /
                                    draw_resolution_scale_x() * shadow_.factor,
                                shadow_.size));
  scissor.bottom = LONG(std::min((scissor_scaled.offset[1] + scissor_scaled.extent[1]) /
                                     draw_resolution_scale_y() * shadow_.factor,
                                 shadow_.size));
  command_processor_.SetScissorRect(scissor);
  return shadow_.factor;
}

void D3D12RenderTargetCache::ShadowNoteResolve(const draw_util::ResolveInfo& resolve_info,
                                               bool copied, D3D12TextureCache& texture_cache) {
  if (!IsShadowEnabled() || !shadow_.factor || shadow_.factor == UINT32_MAX) {
    return;
  }
  const RegisterFile& regs = register_file();
  bool caster_surface = regs[XE_GPU_REG_RB_SURFACE_INFO] == shadow_.pass_surface_info &&
                        regs[XE_GPU_REG_RB_COLOR_INFO] == shadow_.pass_color_info &&
                        regs[XE_GPU_REG_RB_DEPTH_INFO] == shadow_.pass_depth_info;
  if (copied && !resolve_info.IsCopyingDepth()) {
    // DOA4: casters -> resolve A, two blur passes each resolved back to A, then
    // a copy of the surface to B, which the stage samples. A blur resolved
    // straight to a new texture is taken as the mask too.
    uint32_t dest = resolve_info.copy_dest_base;
    bool mask = false;
    if (caster_surface && shadow_.pass_draws) {
      shadow_.holds_base = dest;
      shadow_.holds_blurred = false;
    } else if (caster_surface && shadow_.pass_blurs) {
      if (dest == shadow_.holds_base) {
        shadow_.holds_blurred = true;
      } else {
        mask = true;
      }
    } else if (caster_surface && shadow_.holds_blurred && dest != shadow_.holds_base) {
      mask = true;
    }
    if (mask) {
      ShadowBlur(resolve_info, texture_cache);
    } else {
      // Anything else the guest resolves there replaces a previous mask.
      // shortcut: guest CPU writes to a mask texture keep the stand-in; DOA4
      // only resolves there - watch the pages if a game writes them directly.
      texture_cache.ClearHiResOverride(dest);
    }
  }
  if (caster_surface) {
    shadow_.pass_blurs = false;
    shadow_.pass_draws = 0;
    if (resolve_info.IsClearingColor()) {
      shadow_.clear_color = resolve_info.rb_color_clear;
      shadow_.clear_known = true;
    }
    if (resolve_info.IsClearingDepth()) {
      shadow_.clear_depth = resolve_info.rb_depth_clear;
    }
  }
}

void D3D12RenderTargetCache::ShadowBlur(const draw_util::ResolveInfo& resolve_info,
                                        D3D12TextureCache& texture_cache) {
  const RegisterFile& regs = register_file();
  auto dest_pitch = regs.Get<reg::RB_COPY_DEST_PITCH>();
  uint32_t guest_width = dest_pitch.copy_dest_pitch, guest_height = dest_pitch.copy_dest_height;
  uint32_t width = guest_width * shadow_.factor, height = guest_height * shadow_.factor;
  if (!width || !height || width > shadow_.size || height > shadow_.size) {
    return;
  }
  ID3D12Device* device = command_processor_.GetD3D12Provider().GetDevice();
  uint32_t base = resolve_info.copy_dest_base;
  ShadowTexture* result = nullptr;
  for (auto& entry : shadow_.results) {
    if (entry.first == base) {
      result = &entry.second;
    }
  }
  if (!result) {
    // shortcut: a handful of shadow masks per game, allocated once; cap them so
    // a game resolving many differently-addressed ones doesn't grow unbounded.
    if (shadow_.results.size() >= 8) {
      return;
    }
    ShadowTexture texture;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = shadow_.color_format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    texture.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesDefault,
            command_processor_.GetD3D12Provider().GetHeapFlagCreateNotZeroed(), &desc,
            texture.state, nullptr, IID_PPV_ARGS(&texture.resource)))) {
      REXGPU_ERROR("D3D12RenderTargetCache: Failed to create a high-resolution shadow mask");
      return;
    }
    shadow_.results.emplace_back(base, texture);
    result = &shadow_.results.back().second;
  }

  ui::d3d12::util::DescriptorCpuGpuHandlePair descriptors[4];
  if (!command_processor_.RequestOneUseSingleViewDescriptors(4, descriptors)) {
    return;
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
  srv_desc.Format = shadow_.color_format;
  srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv_desc.Texture2D.MipLevels = 1;
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
  uav_desc.Format = shadow_.color_format;
  uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateShaderResourceView(shadow_.color.resource, &srv_desc, descriptors[0].first);
  device->CreateUnorderedAccessView(shadow_.blur_temp.resource, nullptr, &uav_desc,
                                    descriptors[1].first);
  device->CreateShaderResourceView(shadow_.blur_temp.resource, &srv_desc, descriptors[2].first);
  device->CreateUnorderedAccessView(result->resource, nullptr, &uav_desc, descriptors[3].first);

  float sigma = kShadowGuestBlurSigma * float(REXCVAR_GET(shadow_softness)) * shadow_.factor;
  uint32_t constants[4];
  constants[0] = width | (height << 16);
  std::memcpy(&constants[2], &sigma, sizeof(float));
  constants[3] = 0;
  static_assert(sizeof(draw_util::ResolveCopyShaderConstants::DestRelative) >= sizeof(constants),
                "The shadow blur constants must fit in the resolve copy root constants");
  DeferredCommandList& command_list = command_processor_.GetDeferredCommandList();
  UINT groups_x = (width + 7) / 8, groups_y = (height + 7) / 8;
  command_list.D3DSetComputeRootSignature(resolve_copy_root_signature_);
  command_processor_.SetExternalPipeline(shadow_.blur_pipeline);

  // Across, from the caster copy into the temporary texture.
  ShadowTransition(shadow_.color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  ShadowTransition(shadow_.blur_temp, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  constants[1] = 0;
  command_list.D3DSetComputeRoot32BitConstants(0, 4, constants, 0);
  command_list.D3DSetComputeRootDescriptorTable(1, descriptors[1].second);
  command_list.D3DSetComputeRootDescriptorTable(2, descriptors[0].second);
  command_processor_.SubmitBarriers();
  command_list.D3DDispatch(groups_x, groups_y, 1);

  // Down, into the mask, in the guest texture's component order.
  ShadowTransition(shadow_.blur_temp, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  ShadowTransition(*result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  constants[1] = 1 | (uint32_t(resolve_info.copy_dest_info.copy_dest_swap) << 1);
  command_list.D3DSetComputeRoot32BitConstants(0, 4, constants, 0);
  command_list.D3DSetComputeRootDescriptorTable(1, descriptors[3].second);
  command_list.D3DSetComputeRootDescriptorTable(2, descriptors[2].second);
  command_processor_.SubmitBarriers();
  command_list.D3DDispatch(groups_x, groups_y, 1);
  ShadowTransition(*result, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

  texture_cache.SetHiResOverride(base, guest_width, guest_height, result->resource,
                                 shadow_.color_format);
}

void D3D12RenderTargetCache::ShadowShutdown() {
  for (auto& entry : shadow_.results) {
    ui::d3d12::util::ReleaseAndNull(entry.second.resource);
  }
  ui::d3d12::util::ReleaseAndNull(shadow_.blur_pipeline);
  ui::d3d12::util::ReleaseAndNull(shadow_.dsv_heap);
  ui::d3d12::util::ReleaseAndNull(shadow_.rtv_heap);
  ui::d3d12::util::ReleaseAndNull(shadow_.blur_temp.resource);
  ui::d3d12::util::ReleaseAndNull(shadow_.depth.resource);
  ui::d3d12::util::ReleaseAndNull(shadow_.color.resource);
  shadow_ = {};
}

}  // namespace rex::graphics::d3d12
