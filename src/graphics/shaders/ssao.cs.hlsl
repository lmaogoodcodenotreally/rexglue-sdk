// Screen-space ambient occlusion on a resolved, resolution-scaled scene.
// Both entry points read/write the scaled resolve buffer directly: a 2D tiled
// 32bpp guest texture where each 16-byte guest chunk (4 texels) is expanded to
// scale_x * scale_y consecutive 16-byte chunks (layout taken from
// texture_load_32bpb_scaled_cs).
//
// fxc /T cs_5_1 /E ssao_ao_cs /Vn ssao_ao_cs /Fh bytecode/d3d12_5_1/ssao_ao_cs.h ssao.cs.hlsl
// fxc /T cs_5_1 /D XE_SSAO_APPLY /E ssao_apply_cs /Vn ssao_apply_cs /Fh bytecode/d3d12_5_1/ssao_apply_cs.h ssao.cs.hlsl

cbuffer xe_ssao_constants : register(b0) {
  uint xe_ssao_size;      // host width | host height << 16
  uint xe_ssao_info;      // guest pitch / 32 | scale_x << 8 | scale_y << 12 |
                          // endian << 16 | depth float24 << 20 |
                          // reversed depth << 21 | debug view << 22
  float xe_ssao_radius;   // world radius in units of the near plane
  float xe_ssao_strength;
};

uint XeSsaoSwap(uint v, uint endian) {
  if (endian == 1u || endian == 2u) {
    v = ((v & 0x00FF00FFu) << 8u) | ((v & 0xFF00FF00u) >> 8u);
  }
  if (endian == 2u || endian == 3u) {
    v = (v << 16u) | (v >> 16u);
  }
  return v;
}

// Byte offset of a 32bpp texel in a 2D tiled Xenos texture.
uint XeSsaoTiledOffset(uint x, uint y, uint pitch) {
  uint micro_outer = (y & 6u) << 4u;
  uint outer = (((y >> 5u) * (pitch >> 5u)) << 9u) + ((micro_outer & ~15u) << 1u) +
               (micro_outer & 15u) + ((y & 8u) << 5u) + ((y & 1u) << 4u);
  uint micro_inner = (x & 7u) << 2u;
  uint o = outer + ((x >> 5u) << 9u) + ((micro_inner & ~15u) << 1u) + (micro_inner & 15u);
  return ((o & ~511u) << 3u) + ((o & 448u) << 2u) + (o & 63u) + ((y & 16u) << 7u) +
         (((((y & 8u) >> 2u) + (x >> 3u)) & 3u) << 6u);
}

uint XeSsaoIndex(uint2 p) {
  uint pitch = (xe_ssao_info & 0xFFu) << 5u;
  uint sx = (xe_ssao_info >> 8u) & 0xFu;
  uint sy = (xe_ssao_info >> 12u) & 0xFu;
  uint g4 = p.x >> 2u;
  uint gx4 = g4 / sx;
  uint gy = p.y / sy;
  uint bytes = XeSsaoTiledOffset(gx4 << 2u, gy, pitch) * (sx * sy) +
               ((g4 - gx4 * sx) * sy + (p.y - gy * sy)) * 16u + (p.x & 3u) * 4u;
  return bytes >> 2u;
}

uint2 XeSsaoSize() { return uint2(xe_ssao_size & 0xFFFFu, xe_ssao_size >> 16u); }

#ifdef XE_SSAO_APPLY

RWBuffer<uint> xe_ssao_color : register(u0);
Texture2D<float2> xe_ssao_ao : register(t0);

[numthreads(8, 8, 1)]
void ssao_apply_cs(uint3 id : SV_DispatchThreadID) {
  uint2 size = XeSsaoSize();
  if (any(id.xy >= size)) {
    return;
  }
  float2 center = xe_ssao_ao.Load(int3(id.xy, 0));
  bool debug = (xe_ssao_info & (1u << 22u)) != 0u;
  if (center.y >= 23.0f && !debug) {
    return;  // Background.
  }
  // 4x4 depth-aware box, matching the 4x4 period of the sample rotation.
  float sum = 0.0f, weight = 0.0f;
  [unroll] for (int y = -2; y < 2; ++y) {
    [unroll] for (int x = -2; x < 2; ++x) {
      int2 s = clamp(int2(id.xy) + int2(x, y), int2(0, 0), int2(size) - 1);
      float2 t = xe_ssao_ao.Load(int3(s, 0));
      float w = saturate(1.0f - abs(t.y - center.y) * 16.0f) + 1.0e-4f;
      sum += t.x * w;
      weight += w;
    }
  }
  float ao = sum / weight;
  uint endian = (xe_ssao_info >> 16u) & 3u;
  uint index = XeSsaoIndex(id.xy);
  uint c = XeSsaoSwap(xe_ssao_color[index], endian);
  float3 rgb = float3(c & 0xFFu, (c >> 8u) & 0xFFu, (c >> 16u) & 0xFFu);
  if (debug) {
    rgb = center.y >= 23.0f ? float3(255.0f, 0.0f, 255.0f) : 255.0f * ao;
  } else {
    rgb *= lerp(1.0f, ao, xe_ssao_strength);
  }
  uint3 q = uint3(rgb + 0.5f);
  c = (c & 0xFF000000u) | q.r | (q.g << 8u) | (q.b << 16u);
  xe_ssao_color[index] = XeSsaoSwap(c, endian);
}

#else

Buffer<uint> xe_ssao_depth : register(t0);
RWTexture2D<float2> xe_ssao_out : register(u0);

// Linear depth in units of the near plane (assumes far >> near).
float XeSsaoLinearDepth(uint2 p) {
  uint d24 = XeSsaoSwap(xe_ssao_depth[XeSsaoIndex(p)], (xe_ssao_info >> 16u) & 3u) >> 8u;
  float d;
  if (xe_ssao_info & (1u << 20u)) {
    uint e = d24 >> 20u;
    d = e ? asfloat(((e + 112u) << 23u) | ((d24 & 0xFFFFFu) << 3u)) : 0.0f;
  } else {
    d = float(d24) * (1.0f / 16777215.0f);
  }
  if (xe_ssao_info & (1u << 21u)) {
    d = 1.0f - d;
  }
  return 1.0f / max(1.0f - d, 1.0f / 16777216.0f);
}

static const float kTanHalfFovY = 0.45f;

float3 XeSsaoPosition(float2 p, float z, float2 inv_size, float2 tan_half) {
  return float3(((p + 0.5f) * inv_size * 2.0f - 1.0f) * tan_half * z, z);
}

[numthreads(8, 8, 1)]
void ssao_ao_cs(uint3 id : SV_DispatchThreadID) {
  uint2 size = XeSsaoSize();
  if (any(id.xy >= size)) {
    return;
  }
  float z = XeSsaoLinearDepth(id.xy);
  if (z >= 1.0e6f) {
    xe_ssao_out[id.xy] = float2(1.0f, 24.0f);
    return;
  }
  float2 inv_size = 1.0f / float2(size);
  float sx = float((xe_ssao_info >> 8u) & 0xFu), sy = float((xe_ssao_info >> 12u) & 0xFu);
  // Guest aspect ratio, so non-square resolution scales don't skew the normals.
  float2 tan_half = float2(kTanHalfFovY * (float(size.x) / sx) / (float(size.y) / sy),
                           kTanHalfFovY);
  float3 pc = XeSsaoPosition(float2(id.xy), z, inv_size, tan_half);
  // Normal from the flatter neighbour on each axis, to stay stable at edges.
  int2 last = int2(size) - 1;
  uint2 l = uint2(max(int(id.x) - 1, 0), id.y), r = uint2(min(int(id.x) + 1, last.x), id.y);
  uint2 u = uint2(id.x, max(int(id.y) - 1, 0)), d = uint2(id.x, min(int(id.y) + 1, last.y));
  float3 pl = XeSsaoPosition(float2(l), XeSsaoLinearDepth(l), inv_size, tan_half);
  float3 pr = XeSsaoPosition(float2(r), XeSsaoLinearDepth(r), inv_size, tan_half);
  float3 pu = XeSsaoPosition(float2(u), XeSsaoLinearDepth(u), inv_size, tan_half);
  float3 pd = XeSsaoPosition(float2(d), XeSsaoLinearDepth(d), inv_size, tan_half);
  float3 dx = abs(pr.z - pc.z) < abs(pc.z - pl.z) ? pr - pc : pc - pl;
  float3 dy = abs(pd.z - pc.z) < abs(pc.z - pu.z) ? pd - pc : pc - pu;
  float3 n = normalize(cross(dy, dx));
  if (n.z > 0.0f) {
    n = -n;
  }

  float radius = xe_ssao_radius;
  float radius_px = radius / (z * kTanHalfFovY) * (0.5f * float(size.y));
  if (radius_px < 1.0f) {
    xe_ssao_out[id.xy] = float2(1.0f, log2(z));
    return;
  }
  radius_px = min(radius_px, 0.25f * float(size.y));
  float r2 = radius * radius;
  // Interleaved 4x4 rotation, removed again by the 4x4 blur in the apply pass.
  float rotation = float(((id.y & 3u) << 2u) | (id.x & 3u)) * (6.2831853f / 16.0f);
  const uint kSamples = 8u;
  float sum = 0.0f;
  [unroll] for (uint i = 0u; i < kSamples; ++i) {
    float alpha = (float(i) + 0.5f) / float(kSamples);
    float angle = alpha * (6.2831853f * 3.0f) + rotation;
    float2 offset = float2(cos(angle), sin(angle)) * (alpha * radius_px);
    uint2 s = uint2(clamp(int2(float2(id.xy) + offset + 0.5f), int2(0, 0), last));
    float3 v = XeSsaoPosition(float2(s), XeSsaoLinearDepth(s), inv_size, tan_half) - pc;
    float vv = dot(v, v);
    float f = max(r2 - vv, 0.0f);
    sum += f * f * f * max((dot(v, n) - 0.01f * z) / (vv + 0.01f * r2), 0.0f);
  }
  float ao = saturate(1.0f - sum * (5.0f / float(kSamples)) / (r2 * r2 * r2));
  xe_ssao_out[id.xy] = float2(ao, log2(z));
}

#endif
