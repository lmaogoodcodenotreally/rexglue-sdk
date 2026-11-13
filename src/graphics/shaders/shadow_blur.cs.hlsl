// Separable Gaussian blur of a high-resolution projected shadow mask
// (shadow_quality): one direction per dispatch, source texture to destination
// texture, clamped at the edges. The second pass can swap red and blue to
// match the guest resolve's component order.
//
// fxc /T cs_5_1 /E shadow_blur_cs /Vn shadow_blur_cs /Fh bytecode/d3d12_5_1/shadow_blur_cs.h shadow_blur.cs.hlsl

cbuffer xe_shadow_blur_constants : register(b0) {
  uint xe_shadow_blur_size;    // width | height << 16
  uint xe_shadow_blur_info;    // vertical << 0 | swap red and blue << 1
  float xe_shadow_blur_sigma;  // in texels
  uint xe_shadow_blur_padding;
};

Texture2D<float4> xe_shadow_blur_source : register(t0);
RWTexture2D<unorm float4> xe_shadow_blur_dest : register(u0);

[numthreads(8, 8, 1)]
void shadow_blur_cs(uint3 id : SV_DispatchThreadID) {
  uint2 size = uint2(xe_shadow_blur_size & 0xFFFFu, xe_shadow_blur_size >> 16u);
  if (any(id.xy >= size)) {
    return;
  }
  int2 step = (xe_shadow_blur_info & 1u) ? int2(0, 1) : int2(1, 0);
  float sigma = max(xe_shadow_blur_sigma, 0.01f);
  int radius = min(int(ceil(sigma * 3.0f)), 48);
  float k = -0.5f / (sigma * sigma);
  float4 sum = 0.0f;
  float weight = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    int2 p = clamp(int2(id.xy) + step * i, int2(0, 0), int2(size) - 1);
    float w = exp(float(i * i) * k);
    sum += xe_shadow_blur_source.Load(int3(p, 0)) * w;
    weight += w;
  }
  float4 result = sum / weight;
  if (xe_shadow_blur_info & 2u) {
    result = result.bgra;
  }
  xe_shadow_blur_dest[id.xy] = result;
}
