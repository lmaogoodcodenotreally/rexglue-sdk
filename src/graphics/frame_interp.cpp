/**
 ******************************************************************************
 * ReXGlue : Xbox 360 runtime                                               *
 ******************************************************************************
 */

#include <rex/graphics/frame_interp.h>

#include <algorithm>
#include <bit>
#include <cmath>

#include <rex/graphics/xenos.h>

namespace rex::graphics::frame_interp {

namespace {

constexpr uint32_t kAluFirst = 0x4000;  // SHADER_CONSTANT_000_X
constexpr uint32_t kAluEnd = 0x4800;    // first fetch constant register
constexpr uint32_t kFetchEnd = kAluEnd + 32 * 6;
// How far ahead in the previous frame a draw may be searched for when draws
// were added or culled in between.
constexpr size_t kResyncWindow = 48;
// Relative change of one constant between frames beyond which it is no motion.
constexpr float kJump = 0.5f;

uint32_t Load(const uint32_t* be, size_t i) { return std::byteswap(be[i]); }

uint64_t Mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  return h;
}

}  // namespace

void Parse(const uint32_t* be, size_t n, ParsedFrame& out) {
  out.segments.clear();
  out.alu_offset.clear();
  out.alu_count.clear();
  uint64_t vs = 0, ps = 0, layout = 0, textures = 0;
  uint32_t alu_begin = 0;
  auto close = [&](uint64_t signature, size_t packet) {
    out.segments.push_back({Mix(signature, textures), layout, alu_begin,
                            uint32_t(out.alu_offset.size()), uint32_t(packet)});
    layout = 0;
    textures = 0;
    alu_begin = uint32_t(out.alu_offset.size());
  };
  size_t i = 0;
  while (i < n) {
    const uint32_t h = Load(be, i);
    if (!h) {
      ++i;
      continue;
    }
    const uint32_t count = ((h >> 16) & 0x3FFF) + 1;
    switch (h >> 30) {
      case 0: {
        const uint32_t base = h & 0x7FFF;
        const bool one_reg = (h >> 15) & 1;
        const uint32_t last = one_reg ? base + 1 : base + count;
        if (base < kAluEnd && last > kAluFirst) {
          const uint32_t lo = std::max(base, kAluFirst), hi = std::min(last, kAluEnd);
          // A single-register packet rewrites one constant `count` times; the
          // last write is the one a draw sees, but lerping all is harmless.
          out.alu_offset.push_back(uint32_t(i + 1 + (one_reg ? 0 : lo - base)));
          out.alu_count.push_back(one_reg ? count : hi - lo);
          layout = Mix(layout, h);
        }
        // Draws sharing shaders (UI icons, say) are told apart by the texture
        // base addresses they set: dword 1 of each 6-dword fetch slot (for
        // vertex fetches that dword is the buffer size, also stable).
        for (uint32_t k = 0; k < (one_reg ? 1 : count) && i + 1 + k < n; ++k) {
          const uint32_t reg = base + k;
          if (reg >= kAluEnd && reg < kFetchEnd && (reg - kAluEnd) % 6 == 1) {
            textures = Mix(Mix(textures, reg), Load(be, i + 1 + k) & 0xFFFFF000u);
          }
        }
        i += 1 + count;
        break;
      }
      case 1:
        i += 3;
        break;
      case 2:
        i += 1;
        break;
      default: {
        if (i + count >= n) {  // truncated packet: stop, keep what parsed
          i = n;
          break;
        }
        switch ((h >> 8) & 0x7F) {
          case xenos::PM4_IM_LOAD: {
            const uint32_t addr_type = Load(be, i + 1);
            ((addr_type & 3) ? ps : vs) = Mix(1, addr_type & ~3u);
            break;
          }
          case xenos::PM4_IM_LOAD_IMMEDIATE: {
            uint64_t s = Mix(2, Load(be, i + 2));
            for (uint32_t k = 3; k < std::min<uint32_t>(count + 1, 7); ++k) {
              s = Mix(s, Load(be, i + k));
            }
            (Load(be, i + 1) ? ps : vs) = s;
            break;
          }
          case xenos::PM4_DRAW_INDX:  // viz query, initiator, [index base, size]
            close(Mix(Mix(Mix(vs, ps), Load(be, i + 2)), count >= 4 ? Load(be, i + 3) : 0), i);
            break;
          case xenos::PM4_DRAW_INDX_2:  // initiator, inline indices
            close(Mix(Mix(vs, ps), Load(be, i + 1)), i);
            break;
        }
        i += 1 + count;
        break;
      }
    }
  }
  close(0x7A11, n);  // packets after the last draw (the swap resolve setup)
}

Stats Interpolate(const uint32_t* prev_be, const ParsedFrame& prev, uint32_t* cur_be,
                  const ParsedFrame& cur, float t,
                  std::vector<std::pair<uint32_t, uint32_t>>* lerped_pairs) {
  Stats stats;
  if (lerped_pairs) {
    lerped_pairs->clear();
  }
  stats.segments = uint32_t(cur.segments.size());
  size_t j = 0;
  for (const auto& seg : cur.segments) {
    const size_t end = std::min(prev.segments.size(), j + kResyncWindow);
    size_t found = end;
    for (size_t q = j; q < end; ++q) {
      if (prev.segments[q].signature == seg.signature) {
        found = q;
        break;
      }
    }
    if (found == end) continue;  // new draw this frame: drawn as is
    const auto& pseg = prev.segments[found];
    j = found + 1;
    ++stats.matched;
    if (pseg.layout != seg.layout || pseg.alu_end - pseg.alu_begin != seg.alu_end - seg.alu_begin) {
      continue;
    }
    // lerp == false: the jump check pass; lerp == true: the write pass.
    auto visit = [&](bool lerp) {
      for (uint32_t e = 0; e < seg.alu_end - seg.alu_begin; ++e) {
        const uint32_t co = cur.alu_offset[seg.alu_begin + e];
        const uint32_t po = prev.alu_offset[pseg.alu_begin + e];
        const uint32_t dwords = std::min(cur.alu_count[seg.alu_begin + e],
                                         prev.alu_count[pseg.alu_begin + e]);
        for (uint32_t d = 0; d < dwords; ++d) {
          const uint32_t cb = Load(cur_be, co + d), pb = Load(prev_be, po + d);
          if (cb == pb) continue;
          const float c = std::bit_cast<float>(cb), p = std::bit_cast<float>(pb);
          if (!std::isfinite(c) || !std::isfinite(p)) continue;
          if (!lerp) {
            if (std::abs(c - p) > kJump * std::max({1.0f, std::abs(c), std::abs(p)})) {
              return false;
            }
            continue;
          }
          cur_be[co + d] = std::byteswap(std::bit_cast<uint32_t>(p + (c - p) * t));
        }
      }
      return true;
    };
    if (!visit(false)) {
      ++stats.jumped;
      continue;
    }
    ++stats.lerped;
    visit(true);
    if (lerped_pairs) {
      lerped_pairs->emplace_back(seg.packet, pseg.packet);
    }
  }
  return stats;
}

bool LerpVertexFloats(const uint32_t* prev, uint32_t* cur, uint32_t vertices,
                      uint32_t stride_words, uint64_t float_words, uint32_t endian, float t) {
  const auto e = xenos::Endian(endian);
  for (uint32_t i = 0; i < vertices * stride_words; ++i) {
    if (prev[i] == cur[i]) continue;
    if (!((float_words >> (i % stride_words)) & 1)) {
      return false;
    }
    const float c = std::bit_cast<float>(xenos::GpuSwap(cur[i], e));
    const float p = std::bit_cast<float>(xenos::GpuSwap(prev[i], e));
    // Unlike constants, vertices carry unit normals, whose components turn by
    // up to 2 in one frame of fast rotation: the floor is 2, not 1.
    if (std::isfinite(c) && std::isfinite(p) &&
        std::abs(c - p) > std::max(2.0f, kJump * std::max(std::abs(c), std::abs(p)))) {
      return false;
    }
  }
  for (uint32_t v = 0; v < vertices; ++v, prev += stride_words, cur += stride_words) {
    for (uint64_t m = float_words; m; m &= m - 1) {
      const uint32_t w = uint32_t(std::countr_zero(m));
      if (prev[w] == cur[w]) continue;
      const float c = std::bit_cast<float>(xenos::GpuSwap(cur[w], e));
      const float p = std::bit_cast<float>(xenos::GpuSwap(prev[w], e));
      if (!std::isfinite(c) || !std::isfinite(p)) continue;
      cur[w] = xenos::GpuSwap(std::bit_cast<uint32_t>(p + (c - p) * t), e);
    }
  }
  return true;
}

}  // namespace rex::graphics::frame_interp
