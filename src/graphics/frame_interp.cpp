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
// Relative change of one constant between frames beyond which it is no motion.
constexpr float kJump = 0.5f;
// Absolute floor of that change: unit rotation matrix elements and normals
// move by up to 2 in one tick of fast motion (a strike, a camera swing).
constexpr float kJumpFloor = 2.0f;
// How many previous-frame draws back a constant a draw saw may be looked for.
constexpr uint32_t kLookBack = 16;

bool Jumped(float c, float p) {
  return std::abs(c - p) > std::max(kJumpFloor, kJump * std::max(std::abs(c), std::abs(p)));
}

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
  out.alu_reg.clear();
  uint64_t vs = 0, ps = 0, layout = 0, textures = 0;
  uint32_t alu_begin = 0;
  auto close = [&](uint64_t draw, size_t packet) {
    out.segments.push_back({Mix(draw, textures), draw, layout, alu_begin,
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
          // last write is the one a draw sees.
          out.alu_offset.push_back(uint32_t(one_reg ? i + count : i + 1 + (lo - base)));
          out.alu_count.push_back(one_reg ? 1 : hi - lo);
          out.alu_reg.push_back(lo);
          layout = Mix(layout, h);
        }
        // Draws sharing shaders (UI icons, say) are told apart by the texture
        // base addresses they set: dword 1 of each 6-dword fetch slot (for
        // vertex fetches that dword is the buffer size, also stable).
        const uint32_t fetch_last = std::min(last, kFetchEnd);
        for (uint32_t reg = std::max(base, kAluEnd); reg < fetch_last && i + 1 + (reg - base) < n;
             ++reg) {
          if ((reg - kAluEnd) % 6 == 1) {
            textures = Mix(Mix(textures, reg), Load(be, i + 1 + (reg - base)) & 0xFFFFF000u);
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
  // Draws pair by signature occurrence: the k-th draw of a signature this
  // frame with the k-th of the previous frame. Meshes draw once per pass with
  // the same signature in each, and draws in between are added, culled and
  // re-sorted, so a search ahead pairs across passes and loses what it skips.
  // Where a signature's count changed, the nearest unused one in stream order.
  constexpr uint32_t kNone = UINT32_MAX;
  std::vector<uint32_t> match(cur.segments.size(), kNone);
  std::vector<bool> used(prev.segments.size());
  auto pair_by = [&](uint64_t ParsedFrame::Segment::*key) {
    auto order = [&](const ParsedFrame& frame, auto skip) {
      std::vector<std::pair<uint64_t, uint32_t>> v;
      for (uint32_t i = 0; i < frame.segments.size(); ++i) {
        if (!skip(i)) v.emplace_back(frame.segments[i].*key, i);
      }
      std::sort(v.begin(), v.end());
      return v;
    };
    const auto prev_order = order(prev, [&](uint32_t i) { return bool(used[i]); });
    const auto cur_order = order(cur, [&](uint32_t i) { return match[i] != kNone; });
    for (size_t a = 0, b; a < cur_order.size(); a = b) {
      const uint64_t k = cur_order[a].first;
      for (b = a + 1; b < cur_order.size() && cur_order[b].first == k; ++b) {
      }
      auto [pa, pb] =
          std::equal_range(prev_order.begin(), prev_order.end(), std::make_pair(k, 0u),
                           [](const auto& x, const auto& y) { return x.first < y.first; });
      if (size_t(pb - pa) == b - a) {
        for (size_t c = a; c < b; ++c, ++pa) {
          match[cur_order[c].second] = pa->second;
          used[pa->second] = true;
        }
        continue;
      }
      for (size_t c = a; c < b; ++c) {  // a few draws: the quadratic search is fine
        const int64_t at = cur_order[c].second;
        auto best = pb;
        for (auto q = pa; q != pb; ++q) {
          if (!used[q->second] &&
              (best == pb || std::abs(q->second - at) < std::abs(best->second - at))) {
            best = q;
          }
        }
        if (best != pb) {
          match[at] = best->second;
          used[best->second] = true;
        }
      }
    }
  };
  pair_by(&ParsedFrame::Segment::signature);
  // Then by the draw alone: which textures a draw sets itself depends on what
  // the draw before it left bound, which changes when draws are re-sorted.
  pair_by(&ParsedFrame::Segment::draw);

  // The game uploads only the constant ranges that changed since its previous
  // draw, so one draw can split its writes differently between frames: those
  // pair by register, with the value the previous frame's draw saw - written
  // by itself or by one of the kLookBack draws before it. Scratch per ALU
  // register: the previous stream offset, valid where `found` is the stamp.
  std::vector<uint32_t> prev_at(kAluEnd - kAluFirst), need(kAluEnd - kAluFirst),
      found(kAluEnd - kAluFirst);
  uint32_t stamp = 0;
  auto resolve = [&](const ParsedFrame::Segment& seg, uint32_t m) {
    ++stamp;
    uint32_t needed = 0;
    for (uint32_t e = seg.alu_begin; e < seg.alu_end; ++e) {
      for (uint32_t d = 0; d < cur.alu_count[e]; ++d) {
        const uint32_t r = cur.alu_reg[e] + d - kAluFirst;
        if (need[r] != stamp) {
          need[r] = stamp;
          ++needed;
        }
      }
    }
    for (uint32_t s = m + 1; s-- > 0 && m - s < kLookBack && needed;) {
      const auto& ps = prev.segments[s];
      for (uint32_t e = ps.alu_end; e-- > ps.alu_begin;) {  // the last write wins
        for (uint32_t d = prev.alu_count[e]; d-- > 0;) {
          const uint32_t r = prev.alu_reg[e] + d - kAluFirst;
          if (need[r] == stamp && found[r] != stamp) {
            found[r] = stamp;
            prev_at[r] = prev.alu_offset[e] + d;
            --needed;
          }
        }
      }
    }
    return !needed;
  };

  for (uint32_t i = 0; i < cur.segments.size(); ++i) {
    if (match[i] == kNone) continue;  // new draw this frame: drawn as is
    const auto& seg = cur.segments[i];
    const auto& pseg = prev.segments[match[i]];
    ++stats.matched;
    const bool same_layout =
        pseg.layout == seg.layout && pseg.alu_end - pseg.alu_begin == seg.alu_end - seg.alu_begin;
    if (!same_layout && !resolve(seg, match[i])) {
      continue;
    }
    // lerp == false: the jump check pass; lerp == true: the write pass.
    auto visit = [&](bool lerp) {
      for (uint32_t e = 0; e < seg.alu_end - seg.alu_begin; ++e) {
        const uint32_t co = cur.alu_offset[seg.alu_begin + e];
        for (uint32_t d = 0; d < cur.alu_count[seg.alu_begin + e]; ++d) {
          const uint32_t po =
              same_layout ? prev.alu_offset[pseg.alu_begin + e] + d
                          : prev_at[cur.alu_reg[seg.alu_begin + e] + d - kAluFirst];
          const uint32_t cb = Load(cur_be, co + d), pb = Load(prev_be, po);
          if (cb == pb) continue;
          const float c = std::bit_cast<float>(cb), p = std::bit_cast<float>(pb);
          if (!std::isfinite(c) || !std::isfinite(p)) continue;
          if (!lerp) {
            if (Jumped(c, p)) {
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
    if (std::isfinite(c) && std::isfinite(p) && Jumped(c, p)) {
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
