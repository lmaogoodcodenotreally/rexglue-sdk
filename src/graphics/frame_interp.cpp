/**
 ******************************************************************************
 * ReXGlue : Xbox 360 runtime                                               *
 ******************************************************************************
 */

#include <rex/graphics/frame_interp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>

#include <rex/graphics/xenos.h>
#include <rex/math.h>

namespace rex::graphics::frame_interp {

namespace {

constexpr uint32_t kAluFirst = 0x4000;  // SHADER_CONSTANT_000_X
constexpr uint32_t kAluEnd = 0x4800;    // first fetch constant register
constexpr uint32_t kFetchEnd = kAluEnd + 32 * 6;
constexpr uint32_t kRbModeControl = 0x2208, kRbCopyDestBase = 0x2319;
// Relative change of one constant between frames beyond which it is no motion.
constexpr float kJump = 0.5f;
// Absolute floor of that change: unit rotation matrix elements and normals
// move by up to 2 in one tick of fast motion (a strike, a camera swing).
constexpr float kJumpFloor = 2.0f;
// How much nearer (sum of |cur - prev| constants) another copy of a mesh must
// be than the same occurrence to pair instead: copies told apart by vertex
// data alone (UI text quads) have near-equal constants and keep their order.
constexpr float kCopyMargin = 1.0f;

bool Jumped(float c, float p) {
  return std::abs(c - p) > std::max(kJumpFloor, kJump * std::max(std::abs(c), std::abs(p)));
}

uint32_t Load(const uint32_t* be, size_t i) { return std::byteswap(be[i]); }

// The value segment `s` writes to ALU register `reg`, if it writes it once,
// finite and among its first 64 constants (all `distance` compares).
std::optional<float> AxisValue(const uint32_t* be, const ParsedFrame& f,
                               const ParsedFrame::Segment& s, uint32_t reg) {
  std::optional<float> v;
  uint32_t flat = 0;
  for (uint32_t e = s.alu_begin; e < s.alu_end; flat += f.alu_count[e++]) {
    if (reg < f.alu_reg[e] || reg >= f.alu_reg[e] + f.alu_count[e]) continue;
    if (v || flat + (reg - f.alu_reg[e]) >= 64) return std::nullopt;
    v = std::bit_cast<float>(Load(be, f.alu_offset[e] + (reg - f.alu_reg[e])));
  }
  return v && std::isfinite(*v) ? v : std::nullopt;
}

uint64_t Mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  return h;
}

// One packed vertex word lerped lane by lane, in the lane layout the vertex
// fetch reads (after the endian swap); 16:16 halves lerp as floats.
uint32_t LerpPacked(uint32_t pw, uint32_t cw, xenos::VertexFormat format, bool is_signed,
                    xenos::Endian e, float t) {
  const uint32_t p = xenos::GpuSwap(pw, e), c = xenos::GpuSwap(cw, e);
  uint32_t out = 0;
  if (format == xenos::VertexFormat::k_16_16_FLOAT) {
    for (uint32_t o : {0u, 16u}) {
      const float a = rex::xenos_half_to_float(uint16_t(p >> o));
      const float b = rex::xenos_half_to_float(uint16_t(c >> o));
      out |= uint32_t(rex::float_to_xenos_half(a + (b - a) * t)) << o;
    }
    return xenos::GpuSwap(out, e);
  }
  std::array<uint32_t, 4> widths{};  // from the lowest bits up
  switch (format) {
    case xenos::VertexFormat::k_8_8_8_8:
      widths = {8, 8, 8, 8};
      break;
    case xenos::VertexFormat::k_2_10_10_10:
      widths = {10, 10, 10, 2};
      break;
    case xenos::VertexFormat::k_10_11_11:
      widths = {11, 11, 10};
      break;
    case xenos::VertexFormat::k_11_11_10:
      widths = {10, 11, 11};
      break;
    case xenos::VertexFormat::k_16_16:
      widths = {16, 16};
      break;
    default:
      return cw;
  }
  for (uint32_t o = 0, k = 0; k < 4 && widths[k]; o += widths[k++]) {
    const uint32_t w = widths[k], mask = (1u << w) - 1;
    int32_t a = int32_t((p >> o) & mask), b = int32_t((c >> o) & mask);
    if (is_signed) {
      a = int32_t(uint32_t(a) << (32 - w)) >> (32 - w);
      b = int32_t(uint32_t(b) << (32 - w)) >> (32 - w);
    }
    out |= (uint32_t(std::lround(float(a) + float(b - a) * t)) & mask) << o;
  }
  return xenos::GpuSwap(out, e);
}

// A rigid transform's 3x3 (rows orthogonal, of one length) in both frames,
// as 3 rows of 4 floats (the 4th unused): rewritten as the rotation between,
// so a limb that turned far in one tick (a strike) keeps its size instead of
// the per-element lerp's cos(angle / 2). A mirrored pair (view-space
// matrices) turns as its negation. False if either is not rigid.
bool LerpRotation(const float* p, const float* c, float t, float* out) {
  float scale[2], q[2][4];  // q: w, x, y, z
  bool mirrored[2];
  for (int k = 0; k < 2; ++k) {
    const float* m = k ? c : p;
    float len[3];
    for (int r = 0; r < 3; ++r) {
      len[r] = std::sqrt(m[r * 4] * m[r * 4] + m[r * 4 + 1] * m[r * 4 + 1] +
                         m[r * 4 + 2] * m[r * 4 + 2]);
    }
    if (!(len[0] > 1e-3f) || std::abs(len[1] - len[0]) > 0.02f * len[0] ||
        std::abs(len[2] - len[0]) > 0.02f * len[0]) {
      return false;
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = i + 1; j < 3; ++j) {
        const float dot =
            m[i * 4] * m[j * 4] + m[i * 4 + 1] * m[j * 4 + 1] + m[i * 4 + 2] * m[j * 4 + 2];
        if (std::abs(dot) > 0.02f * len[0] * len[0]) return false;
      }
    }
    const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                      m[2] * (m[4] * m[9] - m[5] * m[8]);
    mirrored[k] = det < 0;
    const float s = (mirrored[k] ? -1.0f : 1.0f) / len[0];
    const float m0 = m[0] * s, m1 = m[1] * s, m2 = m[2] * s, m3 = m[4] * s, m4 = m[5] * s,
                m5 = m[6] * s, m6 = m[8] * s, m7 = m[9] * s, m8 = m[10] * s;
    float* o = q[k];
    const float tr = m0 + m4 + m8;
    if (tr > 0) {
      const float f = std::sqrt(tr + 1) * 2;
      o[0] = f / 4, o[1] = (m7 - m5) / f, o[2] = (m2 - m6) / f, o[3] = (m3 - m1) / f;
    } else if (m0 > m4 && m0 > m8) {
      const float f = std::sqrt(1 + m0 - m4 - m8) * 2;
      o[0] = (m7 - m5) / f, o[1] = f / 4, o[2] = (m1 + m3) / f, o[3] = (m2 + m6) / f;
    } else if (m4 > m8) {
      const float f = std::sqrt(1 + m4 - m0 - m8) * 2;
      o[0] = (m2 - m6) / f, o[1] = (m1 + m3) / f, o[2] = f / 4, o[3] = (m5 + m7) / f;
    } else {
      const float f = std::sqrt(1 + m8 - m0 - m4) * 2;
      o[0] = (m3 - m1) / f, o[1] = (m2 + m6) / f, o[2] = (m5 + m7) / f, o[3] = f / 4;
    }
    scale[k] = len[0];
  }
  if (mirrored[0] != mirrored[1]) return false;
  // Normalized lerp along the shorter arc: the slerp's own midpoint at t = 0.5.
  const float sign =
      q[0][0] * q[1][0] + q[0][1] * q[1][1] + q[0][2] * q[1][2] + q[0][3] * q[1][3] < 0 ? -1.0f
                                                                                          : 1.0f;
  float r[4];
  for (int k = 0; k < 4; ++k) r[k] = q[0][k] + (sign * q[1][k] - q[0][k]) * t;
  const float n = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
  if (!(n > 1e-6f)) return false;
  const float w = r[0] / n, x = r[1] / n, y = r[2] / n, z = r[3] / n;
  const float s = (scale[0] + (scale[1] - scale[0]) * t) * (mirrored[0] ? -1.0f : 1.0f);
  const float m[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                      2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                      2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
  for (int row = 0; row < 3; ++row) {
    for (int k = 0; k < 3; ++k) out[row * 4 + k] = m[row * 3 + k] * s;
  }
  return true;
}

bool HalvesJumped(uint32_t pw, uint32_t cw, xenos::Endian e) {
  const uint32_t p = xenos::GpuSwap(pw, e), c = xenos::GpuSwap(cw, e);
  for (uint32_t o : {0u, 16u}) {
    const float a = rex::xenos_half_to_float(uint16_t(p >> o));
    const float b = rex::xenos_half_to_float(uint16_t(c >> o));
    if (std::isfinite(a) && std::isfinite(b) && Jumped(b, a)) {
      return true;
    }
  }
  return false;
}

}  // namespace

void Parse(const uint32_t* be, size_t n, ParsedFrame& out) {
  out.segments.clear();
  out.alu_offset.clear();
  out.alu_count.clear();
  out.alu_reg.clear();
  uint64_t vs = 0, ps = 0, layout = 0, textures = 0, sampled = 0;
  uint32_t alu_begin = 0;
  // A texture the frame rendered (resolved) itself is the same address for
  // every image copied there (each menu icon, drawn then copied, then shown):
  // it is told apart by the textures the last draw before the copy changed
  // (not merely re-set: the first icon re-sets the whole block).
  uint32_t mode = 0, copy_dest = 0;
  uint64_t changed = 0, drawn = 0;
  std::array<uint32_t, 32> bound{};
  std::vector<std::pair<uint32_t, uint64_t>> resolved;  // copy destination, what was drawn
  auto close = [&](uint64_t draw, uint64_t shaders, size_t packet) {
    if (xenos::EdramMode(mode & 7) == xenos::EdramMode::kCopy) {
      const uint32_t dest = copy_dest & 0xFFFFF000u;
      auto r = std::find_if(resolved.begin(), resolved.end(),
                            [&](const auto& x) { return x.first == dest; });
      if (r == resolved.end()) r = resolved.insert(r, {dest, 0});
      r->second = drawn;
    } else {
      drawn = changed;
    }
    out.segments.push_back({Mix(draw, textures), draw, shaders,
                            sampled ? Mix(draw, sampled) : draw, layout,
                            alu_begin,
                            uint32_t(out.alu_offset.size()), uint32_t(packet)});
    layout = 0;
    textures = 0;
    sampled = 0;
    changed = 0;
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
            const uint32_t texture = Load(be, i + 1 + (reg - base)) & 0xFFFFF000u;
            textures = Mix(Mix(textures, reg), texture);
            if (uint32_t& slot = bound[(reg - kAluEnd) / 6]; slot != texture) {
              slot = texture;
              changed = Mix(Mix(changed, reg), texture);
            }
            for (const auto& [dest, what] : resolved) {
              if (dest == texture) {
                textures = Mix(textures, what);
                sampled = Mix(sampled, what);
              }
            }
          }
        }
        for (uint32_t reg : {kRbModeControl, kRbCopyDestBase}) {
          const size_t at = one_reg ? i + count : i + 1 + (reg - base);
          if (reg >= base && reg < last && at < n) {
            (reg == kRbModeControl ? mode : copy_dest) = Load(be, at);
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
            close(Mix(Mix(Mix(vs, ps), Load(be, i + 2)), count >= 4 ? Load(be, i + 3) : 0),
                  Mix(Mix(vs, ps), Load(be, i + 2) & 0xFFFF), i);
            break;
          case xenos::PM4_DRAW_INDX_2:  // initiator, inline indices
            close(Mix(Mix(vs, ps), Load(be, i + 1)), Mix(Mix(vs, ps), Load(be, i + 1) & 0xFFFF), i);
            break;
        }
        i += 1 + count;
        break;
      }
    }
  }
  close(0x7A11, 0x7A11, n);  // packets after the last draw (the swap resolve setup)
}

Stats Interpolate(const uint32_t* prev_be, const ParsedFrame& prev, uint32_t* cur_be,
                  const ParsedFrame& cur, float t,
                  std::vector<std::pair<uint32_t, uint32_t>>* lerped_pairs) {
  Stats stats;
  if (lerped_pairs) {
    lerped_pairs->clear();
  }
  stats.segments = uint32_t(cur.segments.size());
  // Draws pair within their signature, not by stream position: meshes draw
  // once per pass with the same signature in each, and draws in between are
  // added, culled and re-sorted, so a search ahead pairs across passes and
  // loses what it skips. Copies of one mesh (props, floor sections) share a
  // signature and get re-sorted by depth as the camera moves, so within a
  // signature a draw pairs with the unused one whose constants are nearest -
  // a static prop's are equal - ties going to the same occurrence (the same
  // pass), else to the nearest in stream order.
  constexpr uint32_t kNone = UINT32_MAX;
  std::vector<uint32_t> match(cur.segments.size(), kNone);
  std::vector<bool> used(prev.segments.size());
  std::vector<bool> sure(cur.segments.size());  // its signature's count did not change
  std::vector<uint32_t> copies(cur.segments.size());  // draws of its key, where it had copies
  std::vector<uint32_t> mark(kAluEnd - kAluFirst), mark_at(kAluEnd - kAluFirst);
  uint32_t mark_stamp = 0;
  // The cur draw's ALU constants, by register, for `distance` to a draw with
  // another constant layout.
  const ParsedFrame::Segment* marked = nullptr;
  auto mark_cur = [&](const ParsedFrame::Segment& cs) {
    if (marked == &cs) return;
    marked = &cs;
    ++mark_stamp;
    for (uint32_t e = cs.alu_begin; e < cs.alu_end; ++e) {
      for (uint32_t d = 0; d < cur.alu_count[e]; ++d) {
        const uint32_t r = cur.alu_reg[e] + d - kAluFirst;
        mark[r] = mark_stamp;
        mark_at[r] = cur.alu_offset[e] + d;
      }
    }
  };
  // Sum of |cur - prev| over the ALU constants both draws write, given up
  // past `limit`. shortcut: only the first 64 shared dwords (copies differ in
  // their world matrix, written first); compare more if copies still mispair.
  auto distance = [&](const ParsedFrame::Segment& cs, const ParsedFrame::Segment& ps,
                      float limit) {
    float sum = 0;
    uint32_t common = 0;
    if (cs.layout == ps.layout && cs.alu_end - cs.alu_begin == ps.alu_end - ps.alu_begin) {
      for (uint32_t e = 0; e < cs.alu_end - cs.alu_begin && common < 64; ++e) {
        const uint32_t co = cur.alu_offset[cs.alu_begin + e], po = prev.alu_offset[ps.alu_begin + e];
        for (uint32_t d = 0; d < cur.alu_count[cs.alu_begin + e] && common < 64; ++d, ++common) {
          const float c = std::bit_cast<float>(Load(cur_be, co + d));
          const float p = std::bit_cast<float>(Load(prev_be, po + d));
          if (std::isfinite(c) && std::isfinite(p) && (sum += std::abs(c - p)) > limit) {
            return sum;
          }
        }
      }
      return common ? sum : std::numeric_limits<float>::infinity();
    }
    mark_cur(cs);
    for (uint32_t e = ps.alu_begin; e < ps.alu_end && common < 64; ++e) {
      for (uint32_t d = 0; d < prev.alu_count[e] && common < 64; ++d) {
        const uint32_t r = prev.alu_reg[e] + d - kAluFirst;
        if (mark[r] != mark_stamp) continue;
        ++common;
        const float c = std::bit_cast<float>(Load(cur_be, mark_at[r]));
        const float p = std::bit_cast<float>(Load(prev_be, prev.alu_offset[e] + d));
        if (std::isfinite(c) && std::isfinite(p) && (sum += std::abs(c - p)) > limit) {
          return sum;
        }
      }
    }
    return common ? sum : std::numeric_limits<float>::infinity();
  };
  auto pair_by = [&](uint64_t ParsedFrame::Segment::*key, bool count_sure) {
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
      const bool same_count = size_t(pb - pa) == b - a;
      if (same_count && b - a == 1) {
        match[cur_order[a].second] = pa->second;
        used[pa->second] = true;
        sure[cur_order[a].second] = count_sure;
        continue;
      }
      auto take = [&](int64_t at, uint32_t q, bool is_sure) {
        match[at] = q;
        used[q] = true;
        sure[at] = is_sure;
        copies[at] = uint32_t(b - a);
      };
      // Many copies (instanced meshes, particles: hundreds): candidates are
      // searched outward along the constant that spreads the copies most (an
      // instance position), as |its difference| bounds `distance` from below.
      std::vector<std::pair<float, uint32_t>> axis;  // prev copies by its value
      uint32_t axis_reg = 0;
      if ((b - a) * size_t(pb - pa) >= 1024) {
        // Ranked on a sample, then checked on every copy.
        auto spread = [&](uint32_t reg, size_t step) {
          float lo = std::numeric_limits<float>::infinity(), hi = -lo;
          auto widen = [&](std::optional<float> v) {
            if (v) {
              lo = std::min(lo, *v);
              hi = std::max(hi, *v);
            }
            return v.has_value();
          };
          for (size_t c = a; c < b; c += step) {
            if (!widen(AxisValue(cur_be, cur, cur.segments[cur_order[c].second], reg))) return -1.0f;
          }
          for (size_t k = 0; k < size_t(pb - pa); k += step) {
            if (!widen(AxisValue(prev_be, prev, prev.segments[pa[k].second], reg))) return -1.0f;
          }
          return hi - lo;
        };
        const auto& first = cur.segments[cur_order[a].second];
        const size_t step = std::max<size_t>(1, (b - a + size_t(pb - pa)) / 32);
        std::vector<std::pair<float, uint32_t>> ranked;
        for (uint32_t e = first.alu_begin, flat = 0; e < first.alu_end && flat < 64;
             flat += cur.alu_count[e++]) {
          for (uint32_t d = 0; d < cur.alu_count[e] && flat + d < 64; ++d) {
            if (const float s = spread(cur.alu_reg[e] + d, step); s > 0) {
              ranked.emplace_back(s, cur.alu_reg[e] + d);
            }
          }
        }
        std::sort(ranked.rbegin(), ranked.rend());
        for (size_t r = 0; r < std::min<size_t>(ranked.size(), 4) && !axis_reg; ++r) {
          axis_reg = spread(ranked[r].second, 1) > 0 ? ranked[r].second : 0;
        }
      }
      if (axis_reg) {
        for (auto q = pa; q != pb; ++q) {
          axis.emplace_back(*AxisValue(prev_be, prev, prev.segments[q->second], axis_reg), q->second);
        }
        std::sort(axis.begin(), axis.end());
      }
      // Visits the prev copies `visit` asks for, nearest on the axis first,
      // until they are farther on it than `reach()`; without an axis, nearest
      // in stream order first (prev copies are sorted by index).
      auto outward = [&](uint32_t at, auto reach, auto visit) {
        if (axis.empty()) {
          auto hi = std::lower_bound(pa, pb, at, [](const auto& x, uint32_t v) { return x.second < v; });
          auto lo = hi;
          while (lo != pa || hi != pb) {
            const bool below = lo != pa && (hi == pb || at - (lo - 1)->second <= hi->second - at);
            if (!visit(below ? (--lo)->second : (hi++)->second)) break;
          }
          return;
        }
        const float x = *AxisValue(cur_be, cur, cur.segments[at], axis_reg);
        auto hi = std::lower_bound(axis.begin(), axis.end(), std::make_pair(x, 0u));
        auto lo = hi;
        while (lo != axis.begin() || hi != axis.end()) {
          const float below = lo != axis.begin() ? x - (lo - 1)->first : std::numeric_limits<float>::infinity();
          const float above = hi != axis.end() ? hi->first - x : std::numeric_limits<float>::infinity();
          if (std::min(below, above) > reach()) break;
          if (!visit(below <= above ? (--lo)->second : (hi++)->second)) break;
        }
      };
      // First every copy that its own occurrence fits within the margin of its
      // nearest, so that a copy that moved toward another's place does not
      // take that one's pair; then the rest, nearest among those left. Only
      // a copy nearer than that is searched for: instanced meshes (floor
      // tiles, hundreds of copies) mostly keep their order and constants.
      if (same_count) {
        for (size_t c = a; c < b; ++c) {
          const int64_t at = cur_order[c].second;
          const auto occurrence = pa + (c - a);
          const float bound =
              distance(cur.segments[at], prev.segments[occurrence->second],
                       std::numeric_limits<float>::infinity()) - kCopyMargin;
          bool nearer = false;
          // A draw without constants (text quads, particles) is equally far from all.
          if (bound > 0 && cur.segments[at].alu_begin != cur.segments[at].alu_end) {
            outward(uint32_t(at), [&] { return bound; }, [&](uint32_t q) {
              nearer = distance(cur.segments[at], prev.segments[q], bound) < bound;
              return !nearer;
            });
          }
          if (!nearer) {
            take(at, occurrence->second, count_sure);
          }
        }
      }
      // Guesses: with the count changed, the copy left over is as likely
      // another object (one that came into view at one end of a row while
      // another left at the other); with copies re-sorted, as many in both
      // frames, a copy's first choice is its own. Globally nearest pair
      // first, ties to the nearest in stream order: each copy left keeps its
      // nearest unused candidate in a min-heap, searched again once taken.
      using Option = std::tuple<float, int64_t, uint32_t, uint32_t, bool>;  // distance, tie, at, q, first
      auto consider = [&](uint32_t at, uint32_t q, bool& found, Option& best) {
        const float limit = found ? std::get<0>(best) : std::numeric_limits<float>::infinity();
        const Option o{distance(cur.segments[at], prev.segments[q], limit),
                       std::abs(int64_t(q) - at), at, q, true};
        if (!found || o < best) {
          best = o;
          found = true;
        }
      };
      auto best_for = [&](uint32_t at, Option& best) {
        bool found = false;
        const bool constants = cur.segments[at].alu_begin != cur.segments[at].alu_end;
        outward(
            at,
            [&] { return found ? std::get<0>(best) : std::numeric_limits<float>::infinity(); },
            [&](uint32_t q) {
              if (!used[q]) consider(at, q, found, best);
              // In stream order, once none can be nearer the rest only lose the tie.
              return !(found && axis.empty() && (!constants || std::get<0>(best) == 0));
            });
        return found;
      };
      std::vector<Option> heap;
      for (size_t c = a; c < b; ++c) {
        Option o;
        if (match[cur_order[c].second] == kNone && best_for(cur_order[c].second, o)) {
          heap.push_back(o);
        }
      }
      std::make_heap(heap.begin(), heap.end(), std::greater<>());
      while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<>());
        const auto [d, tie, at, q, first] = heap.back();
        heap.pop_back();
        if (!used[q]) {
          take(at, q, same_count && count_sure && first);
        } else if (Option o; best_for(at, o)) {
          std::get<4>(o) = false;
          heap.push_back(o);
          std::push_heap(heap.begin(), heap.end(), std::greater<>());
        }
      }
    }
  };
  // Copies as many in both frames are the same objects: within a key they
  // pair surely. Where the count changed, the pairings left are guesses (one
  // object came into view at one end of a row while another left).
  pair_by(&ParsedFrame::Segment::signature, true);
  // Then by what a draw shows from textures the frame rendered itself: the
  // textures it sets besides can change with the draw order (an icon drawn
  // first re-sets a texture the others inherit).
  pair_by(&ParsedFrame::Segment::content, true);
  // Then by the draw alone: which textures a draw sets itself depends on what
  // the draw before it left bound, which changes when draws are re-sorted, and
  // a fighter's face shows another texture with each expression.
  pair_by(&ParsedFrame::Segment::draw, true);
  // Last by the shaders alone, as guesses: one mesh may be drawn from new
  // index data each tick, but other meshes share its shaders too.
  pair_by(&ParsedFrame::Segment::shaders, false);
  // The game uploads only the constant ranges that changed since its previous
  // draw, so one draw can split its writes differently between frames: those
  // pair by register, with the value the previous frame's draw saw - the last
  // write before it, however far back (a skinned mesh's bone palette is often
  // written many draws before). A register the previous frame never wrote by
  // then keeps its value. Scratch per ALU register: the previous stream
  // offset, valid where `found` is the stamp.
  constexpr uint32_t kUnwritten = UINT32_MAX;
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
          prev_at[r] = kUnwritten;
          ++needed;
        }
      }
    }
    for (uint32_t s = m + 1; s-- > 0 && needed;) {
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
  };

  // Sum of |cur - prev| over every constant the cur draw writes, against what
  // the prev draw saw: comparable between draws of other constant layouts.
  auto seen = [&](uint32_t i, uint32_t m, float limit) {
    const auto& seg = cur.segments[i];
    const auto& pseg = prev.segments[m];
    // One layout: the prev draw wrote all it saw itself.
    const bool same_layout =
        pseg.layout == seg.layout && pseg.alu_end - pseg.alu_begin == seg.alu_end - seg.alu_begin;
    if (!same_layout) {
      resolve(seg, m);
    }
    float sum = 0;
    for (uint32_t e = seg.alu_begin; e < seg.alu_end; ++e) {
      for (uint32_t d = 0; d < cur.alu_count[e]; ++d) {
        const uint32_t po = same_layout ? prev.alu_offset[pseg.alu_begin + (e - seg.alu_begin)] + d
                                        : prev_at[cur.alu_reg[e] + d - kAluFirst];
        if (po == kUnwritten) continue;
        const float c = std::bit_cast<float>(Load(cur_be, cur.alu_offset[e] + d));
        const float p = std::bit_cast<float>(Load(prev_be, po));
        if (std::isfinite(c) && std::isfinite(p) && (sum += std::abs(c - p)) > limit) {
          return sum;
        }
      }
    }
    return sum;
  };

  // Two copies drawn in the other order this frame can trade the textures
  // they set themselves (two rickshaws meeting halfway), or show another
  // texture each tick (a face's expression: two fighters' faces trade theirs),
  // so their signatures: each paired with the other's previous draw. Within a
  // draw whose copies carry several signatures, two pairings that are nearer
  // swapped, by the constants each draw saw, are swapped.
  {
    std::vector<std::pair<uint64_t, uint32_t>> group;  // draw, cur draw
    for (uint32_t i = 0; i < cur.segments.size(); ++i) {
      if (match[i] != kNone) {
        group.emplace_back(cur.segments[i].draw, i);
      }
    }
    std::sort(group.begin(), group.end());
    std::vector<std::pair<uint32_t, float>> uneasy;  // cur draw paired off by more than the margin
    for (size_t a = 0, b; a < group.size(); a = b) {
      bool mixed = false;
      for (b = a + 1; b < group.size() && group[b].first == group[a].first; ++b) {
        mixed |= cur.segments[group[b].second].signature != cur.segments[group[a].second].signature;
      }
      // shortcut: a draw with more than 64 copies (a particle cloud) is not
      // searched; upgrade if such swaps show.
      if (!mixed || b - a > 64) continue;
      uneasy.clear();
      for (size_t x = a; x < b; ++x) {
        const uint32_t i = group[x].second;
        const float d = seen(i, match[i], std::numeric_limits<float>::infinity());
        if (d > kCopyMargin) uneasy.emplace_back(i, d);
      }
      for (size_t x = 0; x < uneasy.size(); ++x) {
        for (size_t y = x + 1; y < uneasy.size(); ++y) {
          auto& [i, at_i] = uneasy[x];
          auto& [j, at_j] = uneasy[y];
          // Copies of one signature already paired by their nearest constants.
          if (cur.segments[i].signature == cur.segments[j].signature) continue;
          const float now = at_i + at_j - kCopyMargin;
          const float di = seen(i, match[j], now);
          if (di >= now) continue;
          const float dj = seen(j, match[i], now - di);
          if (di + dj >= now) continue;
          std::swap(match[i], match[j]);
          at_i = di;
          at_j = dj;
          sure[i] = sure[j] = false;
        }
      }
    }
  }

  // A draw left as is (new this frame: a prop come into view or swapped LOD;
  // or a guess that jumped) still writes the camera and lights the lerped
  // draws wrote, so it would sit half a tick ahead of the scene: every whole
  // vec4 it writes that is bit-equal to what the last cleanly lerped draw
  // wrote to that register takes that draw's lerped value. So does a sure
  // pairing that jumped, as a per-pass light or eye vector the game switched
  // to another source (draws that carried that source last frame lerp it
  // cleanly) must not blend two sources. Per ALU register: that draw's value
  // as recorded << 32 | as lerped (all ones, a NaN never borrowed: unset).
  std::vector<uint64_t> lerped(kAluEnd - kAluFirst, ~0ull);
  std::vector<uint32_t> saved;  // a jumped draw's constants before its lerp
  std::vector<std::pair<uint32_t, uint32_t>> turned;  // cur offset, value: rotation entries
  // Per vec4 register: the draw (stamp) in which one of its values moved visibly.
  std::vector<uint32_t> moved_at((kAluEnd - kAluFirst) / 4);
  uint32_t moved_stamp = 0;
  auto trivial = [](uint32_t v) { return !(v & 0x7FFFFFFF) || (v & 0x7FFFFFFF) == 0x3F800000; };
  auto borrow = [&](const ParsedFrame::Segment& seg, const uint32_t* own) {
    for (uint32_t e = seg.alu_begin, at = 0; e < seg.alu_end; at += cur.alu_count[e++]) {
      const uint32_t reg = cur.alu_reg[e], count = cur.alu_count[e];
      for (uint32_t d = (4 - reg % 4) % 4; d + 4 <= count; d += 4) {
        const uint64_t* rec = &lerped[reg + d - kAluFirst];
        const uint32_t co = cur.alu_offset[e] + d;
        bool equal = true, plain = true;  // plain: all 0 / +-1, as likely a coincidence
        for (uint32_t k = 0; k < 4 && equal; ++k) {
          const uint32_t cb = own ? own[at + d + k] : Load(cur_be, co + k);
          equal = uint32_t(rec[k] >> 32) == cb;
          plain &= trivial(cb);
        }
        if (!equal || plain) continue;
        for (uint32_t k = 0; k < 4; ++k) {
          cur_be[co + k] = std::byteswap(uint32_t(rec[k]));
        }
      }
    }
  };

  for (uint32_t i = 0; i < cur.segments.size(); ++i) {
    const auto& seg = cur.segments[i];
    if (match[i] == kNone) {  // new draw this frame
      borrow(seg, nullptr);
      continue;
    }
    const auto& pseg = prev.segments[match[i]];
    ++stats.matched;
    const bool same_layout =
        pseg.layout == seg.layout && pseg.alu_end - pseg.alu_begin == seg.alu_end - seg.alu_begin;
    if (!same_layout) {
      resolve(seg, match[i]);
    }
    // A jump only in values a cleanly lerped draw wrote too is a per-pass
    // light or eye vector the game switched to another source, not another
    // draw: the pairing holds (a fighter's part whose texture, so signature,
    // changed with an expression), and those values are borrowed.
    bool switched = false;
    // The prev offset of the constant at run e, element d (kUnwritten: none).
    auto prev_of = [&](uint32_t e, uint32_t d) {
      return same_layout ? prev.alu_offset[pseg.alu_begin + e] + d
                         : prev_at[cur.alu_reg[seg.alu_begin + e] + d - kAluFirst];
    };
    // Between copies of one draw, a whole number that changed alone in its
    // vec4 (the rest bit-equal) is a count or an index, not motion: MK's
    // seagulls divide the vertex index by their mesh's vertex count (c200.x,
    // 122 or 126 by mesh) to find the instance, so a value in between pulls
    // every vertex from another bird. The pairing joined two different
    // meshes: drawn as is. A lone draw keeps lerping (a world row moving
    // along one axis can be whole numbers too).
    bool restructured = false;
    auto whole = [](float v) { return v == std::trunc(v) && std::abs(v) >= 2.0f; };
    // pass 0: the jump check; 1: the write pass, recorded for borrowing; 2: not recorded.
    ++moved_stamp;
    bool moved_any = false;
    auto visit = [&](int pass) {
      size_t next_turned = 0;
      bool clean = true;
      for (uint32_t e = 0; e < seg.alu_end - seg.alu_begin; ++e) {
        const uint32_t co = cur.alu_offset[seg.alu_begin + e];
        uint64_t* rec = &lerped[cur.alu_reg[seg.alu_begin + e] - kAluFirst];
        const uint32_t count = cur.alu_count[seg.alu_begin + e];
        for (uint32_t d = 0; d < count; ++d) {
          // Most constants did not change: skipped in bulk (recorded as they are).
          if (same_layout && (!pass || turned.empty())) {
            const uint32_t* c0 = cur_be + co;
            const uint32_t to = uint32_t(
                std::mismatch(c0 + d, c0 + count, prev_be + prev_of(e, 0) + d).first - c0);
            for (; pass == 1 && d < to; ++d) {
              const uint32_t cb = Load(cur_be, co + d);
              rec[d] = uint64_t(cb) << 32 | cb;
            }
            d = to;
            if (d == count) break;
          }
          const uint32_t po = prev_of(e, d);
          const uint32_t cb = Load(cur_be, co + d);
          if (pass == 1) {
            rec[d] = uint64_t(cb) << 32 | cb;
          }
          if (pass && next_turned < turned.size() && turned[next_turned].first == co + d) {
            const uint32_t value = turned[next_turned++].second;
            if (pass == 1) {
              rec[d] = uint64_t(cb) << 32 | value;
            }
            cur_be[co + d] = std::byteswap(value);
            continue;
          }
          if (po == kUnwritten) continue;
          const uint32_t pb = Load(prev_be, po);
          if (cb == pb) continue;
          const float c = std::bit_cast<float>(cb), p = std::bit_cast<float>(pb);
          if (!std::isfinite(c) || !std::isfinite(p)) continue;
          if (!pass) {
            if (!restructured && copies[i] > 1 && whole(c) && whole(p)) {
              const uint32_t lane = (cur.alu_reg[seg.alu_begin + e] + d) % 4;
              bool alone = d >= lane && d - lane + 4 <= count;
              for (uint32_t k = 0; alone && k < 4; ++k) {
                const uint32_t other = prev_of(e, d - lane + k);
                alone = k == lane || (other != kUnwritten &&
                                      Load(prev_be, other) == Load(cur_be, co + d - lane + k));
              }
              restructured = alone;
            }
            if (std::abs(c - p) > 0.05f) {
              moved_at[(cur.alu_reg[seg.alu_begin + e] + d - kAluFirst) / 4] = moved_stamp;
              moved_any = true;
            }
            if (Jumped(c, p)) {
              if (rec[d] == ~0ull || uint32_t(rec[d] >> 32) != cb || trivial(cb)) {
                clean = false;
              } else {
                switched = true;
              }
            }
            continue;
          }
          const uint32_t value = std::bit_cast<uint32_t>(p + (c - p) * t);
          if (pass == 1) {
            rec[d] = uint64_t(cb) << 32 | value;
          }
          cur_be[co + d] = std::byteswap(value);
        }
      }
      return clean;
    };
    // A sure pairing of a lone draw is the same draw, so a jump there is fast
    // motion (a strike, a spin) and lerping keeps the model whole; a guessed
    // pairing that jumped is likely another draw: drawn as is. So is a copy
    // that jumped: as many copies in both frames does not make it the same
    // object (BO's static props blinked halfway to another copy).
    const bool jumped = !visit(0);
    if (restructured) {
      ++stats.jumped;
      borrow(seg, nullptr);
      continue;
    }
    if (jumped) {
      ++stats.jumped;
      if (!sure[i] || copies[i] > 1) {
        borrow(seg, nullptr);
        continue;
      }
    }
    const bool redo = jumped || switched;
    if (redo) {
      saved.clear();
      for (uint32_t e = seg.alu_begin; e < seg.alu_end; ++e) {
        for (uint32_t d = 0; d < cur.alu_count[e]; ++d) {
          saved.push_back(Load(cur_be, cur.alu_offset[e] + d));
        }
      }
    }
    // Rigid 3x3 blocks (world, bone matrices) that visibly turned lerp as
    // rotations. shortcut: blocks within one write run, starting on a vec4;
    // a matrix split across runs lerps per element.
    turned.clear();
    for (uint32_t e = 0; moved_any && e < seg.alu_end - seg.alu_begin; ++e) {
      const uint32_t reg = cur.alu_reg[seg.alu_begin + e], count = cur.alu_count[seg.alu_begin + e];
      const uint32_t co = cur.alu_offset[seg.alu_begin + e];
      for (uint32_t d = (4 - reg % 4) % 4; d + 11 <= count; d += 4) {
        const uint32_t row = (reg + d - kAluFirst) / 4;
        if (moved_at[row] != moved_stamp && moved_at[row + 1] != moved_stamp &&
            moved_at[row + 2] != moved_stamp) {
          continue;
        }
        float p[11], c[11], out[11];
        for (uint32_t k = 0; k < 11; ++k) {
          c[k] = std::bit_cast<float>(Load(cur_be, co + d + k));
        }
        // Most windows are no rigid block: told by its first two rows alone.
        const float len0 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        const float len1 = c[4] * c[4] + c[5] * c[5] + c[6] * c[6];
        if (!(len0 > 1e-6f) || std::abs(len1 - len0) > 0.05f * len0 ||
            std::abs(c[0] * c[4] + c[1] * c[5] + c[2] * c[6]) > 0.05f * len0) {
          continue;
        }
        float moved = 0;
        bool known = true;
        for (uint32_t k = 0; k < 11 && known; ++k) {
          const uint32_t po = prev_of(e, d + k);
          known = po != kUnwritten;
          if (known && k % 4 != 3) {
            p[k] = std::bit_cast<float>(Load(prev_be, po));
            moved = std::max(moved, std::abs(c[k] - p[k]));
          }
        }
        // Under ~6 degrees the per-element lerp shrinks it by under 0.2%.
        if (!known || !(moved * moved > 0.01f * len0) ||
            !LerpRotation(p, c, t, out)) {
          continue;
        }
        for (uint32_t k = 0; k < 11; ++k) {
          if (k % 4 != 3) turned.emplace_back(co + d + k, std::bit_cast<uint32_t>(out[k]));
        }
        d += 8;  // past the block's other rows
      }
    }
    ++stats.lerped;
    visit(redo ? 2 : 1);
    if (redo) {
      borrow(seg, saved.data());
    }
    if (lerped_pairs) {
      lerped_pairs->emplace_back(seg.packet, pseg.packet);
    }
  }
  return stats;
}

bool LerpVertexFloats(const uint32_t* prev, uint32_t* cur, uint32_t vertices,
                      uint32_t stride_words, uint64_t float_words, const PackedWords& packed,
                      uint32_t endian, float t, uint32_t group) {
  const auto e = xenos::Endian(endian);
  auto load = [e](uint32_t word) { return std::bit_cast<float>(xenos::GpuSwap(word, e)); };
  // Instance data (debris: a rigid 3x4 matrix per "vertex") turns as
  // rotations: its rotation words may swing past a jump while tumbling.
  const bool matrices = stride_words >= 11 && (float_words & 0x7FF) == 0x7FF;
  float rp[11], rc[11], turned[11];
  auto rigid = [&](const uint32_t* p, const uint32_t* c) {
    for (uint32_t k = 0; k < 11; ++k) {
      rp[k] = load(p[k]);
      rc[k] = load(c[k]);
    }
    return LerpRotation(rp, rc, t, turned);
  };
  auto zero = [stride_words](const uint32_t* v) {
    return std::all_of(v, v + stride_words, [](uint32_t x) { return !(x & 0x7FFFFFFF); });
  };
  std::vector<uint32_t> cleared;  // instances cleared (or set) this tick: kept as they are
  // One pass: every changed word is a float that did not jump; per float
  // word, its extent over the vertices and how far it moved.
  std::array<float, 64> lo, hi, moved;
  lo.fill(std::numeric_limits<float>::infinity());
  hi.fill(-std::numeric_limits<float>::infinity());
  moved.fill(0);
  uint64_t changed = 0, packed_changed = 0;
  for (uint32_t v = 0, i = 0; v < vertices; ++v) {
    for (uint32_t w = 0; w < stride_words; ++w, ++i) {
      const bool is_float = (float_words >> w) & 1;
      if (!is_float) {
        if (prev[i] != cur[i]) {
          if (!((packed.words >> w) & 1) ||
              (xenos::VertexFormat(packed.format[w]) == xenos::VertexFormat::k_16_16_FLOAT &&
               HalvesJumped(prev[i], cur[i], e))) {
            return false;
          }
          packed_changed |= uint64_t(1) << w;
        }
        continue;
      }
      if (prev[i] == cur[i]) continue;
      const float c = load(cur[i]);
      if (!std::isfinite(c)) continue;
      {
        const float p = load(prev[i]);
        if (!std::isfinite(p)) continue;
        if (Jumped(c, p)) {
          const uint32_t* pv = prev + (i - w);
          const uint32_t* cv = cur + (i - w);
          if (matrices && zero(pv) != zero(cv)) {
            cleared.push_back(v);
            i += stride_words - w;
            break;
          }
          if (!(matrices && w < 11 && w % 4 != 3 && rigid(pv, cv))) {
            return false;
          }
        }
        changed |= uint64_t(1) << w;
        moved[w] = std::max(moved[w], std::abs(c - p));
      }
    }
  }
  // The extents over the vertices, only of the words the rules below judge.
  const uint64_t judged =
      ((float_words & ~changed) ? changed : 0) | (changed & packed.uv_words);
  for (uint32_t v = 0, next_cleared = 0; judged && v < vertices; ++v) {
    if (next_cleared < cleared.size() && cleared[next_cleared] == v) {
      ++next_cleared;
      continue;
    }
    for (uint64_t m = judged; m; m &= m - 1) {
      const uint32_t w = uint32_t(std::countr_zero(m));
      const float c = load(cur[v * stride_words + w]);
      if (!std::isfinite(c)) continue;
      lo[w] = std::min(lo[w], c);
      hi[w] = std::max(hi[w], c);
    }
  }
  // A sprite switched, not moved: every float word that changed (a texture
  // coordinate picking another atlas cell - a stage name, a digit) moved at
  // least its own extent over the vertices while the others (the position)
  // stayed put. In between would show the cell in between: kept as is.
  bool switched = changed && (float_words & ~changed) != 0;
  for (uint64_t m = changed; switched && m; m &= m - 1) {
    const uint32_t w = uint32_t(std::countr_zero(m));
    switched = hi[w] > lo[w] && moved[w] >= hi[w] - lo[w];
  }
  // The same while it slides (menu description text: each glyph moves to its
  // new place and letter in one tick): a texture coordinate (a 2-float
  // attribute) moved at least its own extent. A particle's position may
  // outrun its size; its texture coordinate stays.
  for (uint64_t m = changed & packed.uv_words; !switched && m; m &= m - 1) {
    const uint32_t w = uint32_t(std::countr_zero(m));
    switched = hi[w] > lo[w] && moved[w] >= hi[w] - lo[w];
  }
  // Over a whole string of glyphs the extent spans the atlas: each glyph
  // (primitive) is judged on its own.
  const uint64_t uv_changed = changed & packed.uv_words;
  for (uint32_t g = 0; group > 1 && uv_changed && !switched && g + group <= vertices; g += group) {
    for (uint64_t m = uv_changed; !switched && m; m &= m - 1) {
      const uint32_t w = uint32_t(std::countr_zero(m));
      float glo = std::numeric_limits<float>::infinity(), ghi = -glo, gmoved = 0;
      for (uint32_t v = g; v < g + group; ++v) {
        const float c = load(cur[v * stride_words + w]), p = load(prev[v * stride_words + w]);
        if (!std::isfinite(c) || !std::isfinite(p)) continue;
        glo = std::min(glo, c);
        ghi = std::max(ghi, c);
        gmoved = std::max(gmoved, std::abs(c - p));
      }
      switched = ghi > glo && gmoved >= ghi - glo;
    }
  }
  if (switched) {
    return false;
  }
  for (uint32_t v = 0, next_cleared = 0; v < vertices;
       ++v, prev += stride_words, cur += stride_words) {
    if (next_cleared < cleared.size() && cleared[next_cleared] == v) {
      ++next_cleared;
      continue;
    }
    uint64_t own = 0;  // words written as a rotation
    if (matrices && !std::equal(prev, prev + 11, cur) && rigid(prev, cur)) {
      for (uint32_t k = 0; k < 11; ++k) {
        if (k % 4 != 3) cur[k] = xenos::GpuSwap(std::bit_cast<uint32_t>(turned[k]), e);
      }
      own = 0x777;
    }
    for (uint64_t m = changed & ~own; m; m &= m - 1) {
      const uint32_t w = uint32_t(std::countr_zero(m));
      if (prev[w] == cur[w]) continue;
      const float c = load(cur[w]), p = load(prev[w]);
      if (!std::isfinite(c) || !std::isfinite(p)) continue;
      cur[w] = xenos::GpuSwap(std::bit_cast<uint32_t>(p + (c - p) * t), e);
    }
    for (uint64_t m = packed_changed; m; m &= m - 1) {
      const uint32_t w = uint32_t(std::countr_zero(m));
      if (prev[w] != cur[w]) {
        cur[w] = LerpPacked(prev[w], cur[w], xenos::VertexFormat(packed.format[w]),
                            (packed.is_signed >> w) & 1, e, t);
      }
    }
  }
  return true;
}

uint32_t LerpPrimitives(const uint32_t* prev, uint32_t prev_vertices, uint32_t* cur,
                        uint32_t vertices, uint32_t group, uint32_t stride_words,
                        uint64_t float_words, const PackedWords& packed, uint32_t endian,
                        float t) {
  constexpr uint32_t kReach = 8;
  const auto e = xenos::Endian(endian);
  const uint32_t words = group * stride_words, prims = vertices / group,
                 prev_prims = prev_vertices / group;
  // Sum of |cur - prev| float words of two primitives; infinite where another word differs.
  auto distance = [&](const uint32_t* p, const uint32_t* c) {
    float sum = 0;
    for (uint32_t i = 0; i < words; ++i) {
      const uint32_t w = i % stride_words;
      if (!((float_words >> w) & 1)) {
        if (p[i] != c[i] && !((packed.words >> w) & 1)) {
          return std::numeric_limits<float>::infinity();
        }
        continue;
      }
      const float a = std::bit_cast<float>(xenos::GpuSwap(p[i], e));
      const float b = std::bit_cast<float>(xenos::GpuSwap(c[i], e));
      if (std::isfinite(a) && std::isfinite(b)) sum += std::abs(a - b);
    }
    return sum;
  };
  std::vector<uint32_t> scratch(words);
  uint32_t lerped = 0;
  for (uint32_t k = 0; k < prims; ++k) {
    uint32_t* c = cur + k * words;
    // Most keep their place.
    if (k < prev_prims) {
      std::copy(c, c + words, scratch.begin());
      if (LerpVertexFloats(prev + k * words, scratch.data(), group, stride_words, float_words,
                           packed, endian, t)) {
        std::copy(scratch.begin(), scratch.end(), c);
        ++lerped;
        continue;
      }
    }
    float best = std::numeric_limits<float>::infinity();
    const uint32_t* from = nullptr;
    // Nearest first: the nearer place wins a tie.
    for (uint32_t step = 1; step <= 2 * kReach; ++step) {
      const int64_t q = int64_t(k) + (step & 1 ? -int64_t(step + 1) / 2 : int64_t(step) / 2);
      if (q < 0 || q >= prev_prims) continue;
      if (const float d = distance(prev + q * words, c); d < best) {
        best = d;
        from = prev + q * words;
      }
    }
    if (!from) continue;
    std::copy(c, c + words, scratch.begin());
    if (LerpVertexFloats(from, scratch.data(), group, stride_words, float_words, packed,
                         endian, t)) {
      std::copy(scratch.begin(), scratch.end(), c);
      ++lerped;
    }
  }
  return lerped;
}

}  // namespace rex::graphics::frame_interp
