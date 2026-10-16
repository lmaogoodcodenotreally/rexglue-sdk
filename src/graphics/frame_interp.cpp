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
  std::vector<uint32_t> copies(cur.segments.size());  // TEMP DIAGNOSTIC: draws of its signature
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
  auto pair_by = [&](uint64_t ParsedFrame::Segment::*key, bool sure_key) {
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
        sure[cur_order[a].second] = sure_key;
        continue;
      }
      auto take = [&](int64_t at, uint32_t q, bool is_sure) {
        match[at] = q;
        used[q] = true;
        sure[at] = is_sure;
        copies[at] = uint32_t(b - a);
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
          for (auto q = pa; q != pb && bound > 0 && !nearer; ++q) {
            nearer = distance(cur.segments[at], prev.segments[q->second], bound) < bound;
          }
          if (!nearer) {
            take(at, occurrence->second, sure_key);
          }
        }
      }
      // Guesses: with copies re-sorted or the count changed, the copy left
      // over is as likely another object (one that came into view at one
      // end of a row while another left at the other). Globally nearest pair
      // first, ties to the nearest in stream order: each copy left keeps its
      // nearest unused candidate in a min-heap, searched again once taken.
      using Option = std::tuple<float, int64_t, uint32_t, uint32_t>;  // distance, tie, at, q
      auto consider = [&](uint32_t at, uint32_t q, bool& found, Option& best) {
        const float limit = found ? std::get<0>(best) : std::numeric_limits<float>::infinity();
        const Option o{distance(cur.segments[at], prev.segments[q], limit),
                       std::abs(int64_t(q) - at), at, q};
        if (!found || o < best) {
          best = o;
          found = true;
        }
      };
      // Many copies left (instanced meshes: hundreds): candidates are searched
      // outward along the constant that spreads the copies most (an instance
      // position), as |its difference| bounds `distance` from below.
      std::vector<std::pair<float, uint32_t>> axis;  // unused prev copies by its value
      std::vector<uint32_t> left_cur, left_prev;
      for (size_t c = a; c < b; ++c) {
        if (match[cur_order[c].second] == kNone) left_cur.push_back(cur_order[c].second);
      }
      for (auto q = pa; q != pb; ++q) {
        if (!used[q->second]) left_prev.push_back(q->second);
      }
      uint32_t axis_reg = 0;
      if (left_cur.size() * left_prev.size() >= 1024) {
        // Ranked on a sample, then checked on every copy left.
        auto spread = [&](uint32_t reg, size_t step) {
          float lo = std::numeric_limits<float>::infinity(), hi = -lo;
          for (int f = 0; f < 2; ++f) {
            const auto& left = f ? left_prev : left_cur;
            for (size_t k = 0; k < left.size(); k += step) {
              const auto v = f ? AxisValue(prev_be, prev, prev.segments[left[k]], reg)
                               : AxisValue(cur_be, cur, cur.segments[left[k]], reg);
              if (!v) return -1.0f;
              lo = std::min(lo, *v);
              hi = std::max(hi, *v);
            }
          }
          return hi - lo;
        };
        const auto& first = cur.segments[left_cur[0]];
        std::vector<std::pair<float, uint32_t>> ranked;
        for (uint32_t e = first.alu_begin, flat = 0; e < first.alu_end && flat < 64;
             flat += cur.alu_count[e++]) {
          for (uint32_t d = 0; d < cur.alu_count[e] && flat + d < 64; ++d) {
            const size_t step = std::max<size_t>(1, (left_cur.size() + left_prev.size()) / 32);
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
        for (const uint32_t q : left_prev) {
          axis.emplace_back(*AxisValue(prev_be, prev, prev.segments[q], axis_reg), q);
        }
        std::sort(axis.begin(), axis.end());
      }
      auto best_for = [&](uint32_t at, Option& best) {
        bool found = false;
        if (axis.empty()) {
          for (auto q = pa; q != pb; ++q) {
            if (!used[q->second]) consider(at, q->second, found, best);
          }
          return found;
        }
        const float x = *AxisValue(cur_be, cur, cur.segments[at], axis_reg);
        auto hi = std::lower_bound(axis.begin(), axis.end(), std::make_pair(x, 0u));
        auto lo = hi;
        while (lo != axis.begin() || hi != axis.end()) {
          const float below = lo != axis.begin() ? x - (lo - 1)->first : std::numeric_limits<float>::infinity();
          const float above = hi != axis.end() ? hi->first - x : std::numeric_limits<float>::infinity();
          if (found && std::min(below, above) > std::get<0>(best)) break;
          const uint32_t q = below <= above ? (--lo)->second : (hi++)->second;
          if (!used[q]) consider(at, q, found, best);
        }
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
        const auto [d, tie, at, q] = heap.back();
        heap.pop_back();
        if (!used[q]) {
          take(at, q, false);
        } else if (Option o; best_for(at, o)) {
          heap.push_back(o);
          std::push_heap(heap.begin(), heap.end(), std::greater<>());
        }
      }
    }
  };
  pair_by(&ParsedFrame::Segment::signature, true);
  // Then by the draw alone: which textures a draw sets itself depends on what
  // the draw before it left bound, which changes when draws are re-sorted.
  // Those pairings are guesses: a copy left over from the first pass is
  // as likely another object.
  pair_by(&ParsedFrame::Segment::draw, false);

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
  auto borrow = [&](const ParsedFrame::Segment& seg, const uint32_t* own) {
    auto trivial = [](uint32_t v) { return !(v & 0x7FFFFFFF) || (v & 0x7FFFFFFF) == 0x3F800000; };
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
    if (!same_layout && !resolve(seg, match[i])) {
      borrow(seg, nullptr);
      continue;
    }
    // pass 0: the jump check; 1: the write pass, recorded for borrowing; 2: not recorded.
    auto visit = [&](int pass) {
      for (uint32_t e = 0; e < seg.alu_end - seg.alu_begin; ++e) {
        const uint32_t co = cur.alu_offset[seg.alu_begin + e];
        uint64_t* rec = &lerped[cur.alu_reg[seg.alu_begin + e] - kAluFirst];
        for (uint32_t d = 0; d < cur.alu_count[seg.alu_begin + e]; ++d) {
          const uint32_t po =
              same_layout ? prev.alu_offset[pseg.alu_begin + e] + d
                          : prev_at[cur.alu_reg[seg.alu_begin + e] + d - kAluFirst];
          const uint32_t cb = Load(cur_be, co + d), pb = Load(prev_be, po);
          if (pass == 1) {
            rec[d] = uint64_t(cb) << 32 | cb;
          }
          if (cb == pb) continue;
          const float c = std::bit_cast<float>(cb), p = std::bit_cast<float>(pb);
          if (!std::isfinite(c) || !std::isfinite(p)) continue;
          if (!pass) {
            if (Jumped(c, p)) {
              return false;
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
      return true;
    };
    // A sure pairing is the same draw, so a jump there is fast motion (a
    // strike, a spin) and lerping keeps the model whole; a guessed pairing
    // that jumped is likely another draw: drawn as is.
    const bool jumped = !visit(0);
    if (jumped) {
      ++stats.jumped;
      stats.suspect += copies[i] > 2;  // TEMP DIAGNOSTIC
      if (!sure[i]) {
        borrow(seg, nullptr);
        continue;
      }
      saved.clear();
      for (uint32_t e = seg.alu_begin; e < seg.alu_end; ++e) {
        for (uint32_t d = 0; d < cur.alu_count[e]; ++d) {
          saved.push_back(Load(cur_be, cur.alu_offset[e] + d));
        }
      }
    }
    ++stats.lerped;
    visit(jumped ? 2 : 1);
    if (jumped) {
      borrow(seg, saved.data());
    }
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
  // A sprite switched, not moved: every float word that changed (a texture
  // coordinate picking another atlas cell - a stage name, a digit) moved at
  // least its own extent over the vertices while the others (the position)
  // stayed put. In between would show the cell in between: kept as is.
  std::array<float, 64> lo, hi, moved;
  uint64_t still = float_words;
  lo.fill(std::numeric_limits<float>::infinity());
  hi.fill(-std::numeric_limits<float>::infinity());
  moved.fill(0);
  for (uint32_t v = 0; v < vertices; ++v) {
    for (uint64_t m = float_words; m; m &= m - 1) {
      const uint32_t w = uint32_t(std::countr_zero(m)), i = v * stride_words + w;
      const float c = std::bit_cast<float>(xenos::GpuSwap(cur[i], e));
      const float p = std::bit_cast<float>(xenos::GpuSwap(prev[i], e));
      if (!std::isfinite(c) || !std::isfinite(p)) continue;
      lo[w] = std::min(lo[w], c);
      hi[w] = std::max(hi[w], c);
      if (prev[i] != cur[i]) {
        still &= ~(uint64_t(1) << w);
        moved[w] = std::max(moved[w], std::abs(c - p));
      }
    }
  }
  bool switched = still != 0;
  for (uint64_t m = float_words & ~still; switched && m; m &= m - 1) {
    const uint32_t w = uint32_t(std::countr_zero(m));
    switched = hi[w] > lo[w] && moved[w] >= hi[w] - lo[w];
  }
  if (switched) {
    return false;
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
