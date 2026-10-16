#pragma once
/**
 ******************************************************************************
 * ReXGlue : Xbox 360 runtime                                               *
 ******************************************************************************
 *
 * Frame interpolation by command-stream replay (frame_interp cvar).
 *
 * The guest keeps its exact 60 Hz tick: one swap per vblank, untouched. The
 * command processor records every packet a frame executes (indirect buffers
 * flattened), and at the swap of frame N it replays N's stream once more with
 * the shader ALU constants (registers 0x4000-0x47FF: camera, world and bone
 * matrices) moved halfway back toward frame N-1, presents that replay as the
 * in-between frame, and presents the real frame N half a tick later.
 *
 * This file is the pure part: parse a recorded stream into per-draw segments
 * and lerp the constant payloads of segments that pair up with the previous
 * frame. Streams are stored exactly as the guest wrote them (big-endian).
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rex::graphics::frame_interp {

struct ParsedFrame {
  struct Segment {
    uint64_t signature;  // shaders + draw initiator (+ index base) of the closing draw
    uint64_t layout;     // the headers of the ALU-constant packets in the segment
    uint32_t alu_begin, alu_end;  // range in alu_offset / alu_count
  };
  std::vector<Segment> segments;  // one per draw, plus the tail after the last draw
  std::vector<uint32_t> alu_offset;  // dword offset of the first ALU constant written
  std::vector<uint32_t> alu_count;   // ALU constant dwords written from there
};

struct Stats {
  uint32_t segments = 0;
  uint32_t matched = 0;  // paired with a previous-frame segment of the same signature
  uint32_t lerped = 0;   // ...and the same constant layout, so its constants were lerped
};

void Parse(const uint32_t* stream_be, size_t dword_count, ParsedFrame& out);

// Moves every ALU constant written by a paired segment of `cur` from its value
// toward the previous frame's value: result = prev + (cur - prev) * t. t = 1
// leaves `cur` as is, t = 0.5 is the midpoint. Non-finite values are kept.
Stats Interpolate(const uint32_t* prev_be, const ParsedFrame& prev, uint32_t* cur_be,
                  const ParsedFrame& cur, float t);

}  // namespace rex::graphics::frame_interp
