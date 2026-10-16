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
#include <utility>
#include <vector>

namespace rex::graphics::frame_interp {

struct ParsedFrame {
  struct Segment {
    uint64_t signature;  // `draw` + the texture bases the segment sets
    uint64_t draw;       // shaders + draw initiator (+ index base) of the closing draw
    uint64_t layout;     // the headers of the ALU-constant packets in the segment
    uint32_t alu_begin, alu_end;  // range in alu_offset / alu_count
    uint32_t packet;     // dword offset of the closing draw packet (the stream size for the tail)
  };
  std::vector<Segment> segments;  // one per draw, plus the tail after the last draw
  std::vector<uint32_t> alu_offset;  // dword offset of the first ALU constant written
  std::vector<uint32_t> alu_count;   // ALU constant dwords written from there
  std::vector<uint32_t> alu_reg;     // the register of the first one
};

struct Stats {
  uint32_t segments = 0;
  uint32_t matched = 0;  // paired with a previous-frame segment of the same signature
  uint32_t lerped = 0;   // ...and its constants were lerped
  uint32_t jumped = 0;   // ...where a constant jumped (fast motion, a cut or a mispairing)
  uint32_t suspect = 0;  // TEMP DIAGNOSTIC: ...a draw with more than 2 copies
};

void Parse(const uint32_t* stream_be, size_t dword_count, ParsedFrame& out);

// Moves every ALU constant written by a paired segment of `cur` from its value
// toward the previous frame's value: result = prev + (cur - prev) * t. t = 1
// leaves `cur` as is, t = 0.5 is the midpoint. Non-finite values are kept.
// Draws pair by signature, then by draw alone; constants by position when both
// frames wrote the same layout, else by register. A constant that moved by
// more than half its magnitude (floor 2) counts the segment as jumped (camera
// cuts are told by those); where the pairing was a guess - the signature's
// count changed - the segment then keeps `cur`, as two different draws.
// A segment kept as `cur` (unpaired, or such a guess), or a sure pairing that
// jumped, takes the lerped value of each whole vec4 it writes bit-equal to
// what the last cleanly lerped segment wrote to that register (the shared
// camera, a light switched to another source), unless it is all 0 / +-1.
// `lerped_pairs`, if given, receives (cur packet, prev packet) of every lerped
// segment, in stream order.
Stats Interpolate(const uint32_t* prev_be, const ParsedFrame& prev, uint32_t* cur_be,
                  const ParsedFrame& cur, float t,
                  std::vector<std::pair<uint32_t, uint32_t>>* lerped_pairs = nullptr);

// Vertex data the CPU rewrites every frame (CPU-skinned meshes) moves with the
// constants: lerps, in place in `cur`, the float32 words of each vertex (bit k
// of `float_words` = word k of the `stride_words`-word vertex) from `prev`, as
// Interpolate does. Words are as stored in guest memory, `endian` is the vertex
// fetch constant's xenos::Endian. Non-finite values are kept. Returns false,
// leaving `cur` as is, when the data does not correspond: a word outside
// `float_words` (a packed color, say) differs, or a float jumped as
// Interpolate's constants would, or a sprite switched rather than moved (every
// float that changed jumped by its own whole extent while another word stayed:
// a texture coordinate picking another atlas cell under a quad standing still).
bool LerpVertexFloats(const uint32_t* prev, uint32_t* cur, uint32_t vertices,
                      uint32_t stride_words, uint64_t float_words, uint32_t endian, float t);

}  // namespace rex::graphics::frame_interp
