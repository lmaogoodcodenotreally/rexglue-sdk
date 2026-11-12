#include <bit>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <rex/graphics/frame_interp.h>
#include <rex/graphics/xenos.h>

using namespace rex::graphics::frame_interp;
namespace xenos = rex::graphics::xenos;

namespace {

// Builds a big-endian PM4 stream the way the guest writes it.
struct Stream {
  std::vector<uint32_t> be;
  void Put(uint32_t v) { be.push_back(std::byteswap(v)); }
  void Type3(uint32_t op, std::vector<uint32_t> payload) {
    Put(0xC0000000u | (uint32_t(payload.size() - 1) << 16) | (op << 8));
    for (uint32_t v : payload) Put(v);
  }
  void Shaders(uint32_t vs, uint32_t ps) {
    Type3(0x27, {vs | 0, 4});  // IM_LOAD vertex
    Type3(0x27, {ps | 1, 4});  // IM_LOAD pixel
  }
  void Constants(uint32_t reg, std::vector<float> values) {
    Put((uint32_t(values.size() - 1) << 16) | reg);  // type-0 register write
    for (float f : values) Put(std::bit_cast<uint32_t>(f));
  }
  void Draw(uint32_t initiator) { Type3(0x36, {initiator}); }  // DRAW_INDX_2
  float Float(size_t i) const { return std::bit_cast<float>(std::byteswap(be[i])); }
};

}  // namespace

TEST_CASE("frame_interp lerps paired constants and resyncs past an added draw") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {10, 10, 10, 10});
  prev.Draw(0x11);
  prev.Shaders(0x3000, 0x4000);
  prev.Constants(0x4010, {10, 10});
  prev.Draw(0x22);

  cur.Shaders(0x5000, 0x6000);  // a draw that did not exist last frame
  cur.Constants(0x4000, {7});
  size_t added = cur.be.size() - 1;
  cur.Draw(0x33);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {12, 14, 16, 18});
  size_t first = cur.be.size() - 4;
  cur.Draw(0x11);
  cur.Shaders(0x3000, 0x4000);
  cur.Constants(0x4010, {12, 14});
  size_t second = cur.be.size() - 2;
  cur.Draw(0x22);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  REQUIRE(pc.segments.size() == 4);  // three draws + tail

  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.matched == 3);  // both old draws + tail; the new draw stays unpaired
  CHECK(s.lerped == 3);
  CHECK(cur.Float(first + 0) == 11.0f);
  CHECK(cur.Float(first + 3) == 14.0f);
  CHECK(cur.Float(second + 0) == 11.0f);
  CHECK(cur.Float(second + 1) == 12.0f);
  CHECK(cur.Float(added) == 7.0f);  // the new draw's constant is untouched
}

TEST_CASE("frame_interp keeps the registers the previous frame never wrote") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {0, 0});
  prev.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4004, {5, 5});  // same draw, different registers written
  cur.Draw(0x11);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.matched == 2);
  CHECK(s.lerped == 2);
  CHECK(cur.Float(cur.be.size() - 3) == 5.0f);
}

TEST_CASE("frame_interp pairs constants by register when the upload split changed") {
  // The game uploads only the ranges that changed since its previous draw.
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {8, 8, 8, 8});
  prev.Draw(0x11);
  prev.Shaders(0x3000, 0x4000);
  prev.Constants(0x4010, {8});  // 0x4000-0x4003 kept from the draw before
  prev.Draw(0x22);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {10, 10});
  cur.Constants(0x4002, {10, 10});
  size_t split = cur.be.size() - 2;
  cur.Draw(0x11);
  cur.Shaders(0x3000, 0x4000);
  cur.Constants(0x4002, {10});
  size_t back = cur.be.size() - 1;
  cur.Constants(0x4010, {10});
  cur.Draw(0x22);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.lerped == 3);
  CHECK(cur.Float(split) == 9.0f);
  CHECK(cur.Float(back) == 9.0f);  // what the previous frame's draw saw
}

TEST_CASE("frame_interp pairs same-shader draws by their textures") {
  auto icon = [](Stream& s, uint32_t texture_base, float x) {
    s.Shaders(0x1000, 0x2000);
    s.Put((1u << 16) | 0x4800);  // fetch slot 0, dwords 0-1
    s.Put(0x2);                  // texture fetch
    s.Put(texture_base);
    s.Constants(0x4000, {x});
    s.Draw(0x11);
  };
  Stream prev, cur;
  icon(prev, 0xA0000000, 8);
  icon(prev, 0xB0000000, 100);
  icon(cur, 0xB0000000, 110);  // the icons swapped draw order
  size_t b = cur.be.size() - 3;
  icon(cur, 0xA0000000, 10);
  size_t a = cur.be.size() - 3;

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(cur.Float(b) == 105.0f);  // B lerps with B, not with A
  CHECK(cur.Float(a) == 9.0f);
}

TEST_CASE("frame_interp pairs the k-th draw of a signature with the k-th") {
  // One mesh drawn in two passes, with another draw culled between them.
  auto mesh = [](Stream& s, float x) {
    s.Shaders(0x1000, 0x2000);
    s.Constants(0x4000, {x});
    s.Draw(0x11);
  };
  Stream prev, cur;
  mesh(prev, 8);
  prev.Shaders(0x3000, 0x4000);
  prev.Draw(0x22);
  mesh(prev, 100);
  cur.Shaders(0x3000, 0x4000);  // re-sorted ahead of the first pass
  cur.Draw(0x22);
  mesh(cur, 10);
  size_t first = cur.be.size() - 3;
  mesh(cur, 110);
  size_t second = cur.be.size() - 3;

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.matched == 4);
  CHECK(s.jumped == 0);
  CHECK(cur.Float(first) == 9.0f);
  CHECK(cur.Float(second) == 105.0f);
}

TEST_CASE("frame_interp pairs copies of one mesh by their constants") {
  // Two static props of one mesh, re-sorted by depth: each keeps its place.
  auto prop = [](Stream& s, float x) {
    s.Shaders(0x1000, 0x2000);
    s.Constants(0x4000, {x, 1});
    s.Draw(0x11);
  };
  Stream prev, cur;
  prop(prev, 10);
  prop(prev, 20);
  prop(cur, 20);
  size_t first = cur.be.size() - 4;
  prop(cur, 10);
  size_t second = cur.be.size() - 4;

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(cur.Float(first) == 20.0f);  // not 15: halfway to the other prop
  CHECK(cur.Float(second) == 10.0f);
}

TEST_CASE("frame_interp keeps the order of copies whose constants barely differ") {
  // UI text quads: one shader and texture, told apart by their vertex data.
  auto quad = [](Stream& s, float x) {
    s.Shaders(0x1000, 0x2000);
    s.Constants(0x4000, {x});
    s.Draw(0x11);
  };
  Stream prev, cur;
  quad(prev, 1.0f);
  quad(prev, 1.25f);
  quad(cur, 1.25f);
  size_t first = cur.be.size() - 3;
  quad(cur, 1.0f);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f, &pairs);
  REQUIRE(pairs.size() >= 2);
  CHECK(pairs[0].second == pp.segments[0].packet);  // the first with the first
  CHECK(cur.Float(first) == 1.125f);
}

TEST_CASE("frame_interp leaves a guessed pairing whose constants jumped") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {0, 5});  // light A
  prev.Draw(0x11);
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {300, 5});  // light C, gone this frame
  prev.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {100, 5});  // light B: not motion
  size_t x = cur.be.size() - 2;
  cur.Draw(0x11);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.jumped == 1);
  CHECK(cur.Float(x) == 100.0f);
}

TEST_CASE("frame_interp draws as is a copy paired onto another mesh") {
  // MK seagulls: c1.x is the mesh's vertex count the shader divides the vertex
  // index by; a copy whose nearest pair is another mesh must not show 124.
  Stream prev, cur;
  for (float pos : {0.5f, 5.5f}) {
    prev.Shaders(0x1000, 0x2000);
    prev.Constants(0x4000, {pos, pos, pos, 1, 122, 0, 0, 0});
    prev.Draw(0x11);
  }
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {0.7f, 0.7f, 0.7f, 1, 126, 0, 0, 0});  // switched mesh
  const size_t x = cur.be.size() - 8;
  cur.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {5.7f, 5.7f, 5.7f, 1, 122, 0, 0, 0});
  const size_t y = cur.be.size() - 8;
  cur.Draw(0x11);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f, &pairs);
  CHECK(s.jumped == 1);
  CHECK(cur.Float(x + 4) == 126.0f);
  CHECK(cur.Float(x + 0) == 0.7f);  // the whole draw as recorded
  CHECK(std::abs(cur.Float(y + 0) - 5.6f) < 1e-5f);  // the other copy lerps
  CHECK(pairs.size() == 2);         // that copy and the tail: x's vertex data is not lerped
}

TEST_CASE("frame_interp lerps whole numbers that move with their row") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {100, 40, 0, 1});  // a quad's pixel position
  prev.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {108, 44, 0, 1});
  size_t x = cur.be.size() - 4;
  cur.Draw(0x11);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.jumped == 0);
  CHECK(cur.Float(x + 0) == 104.0f);
  CHECK(cur.Float(x + 1) == 42.0f);
}

TEST_CASE("frame_interp gives a new draw the lerped camera it shares") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {10, 20, 30, 40});  // camera row
  prev.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {12, 22, 32, 42});
  cur.Draw(0x11);
  cur.Shaders(0x5000, 0x6000);  // a prop come into view: no pair
  cur.Constants(0x4000, {12, 22, 32, 42, 5, 6, 7, 8, 1, 0, 0, 0});  // camera, own, plain
  size_t x = cur.be.size() - 12;
  cur.Draw(0x33);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(cur.Float(x + 0) == 11.0f);
  CHECK(cur.Float(x + 3) == 41.0f);
  CHECK(cur.Float(x + 4) == 5.0f);  // its own constants stay
  CHECK(cur.Float(x + 8) == 1.0f);
}

TEST_CASE("frame_interp pairs many re-sorted copies by their nearest constants") {
  // 40 tiles of one mesh, re-sorted, plus one come into view: the searched
  // path for many copies left must still find each tile's own.
  Stream prev, cur;
  for (int k = 0; k < 40; ++k) {
    prev.Shaders(0x1000, 0x2000);
    prev.Constants(0x4000, {float(k * 8), 5, 1});
    prev.Draw(0x11);
  }
  std::vector<size_t> at(41);
  for (int i = 0; i < 41; ++i) {
    const int k = i == 40 ? 100 : (i * 7) % 40;
    cur.Shaders(0x1000, 0x2000);
    cur.Constants(0x4000, {float(k * 8) + 0.5f, 5, 1});
    at[i] = cur.be.size() - 3;
    cur.Draw(0x11);
  }
  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  for (int i = 0; i < 40; ++i) {
    CHECK(cur.Float(at[i]) == float((i * 7) % 40 * 8) + 0.25f);
  }
  CHECK(cur.Float(at[40]) == 800.5f);  // no copy left for it
}

TEST_CASE("frame_interp follows a light switched to another source") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {2, 0.9f, -2.9f, 1});  // light B
  prev.Draw(0x11);
  prev.Shaders(0x3000, 0x4000);
  prev.Constants(0x4000, {2, 0.8f, -0.2f, 1});  // light A
  prev.Draw(0x22);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {2, 0.9f, -3.0f, 1});
  cur.Draw(0x11);
  cur.Shaders(0x3000, 0x4000);
  cur.Constants(0x4000, {2, 0.9f, -3.0f, 1});  // now lit by B: a jump, not motion
  size_t x = cur.be.size() - 4;
  cur.Draw(0x22);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.jumped == 0);  // a light that drew another draw cleanly: no jump
  CHECK(cur.Float(x + 1) == 0.9f);  // B's in-between, not halfway from A
  CHECK(cur.Float(x + 2) == -2.95f);
}

TEST_CASE("frame_interp lerps a sure pairing through fast motion") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {0.9f, -0.4f});  // a rotation row spinning fast
  prev.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {-0.9f, 3.0f});
  size_t x = cur.be.size() - 2;
  cur.Draw(0x11);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.jumped == 1);  // still counted, for cut detection
  CHECK(s.lerped == 2);
  CHECK(cur.Float(x) == 0.0f);
}

TEST_CASE("frame_interp draws as is a copy whose sure pairing jumped") {
  // Two props of one mesh, as many in both frames, both far from where either
  // was: the pairing is sure by count, but a jump means another object - in
  // between it would blink halfway to the other copy.
  auto prop = [](Stream& s, float x) {
    s.Shaders(0x1000, 0x2000);
    s.Constants(0x4000, {x, 1});
    s.Draw(0x11);
  };
  Stream prev, cur;
  prop(prev, 0);
  prop(prev, 100);
  prop(cur, 50);
  size_t first = cur.be.size() - 4;
  prop(cur, 150);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.jumped == 1);  // 100 -> 150 is motion
  CHECK(cur.Float(first) == 50.0f);  // not 25
}

PackedWords Packed(uint32_t word, xenos::VertexFormat format) {
  PackedWords packed;
  packed.words = uint64_t(1) << word;
  packed.format[word] = uint8_t(format);
  return packed;
}

TEST_CASE("frame_interp turns a rigid matrix that turned far in one tick") {
  // A fist's world matrix (3 basis rows, translation row) turning 90 degrees
  // about z in one tick of a strike: halfway it is turned 45 degrees, at full
  // size (a per-element lerp would shrink it to 0.71), and moved halfway.
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 10, 0, 0, 1});
  prev.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {0, 2, 0, 0, -2, 0, 0, 0, 0, 0, 2, 0, 12, 0, 0, 1});
  const size_t x = cur.be.size() - 16;
  cur.Draw(0x11);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  const float h = 2 * std::sqrt(0.5f);
  CHECK(std::abs(cur.Float(x + 0) - h) < 1e-4f);
  CHECK(std::abs(cur.Float(x + 1) - h) < 1e-4f);
  CHECK(std::abs(cur.Float(x + 4) + h) < 1e-4f);
  CHECK(std::abs(cur.Float(x + 5) - h) < 1e-4f);
  CHECK(std::abs(cur.Float(x + 10) - 2) < 1e-4f);
  CHECK(cur.Float(x + 12) == 11.0f);
}

TEST_CASE("frame_interp turns a mirrored rigid matrix") {
  // A view-space matrix with z flipped (determinant -1) turning 90 degrees.
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0});
  prev.Draw(0x11);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {0, 1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0});
  const size_t x = cur.be.size() - 12;
  cur.Draw(0x11);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  const float h = std::sqrt(0.5f);
  CHECK(std::abs(cur.Float(x + 0) - h) < 1e-4f);
  CHECK(std::abs(cur.Float(x + 1) - h) < 1e-4f);
  CHECK(std::abs(cur.Float(x + 4) + h) < 1e-4f);
  CHECK(std::abs(cur.Float(x + 10) + 1) < 1e-4f);
}

TEST_CASE("frame_interp lerps the float words of vertex data in guest byte order") {
  // Two 3-word vertices: a big-endian (k8in32 = 2) float pair and an unlisted
  // word (a packed color) that must match between the frames.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto f = [](uint32_t w) { return std::bit_cast<float>(std::byteswap(w)); };
  const uint32_t prev[] = {be(1.5f), be(10), 0x33333333, be(4), be(1.5f), 0x44444444};
  uint32_t cur[] = {be(2), be(20), 0x33333333, be(4), be(2), 0x44444444};
  REQUIRE(LerpVertexFloats(prev, cur, 2, 3, 0b011, {}, 2, 0.5f));
  REQUIRE(f(cur[0]) == 1.75f);
  REQUIRE(f(cur[1]) == 15.0f);
  REQUIRE(cur[2] == 0x33333333);
  REQUIRE(f(cur[3]) == 4.0f);
  REQUIRE(f(cur[4]) == 1.75f);

  uint32_t recolored[] = {be(2), be(20), 0x55555555, be(4), be(2), 0x44444444};
  REQUIRE(!LerpVertexFloats(prev, recolored, 2, 3, 0b011, {}, 2, 0.5f));  // another vertex
  REQUIRE(f(recolored[0]) == 2.0f);
  uint32_t jumped[] = {be(100), be(20), 0x33333333, be(4), be(2), 0x44444444};
  REQUIRE(!LerpVertexFloats(prev, jumped, 2, 3, 0b011, {}, 2, 0.5f));  // 1.5 -> 100
  REQUIRE(f(jumped[0]) == 100.0f);
}

TEST_CASE("frame_interp lerps a fading vertex color per byte") {
  // A text quad's 8:8:8:8 color fading out while it moves: the color lerps per
  // byte (as stored: lanes do not depend on the swap), the position as floats.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto f = [](uint32_t w) { return std::bit_cast<float>(std::byteswap(w)); };
  const uint32_t prev[] = {be(1), be(2), 0xEBEEFF20, be(5), be(2), 0xEBEEFF20};
  uint32_t cur[] = {be(2), be(2), 0xEBEEFF10, be(6), be(2), 0xEBEEFF10};
  REQUIRE(LerpVertexFloats(prev, cur, 2, 3, 0b011, Packed(2, xenos::VertexFormat::k_8_8_8_8), 2, 0.5f));
  CHECK(f(cur[0]) == 1.5f);
  CHECK(cur[2] == 0xEBEEFF18);
  CHECK(f(cur[3]) == 5.5f);
  // A fade alone, the quad standing still, is not a sprite switch.
  uint32_t still[] = {be(1), be(2), 0xEBEEFF10, be(5), be(2), 0xEBEEFF10};
  REQUIRE(LerpVertexFloats(prev, still, 2, 3, 0b011, Packed(2, xenos::VertexFormat::k_8_8_8_8), 2, 0.5f));
  CHECK(still[5] == 0xEBEEFF18);
}

TEST_CASE("frame_interp lerps a skinned vertex's packed normal per lane") {
  // A CPU-skinned limb: position floats plus a signed 10:11:11 normal that
  // turns every tick. Lanes (from the low bit, after the swap): 11, 11, 10.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto normal = [](int32_t x, int32_t y, int32_t z) {
    return std::byteswap((uint32_t(x) & 0x7FF) | (uint32_t(y) & 0x7FF) << 11 |
                         (uint32_t(z) & 0x3FF) << 22);
  };
  PackedWords packed = Packed(1, xenos::VertexFormat::k_10_11_11);
  packed.is_signed = 0b10;
  const uint32_t prev[] = {be(1), normal(100, -200, 50)};
  uint32_t cur[] = {be(3), normal(200, -100, -50)};
  REQUIRE(LerpVertexFloats(prev, cur, 1, 2, 0b01, packed, 2, 0.5f));
  CHECK(cur[1] == normal(150, -150, 0));
  // Unlisted, the changed word is another vertex.
  uint32_t other[] = {be(3), normal(200, -100, -50)};
  CHECK(!LerpVertexFloats(prev, other, 1, 2, 0b01, {}, 2, 0.5f));
}

TEST_CASE("frame_interp turns tumbling debris instances") {
  // Per-instance 3x4 matrices (translation in w), 16 floats each: one piece
  // spins 120 degrees in a tick (its entries swing past a jump), another's
  // slot is cleared. The piece turns 60 degrees; the cleared slot stays clear.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto f = [](uint32_t w) { return std::bit_cast<float>(std::byteswap(w)); };
  const float c = 1.5f * -0.5f, s = 1.5f * std::sqrt(0.75f);
  const uint32_t prev[32] = {be(1.5f), 0, 0, be(3), 0, be(1.5f), 0, be(4), 0, 0, be(1.5f), be(5),
                             0, 0, 0, be(1),
                             be(1), 0, 0, be(7), 0, be(1), 0, be(8), 0, 0, be(1), be(9),
                             0, 0, 0, be(1)};
  uint32_t cur[32] = {be(c), be(-s), 0, be(5), be(s), be(c), 0, be(4), 0, 0, be(1.5f), be(5),
                      0, 0, 0, be(1)};
  REQUIRE(LerpVertexFloats(prev, cur, 2, 16, 0xFFFF, {}, 2, 0.5f));
  CHECK(std::abs(f(cur[0]) - 0.75f) < 1e-4f);
  CHECK(std::abs(f(cur[1]) + 1.5f * std::sqrt(0.75f)) < 1e-4f);
  CHECK(std::abs(f(cur[10]) - 1.5f) < 1e-4f);
  CHECK(f(cur[3]) == 4.0f);
  CHECK(cur[16 + 3] == 0);
}

TEST_CASE("frame_interp keeps a sprite that switched atlas cells") {
  // A quad (x, y, u, v) standing still while v picks another stage name: the
  // in-between would sample the name between the two. A scroll by less than
  // the quad's own extent is motion and lerps.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto f = [](uint32_t w) { return std::bit_cast<float>(std::byteswap(w)); };
  auto quad = [&](float v0, std::vector<uint32_t>& out) {
    const float xy[4][2] = {{0, 0}, {100, 0}, {100, 20}, {0, 20}};
    for (int k = 0; k < 4; ++k) {
      out.insert(out.end(), {be(xy[k][0]), be(xy[k][1]), be(k == 1 || k == 2 ? 1.0f : 0.0f),
                             be(v0 + (k >= 2 ? 0.1f : 0.0f))});
    }
  };
  std::vector<uint32_t> prev, name, scroll;
  quad(0.0f, prev);
  quad(0.2f, name);
  REQUIRE(!LerpVertexFloats(prev.data(), name.data(), 4, 4, 0b1111, {}, 2, 0.5f));
  CHECK(f(name[3]) == 0.2f);
  quad(0.02f, scroll);
  REQUIRE(LerpVertexFloats(prev.data(), scroll.data(), 4, 4, 0b1111, {}, 2, 0.5f));
  CHECK(f(scroll[3]) == 0.01f);
}

TEST_CASE("frame_interp keeps a glyph that slid and switched letters") {
  // A text quad (x, y, z, u, v; u v a 2-float attribute) moving 8 px (under
  // its width) while u picks another letter: the in-between would sample the
  // atlas between the two letters. Sliding with its letter kept lerps.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto f = [](uint32_t w) { return std::bit_cast<float>(std::byteswap(w)); };
  auto quad = [&](float x0, float u0, std::vector<uint32_t>& out) {
    for (int k = 0; k < 4; ++k) {
      const float dx = k == 1 || k == 2 ? 1.0f : 0.0f, dy = k >= 2 ? 1.0f : 0.0f;
      out.insert(out.end(), {be(x0 + 16 * dx), be(611 + 32 * dy), be(0.0f), be(u0 + 0.0625f * dx),
                             be(0.125f + 0.125f * dy)});
    }
  };
  PackedWords uv;
  uv.uv_words = 0b11000;
  std::vector<uint32_t> prev, letter, slide;
  quad(300, 0.25f, prev);
  quad(308, 0.5f, letter);
  REQUIRE(!LerpVertexFloats(prev.data(), letter.data(), 4, 5, 0b11111, uv, 2, 0.5f));
  CHECK(f(letter[0]) == 308.0f);
  quad(308, 0.25f, slide);
  REQUIRE(LerpVertexFloats(prev.data(), slide.data(), 4, 5, 0b11111, uv, 2, 0.5f));
  CHECK(f(slide[0]) == 304.0f);
  // In a string the extent of u spans the atlas: one glyph of two switching
  // letters is caught only per glyph (a quad list's 4 vertices).
  std::vector<uint32_t> two, two_next, whole;
  quad(300, 0.0f, two);
  quad(316, 0.5f, two);
  quad(304, 0.0f, two_next);
  quad(320, 0.25f, two_next);
  whole = two_next;
  CHECK(LerpVertexFloats(two.data(), whole.data(), 8, 5, 0b11111, uv, 2, 0.5f));
  REQUIRE(!LerpVertexFloats(two.data(), two_next.data(), 8, 5, 0b11111, uv, 2, 0.5f, 4));
  CHECK(f(two_next[0]) == 304.0f);
}

TEST_CASE("frame_interp lerps a particle that outran its own size") {
  // One quad of a particle buffer moved by more than its width while its
  // texture coordinate stayed: motion, not a sprite switch.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto f = [](uint32_t w) { return std::bit_cast<float>(std::byteswap(w)); };
  std::vector<uint32_t> prev, cur;
  for (int k = 0; k < 4; ++k) {
    const float x = k == 1 || k == 2 ? 1.0f : 0.0f, y = k >= 2 ? 1.0f : 0.0f;
    prev.insert(prev.end(), {be(x), be(y), be(0.5f + x * 0.1f)});
    cur.insert(cur.end(), {be(x + 1.5f), be(y + 0.25f), be(0.5f + x * 0.1f)});
  }
  REQUIRE(LerpVertexFloats(prev.data(), cur.data(), 4, 3, 0b111, {}, 2, 0.5f));
  CHECK(f(cur[0]) == 0.75f);
}

TEST_CASE("frame_interp swaps copies that traded the texture they set") {
  // Two rickshaws of one mesh: the first drawn binds the texture, the second
  // finds it bound. Drawn in the other order, each takes the other's
  // signature; paired by it they would meet halfway.
  auto rickshaw = [](Stream& s, bool binds, float x) {
    s.Shaders(0x1000, 0x2000);
    if (binds) {
      s.Put((1u << 16) | 0x4800);
      s.Put(0x2);
      s.Put(0xA0000000);
    }
    s.Constants(0x4000, {x, 0, 0, 1});
    s.Draw(0x11);
  };
  Stream prev, cur;
  rickshaw(prev, true, 10);
  rickshaw(prev, false, 13);
  rickshaw(cur, true, 13.5f);
  const size_t b = cur.be.size() - 6;
  rickshaw(cur, false, 10.5f);
  const size_t a = cur.be.size() - 6;

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(cur.Float(b) == 13.25f);  // each with itself
  CHECK(cur.Float(a) == 10.25f);
}

TEST_CASE("frame_interp follows two faces that traded their expression textures") {
  // Two fighters' faces of one mesh, each binding its expression texture; the
  // expressions changed so that each now shows the other's last texture, and
  // the second face's constants arrive split differently.
  auto face = [](Stream& s, uint32_t texture, float x, bool split) {
    s.Shaders(0x1000, 0x2000);
    s.Put((1u << 16) | 0x4800);
    s.Put(0x2);
    s.Put(texture);
    if (split) {
      s.Constants(0x4000, {x, 0});
      s.Constants(0x4002, {0, 1});
    } else {
      s.Constants(0x4000, {x, 0, 0, 1});
    }
    s.Draw(0x11);
    return s.be.size() - (split ? 7 : 6);
  };
  Stream prev, cur;
  face(prev, 0xA0000000, 10, false);
  face(prev, 0xB0000000, 13, false);
  const size_t a = face(cur, 0xB0000000, 10.5f, false);
  const size_t b = face(cur, 0xA0000000, 13.5f, true);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(cur.Float(a) == 10.25f);  // each with itself
  CHECK(cur.Float(b) == 13.25f);
}

TEST_CASE("frame_interp pairs a mesh drawn from moved index data") {
  Stream prev, cur;
  prev.Shaders(0x1000, 0x2000);
  prev.Constants(0x4000, {10, 0, 0, 1});
  prev.Type3(0x22, {0, 0x00300004, 0x100000, 0x30});  // DRAW_INDX: viz, initiator, base, size
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {11, 0, 0, 1});
  const size_t x = cur.be.size() - 4;
  cur.Type3(0x22, {0, 0x00240004, 0x101000, 0x24});  // ring-allocated indices, fewer this tick

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.lerped == 2);
  CHECK(cur.Float(x) == 10.5f);
}

TEST_CASE("frame_interp lerps particle quads whose list shifted") {
  // Quads of 4 vertices (x, y, z, u, v); a particle born at the front this
  // tick shifts the others one place on.
  auto quads = [](std::vector<float> xs) {
    std::vector<uint32_t> words;
    for (float x : xs) {
      for (int v = 0; v < 4; ++v) {
        for (float f : {x + (v & 1), 1.0f, 2.0f, float(v >> 1), float(v & 1)}) {
          words.push_back(std::bit_cast<uint32_t>(f));
        }
      }
    }
    return words;
  };
  const std::vector<uint32_t> prev = quads({0, 10, 20});
  std::vector<uint32_t> cur = quads({100, 0.4f, 10.4f, 20.4f});
  CHECK(!LerpVertexFloats(prev.data(), cur.data(), 12, 5, 0x1F, {}, 0, 0.5f));
  CHECK(LerpPrimitives(prev.data(), 12, cur.data(), 16, 4, 5, 0x1F, {}, 0, 0.5f) == 3);
  auto x = [&](int quad) { return std::bit_cast<float>(cur[quad * 20]); };
  CHECK(x(0) == 100.0f);  // the newborn has no previous place
  CHECK(x(1) == 0.2f);
  CHECK(x(2) == 10.2f);
  CHECK(x(3) == 20.2f);
}
