#include <bit>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <rex/graphics/frame_interp.h>

using namespace rex::graphics::frame_interp;

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

TEST_CASE("frame_interp leaves segments whose constant layout changed") {
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
  CHECK(s.lerped == 1);  // only the empty tail
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
  CHECK(s.jumped == 1);
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

TEST_CASE("frame_interp lerps the float words of vertex data in guest byte order") {
  // Two 3-word vertices: a big-endian (k8in32 = 2) float pair and an unlisted
  // word (a packed color) that must match between the frames.
  auto be = [](float f) { return std::byteswap(std::bit_cast<uint32_t>(f)); };
  auto f = [](uint32_t w) { return std::bit_cast<float>(std::byteswap(w)); };
  const uint32_t prev[] = {be(1.5f), be(10), 0x33333333, be(4), be(1.5f), 0x44444444};
  uint32_t cur[] = {be(2), be(20), 0x33333333, be(4), be(2), 0x44444444};
  REQUIRE(LerpVertexFloats(prev, cur, 2, 3, 0b011, 2, 0.5f));
  REQUIRE(f(cur[0]) == 1.75f);
  REQUIRE(f(cur[1]) == 15.0f);
  REQUIRE(cur[2] == 0x33333333);
  REQUIRE(f(cur[3]) == 4.0f);
  REQUIRE(f(cur[4]) == 1.75f);

  uint32_t recolored[] = {be(2), be(20), 0x55555555, be(4), be(2), 0x44444444};
  REQUIRE(!LerpVertexFloats(prev, recolored, 2, 3, 0b011, 2, 0.5f));  // another vertex
  REQUIRE(f(recolored[0]) == 2.0f);
  uint32_t jumped[] = {be(100), be(20), 0x33333333, be(4), be(2), 0x44444444};
  REQUIRE(!LerpVertexFloats(prev, jumped, 2, 3, 0b011, 2, 0.5f));  // 1.5 -> 100
  REQUIRE(f(jumped[0]) == 100.0f);
}
