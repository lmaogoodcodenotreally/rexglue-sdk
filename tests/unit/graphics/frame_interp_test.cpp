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
  prev.Constants(0x4000, {0, 0, 0, 0});
  prev.Draw(0x11);
  prev.Shaders(0x3000, 0x4000);
  prev.Constants(0x4010, {10, 10});
  prev.Draw(0x22);

  cur.Shaders(0x5000, 0x6000);  // a draw that did not exist last frame
  cur.Constants(0x4000, {7});
  size_t added = cur.be.size() - 1;
  cur.Draw(0x33);
  cur.Shaders(0x1000, 0x2000);
  cur.Constants(0x4000, {2, 4, 6, 8});
  size_t first = cur.be.size() - 4;
  cur.Draw(0x11);
  cur.Shaders(0x3000, 0x4000);
  cur.Constants(0x4010, {20, 30});
  size_t second = cur.be.size() - 2;
  cur.Draw(0x22);

  ParsedFrame pp, pc;
  Parse(prev.be.data(), prev.be.size(), pp);
  Parse(cur.be.data(), cur.be.size(), pc);
  REQUIRE(pc.segments.size() == 4);  // three draws + tail

  Stats s = Interpolate(prev.be.data(), pp, cur.be.data(), pc, 0.5f);
  CHECK(s.matched == 3);  // both old draws + tail; the new draw stays unpaired
  CHECK(s.lerped == 3);
  CHECK(cur.Float(first + 0) == 1.0f);
  CHECK(cur.Float(first + 3) == 4.0f);
  CHECK(cur.Float(second + 0) == 15.0f);
  CHECK(cur.Float(second + 1) == 20.0f);
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
  icon(prev, 0xA0000000, 0);
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
  CHECK(cur.Float(a) == 10.0f);   // A sits behind the resync point: drawn as is
}
