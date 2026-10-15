#pragma once
/**
 ******************************************************************************
 * ReXGlue : Xbox 360 runtime                                               *
 ******************************************************************************
 *
 * Emulator-side FMV skip support (retail DOA4: intro movie skippable like
 * the disclaimer).
 *
 * The guest never polls input while an FMV plays (user-verified: the
 * disclaimer accepts A/Start, the movie ignores every key), so the skip
 * cannot come from the guest. Input handlers call Request() on a FRESH key
 * or mouse-button press while the game window has focus (edge-gated:
 * auto-repeat and keys held over from the disclaimer do not count - R45
 * showed those leaking across the Reset and black-screening the movie);
 * the first XFile read of an .sfd movie while the flag is set forces the
 * guest MWPLY player's status dword into END (retail handle global
 * dword_833F8070, status at +4 - see xfile.cpp ReadInternal), and the game
 * performs its own movie-finished transition (stop -> fade -> next scene).
 * Faking a zero-byte EOF does NOT work: SOFDEC never treats it as
 * end-of-stream (R45: black movie hung for the whole 90 s run). Opening a
 * new .sfd file calls Reset() so a press aimed at the disclaimer (or a
 * previous movie) never leaks into the next playback.
 *
 * All users of this flag live inside rexruntime (mnk_input_driver.cpp and
 * xfile.cpp are both linked into rexruntimerd.dll via the rexinput object
 * library), so a header-local function-static is a single shared instance.
 *
 * The movie_skip cvar (defined in xfile.cpp) gates serving the EOF.
 */

#include <atomic>

namespace rex::system::movie_skip {

// Set by input handlers on user key/button presses; consumed by XFile reads.
inline std::atomic<bool>& requested() {
  static std::atomic<bool> requested{false};
  return requested;
}

// Diagnostic: which handler raised the current request. Plain pointer store;
// only read for logging.
inline const char*& source() {
  static const char* source{"none"};
  return source;
}

// Remember that the user asked to end the current (or next) movie. Callers
// must edge-gate (fresh press only) so a key held across the Reset at movie
// open - or SDL key auto-repeat - cannot leak a skip (R45: EOF served on the
// first read with no press during the movie).
inline void Request(const char* from) {
  source() = from;
  requested().store(true, std::memory_order_release);
}

// Arm a fresh window when a new movie file opens.
inline void Reset() {
  source() = "none";
  requested().store(false, std::memory_order_release);
}

inline bool IsRequested() { return requested().load(std::memory_order_acquire); }

inline const char* Source() { return source(); }

}  // namespace rex::system::movie_skip
