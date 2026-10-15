/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <rex/input/input_driver.h>

#include <SDL3/SDL.h>

#define HID_SDL_THUMB_THRES 0x4E00
#define HID_SDL_TRIGG_THRES 0x1F
#define HID_SDL_REPEAT_DELAY 400
#define HID_SDL_REPEAT_RATE 100

namespace rex::input::sdl {

class SDLInputDriver final : public InputDriver, public rex::ui::WindowListener {
 public:
  explicit SDLInputDriver(rex::ui::Window* window, size_t window_z_order);
  ~SDLInputDriver() override;

  X_STATUS Setup() override;

  void EnumerateDevices(std::vector<DeviceInfo>& out) override;
  X_RESULT GetDeviceState(DeviceId id, X_INPUT_STATE* out_state) override;
  X_RESULT GetDeviceCapabilities(DeviceId id, uint32_t flags,
                                 X_INPUT_CAPABILITIES* out_caps) override;
  X_RESULT SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION* vibration) override;
  X_RESULT GetDeviceKeystroke(DeviceId id, uint32_t flags,
                              X_INPUT_KEYSTROKE* out_keystroke) override;
  void OnWindowAvailable(rex::ui::Window* window) override;

 private:
  enum class RepeatState {
    Idle,       // no buttons pressed or repeating has ended
    Waiting,    // a button is held and the delay is awaited
    Repeating,  // actively repeating at a rate
  };
  struct KeystrokeState {
    uint64_t buttons;
    RepeatState repeat_state;
    // the button number that was pressed last:
    uint8_t repeat_butt_idx;
    // the last time (ms) a down (and/or repeat) event for that button was send:
    uint32_t repeat_time;
  };

  struct ControllerState {
    SDL_Gamepad* sdl;
    X_INPUT_CAPABILITIES caps;
    X_INPUT_STATE state;
    bool state_changed;
    bool is_active;
    DeviceId id;
    // Captured once at connect: EnumerateDevices runs on every guest XInput
    // query, and SDL's name/GUID getters take SDL's joystick lock, which SDL
    // holds for ~80 ms while Windows processes a pad arriving or leaving.
    std::string name;
    std::string guid;
    // Cached for the same reason: matching SDL events through
    // SDL_GetJoystickID would take the joystick lock on the guest thread.
    SDL_JoystickID instance_id;
    // Latest guest vibration, sent to SDL by the worker (rumble can block
    // ~100 ms while a pad connects or disconnects).
    uint16_t rumble_left;
    uint16_t rumble_right;
    bool rumble_dirty;
    // What the worker last sent. A pad connects with its motors off, so the
    // zero-initialized values make the guest's repeated (0,0) a no-op instead
    // of Bluetooth traffic that can stall the whole system's presents.
    uint16_t rumble_sent_left;
    uint16_t rumble_sent_right;
    uint64_t rumble_sent_ms;
    // Per pad rather than per guest user, so it survives reassignment.
    KeystrokeState keystroke;
  };

  // WindowListener
  void OnClosing(rex::ui::UIEvent& e) override;
  void OnLostFocus(rex::ui::UISetupEvent& e) override;
  void OnGotFocus(rex::ui::UISetupEvent& e) override;

  void HandleEvent(const SDL_Event& event);
  std::unique_lock<std::mutex> DrainAndLock();
  void ProcessEventLocked(const SDL_Event& event);
  // Slow SDL calls (open/close, rumble, capability queries) run on a worker
  // thread so a guest XInput call never waits on SDL's joystick lock.
  void PostToWorker(std::function<void()> job);
  void WorkerMain();
  void StopWorker();
  void OpenPadOnWorker(SDL_JoystickID instance_id);
  void ClosePadOnWorker(SDL_JoystickID instance_id);
  void FlushRumbleOnWorker();
  void RefreshCapsOnWorker();
  void OnControllerDeviceAxisMotionLocked(const SDL_Event& event);
  void OnControllerDeviceButtonChangedLocked(const SDL_Event& event);

  inline uint64_t AnalogToKeyfield(const X_INPUT_GAMEPAD& gamepad) const;
  std::optional<size_t> GetControllerIndexFromInstanceID(SDL_JoystickID instance_id);
  ControllerState* FindController(DeviceId id);
  DeviceId AllocateDeviceId();
  bool TestSDLVersion() const;
  void UpdateXCapabilities(ControllerState& state);
  void QueueControllerUpdate();

  rex::ui::Window* attached_window_ = nullptr;
  bool sdl_events_initialized_;
  bool SDL_Gamepad_initialized_;
  std::atomic<int> sdl_events_unflushed_;
  std::atomic<bool> sdl_pumpevents_queued_;
  // Appended in connection order, never re-sorted, and unbounded: the
  // assignment decides how many the guest sees.
  std::vector<ControllerState> controllers_;
  std::mutex controllers_mutex_;
  // controllers_[0]'s instance id (0 = none), readable without the lock from
  // the SDL event watch, which may run while SDL holds its joystick lock.
  std::atomic<SDL_JoystickID> first_pad_instance_id_{0};
  std::mutex event_queue_mutex_;
  std::vector<SDL_Event> pending_events_;
  std::thread worker_;
  std::mutex worker_mutex_;
  std::condition_variable worker_cv_;
  std::deque<std::function<void()>> worker_jobs_;
  bool worker_stop_ = false;
  std::atomic<bool> rumble_queued_{false};
  std::atomic<bool> caps_queued_{false};
  // Never rewound, so a handle held past a disconnect resolves to nothing.
  uint64_t next_device_id_ = 1;
};

}  // namespace rex::input::sdl
