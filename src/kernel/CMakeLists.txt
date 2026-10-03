set(REXKERNEL_SOURCES
    kernel_init.cpp

    # XAM subsystem - provides __imp__Xam* symbols
    xam/xam_module.cpp
    xam/xam_avatar.cpp
    xam/xam_content.cpp
    xam/xam_content_aggregate.cpp
    xam/xam_content_device.cpp
    xam/xam_debug.cpp
    xam/xam_enum.cpp
    xam/xam_info.cpp
    xam/xam_input.cpp
    xam/xam_locale.cpp
    xam/xam_msg.cpp
    xam/xam_net.cpp
    xam/xam_notify.cpp
    xam/xam_nui.cpp
    xam/xam_party.cpp
    xam/xam_task.cpp
    xam/xam_ui.cpp
    xam/xam_user.cpp
    xam/xam_video.cpp
    xam/xam_voice.cpp
    xam/xam_misc.cpp
    xam/apps/xam_app.cpp
    xam/apps/xgi_app.cpp
    xam/apps/xlivebase_app.cpp
    xam/apps/xmp_app.cpp

    # XBDM subsystem - provides __imp__Xbdm* symbols
    xbdm/xbdm_module.cpp
    xbdm/xbdm_misc.cpp

    # xboxkrnl exports - provides __imp__ symbols for generated code
    # xboxkrnl/cert_monitor.cpp    # TODO: Translate JIT methods for AOT
    # xboxkrnl/debug_monitor.cpp   # TODO: Translate JIT methods for AOT
    xboxkrnl/xboxkrnl_crypt.cpp
    xboxkrnl/xboxkrnl_debug.cpp
    xboxkrnl/xboxkrnl_error.cpp
    xboxkrnl/xboxkrnl_hal.cpp
    xboxkrnl/xboxkrnl_hid.cpp
    xboxkrnl/xboxkrnl_io.cpp
    xboxkrnl/xboxkrnl_io_info.cpp
    xboxkrnl/xboxkrnl_memory.cpp
    xboxkrnl/xboxkrnl_misc.cpp
    xboxkrnl/xboxkrnl_module.cpp
    xboxkrnl/xboxkrnl_modules.cpp
    xboxkrnl/xboxkrnl_ob.cpp
    xboxkrnl/xboxkrnl_rtl.cpp
    xboxkrnl/xboxkrnl_strings.cpp
    xboxkrnl/xboxkrnl_threading.cpp
    # xboxkrnl/xboxkrnl_usbcam.cpp  # TODO: lol eventually.
    xboxkrnl/xboxkrnl_video.cpp
    xboxkrnl/xboxkrnl_xconfig.cpp
    xboxkrnl/xboxkrnl_audio.cpp
    xboxkrnl/xboxkrnl_audio_xma.cpp

    crt/heap.cpp
    crt/file.cpp
    crt/memory.cpp
    crt/string.cpp
    crt/threading.cpp
)

list(TRANSFORM REXKERNEL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
target_sources(rexruntime PRIVATE ${REXKERNEL_SOURCES})

target_link_libraries(rexruntime PRIVATE o1heap rexcore rexfilesystem rexui rexaudio rexinput)

# imgui is PUBLIC on rexruntime, not PRIVATE.
#
# include/rex/rex_app.h is a public header of rexruntime and it reaches
# <imgui.h> transitively:
#
#   rex/rex_app.h -> rex/ui/imgui_dialog.h -> rex/ui/imgui_drawer.h
#                 -> rex/ui/style.h -> #include <imgui.h>
#
# The imgui target does export its include directory as PUBLIC, but a PRIVATE
# link from rexruntime stops the propagation there. Every consumer of rexruntime
# -- which is every generated project, since that is how the runtime is reached
# -- then fails to compile, with an error in a file it never wrote:
#
#   include/rex/ui/style.h:18:10: fatal error: 'imgui.h' file not found
#
# Nothing about imgui needs to be private here: it is vendored into the build,
# and this target is the only thing standing between it and the consumer.
target_link_libraries(rexruntime PUBLIC imgui)
