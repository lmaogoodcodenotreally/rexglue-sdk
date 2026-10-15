# ReXGlue 

add_executable(rexglue
    main.cpp
    ui/ui.cpp
    ui/progress.cpp
    commands/codegen_command.cpp
    commands/init_command.cpp
    commands/test_recompiler.cpp
    commands/project_scan.cpp
)

rex_config_postfix(rexglue)

target_include_directories(rexglue
    PRIVATE
        ${REXGLUE_ROOT}/src
        ${REXGLUE_ROOT}/thirdparty/disasm
        ${REXGLUE_ROOT}/thirdparty/cli11/include
)

target_link_libraries(rexglue
    PRIVATE
        rexcodegen
)

rex_target_pch(rexglue <CLI/CLI.hpp>)

add_executable(rex::rexglue ALIAS rexglue)
