# embed_spirv.cmake — Generate C++ header with embedded SPIR-V bytecode.
#
# Called by CMake as:
#   cmake -DSPV_DIR=<dir> -DOUTPUT=<output.h> -P embed_spirv.cmake
#
# Reads *.spv files from SPV_DIR and produces a header with:
#   namespace gpu_spirv { static const unsigned char <name>_data[]; ... }

set(HEADER "// Auto-generated — do not edit.  Regenerate with: cmake -P embed_spirv.cmake\n")
string(APPEND HEADER "#pragma once\n#include <cstddef>\n\nnamespace gpu_spirv {\n\n")

file(GLOB SPV_FILES "${SPV_DIR}/*.spv")
foreach(SPV_FILE ${SPV_FILES})
    get_filename_component(BASENAME ${SPV_FILE} NAME_WE)

    file(READ "${SPV_FILE}" SPV_HEX HEX)
    file(SIZE "${SPV_FILE}" SPV_SIZE)

    string(APPEND HEADER "static const unsigned char ${BASENAME}_data[] = {\n")

    string(LENGTH "${SPV_HEX}" HEX_LEN)
    set(POS 0)
    set(LINE "    ")
    set(COL 0)
    while(POS LESS HEX_LEN)
        string(SUBSTRING "${SPV_HEX}" ${POS} 2 BYTE)
        string(APPEND LINE "0x${BYTE},")
        math(EXPR POS "${POS} + 2")
        math(EXPR COL "${COL} + 1")
        if(COL EQUAL 16)
            string(APPEND HEADER "${LINE}\n")
            set(LINE "    ")
            set(COL 0)
        endif()
    endwhile()
    if(COL GREATER 0)
        string(APPEND HEADER "${LINE}\n")
    endif()

    string(APPEND HEADER "};\n")
    string(APPEND HEADER "static const size_t ${BASENAME}_size = ${SPV_SIZE};\n\n")
endforeach()

string(APPEND HEADER "} // namespace gpu_spirv\n")

file(WRITE "${OUTPUT}" "${HEADER}")
message(STATUS "Generated embedded SPIR-V header: ${OUTPUT}")
