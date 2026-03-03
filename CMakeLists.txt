set(CMAKE_CXX_STANDARD 11)

# define build type
IF( DEFINED CMAKE_BUILD_TYPE )
  SET( CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Set to either \"Release\" or \"Debug\"" )
ELSE()
  SET( CMAKE_BUILD_TYPE Release CACHE STRING "Set to either \"Release\" or \"Debug\"" )
ENDIF()

# In CMake 4, 3.10 is deprecated and 3.5 has been removed.
# use 3.1...3.10 so all of these versions are acceptable as min. version.
# https://cmake.org/cmake/help/latest/command/cmake_minimum_required.html
cmake_minimum_required(VERSION 3.1...3.10)

PROJECT(openEMS CXX C)

# default
set(LIB_VERSION_MAJOR 0)
set(LIB_VERSION_MINOR 37)
set(LIB_VERSION_PATCH 0)
set(LIB_VERSION_STRING ${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}.${LIB_VERSION_PATCH})

set(VERSION "v${LIB_VERSION_STRING}")

IF(EXISTS ${PROJECT_SOURCE_DIR}/localConfig.cmake)
   include(${PROJECT_SOURCE_DIR}/localConfig.cmake)
ENDIF()

#ADD_DEFINITIONS( -D__SSE2__ )

# add git revision
IF(EXISTS ${PROJECT_SOURCE_DIR}/.git )
  FIND_PACKAGE(Git)
  # Get the latest abbreviated commit hash of the working branch
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --always
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GITREV
  )
  set(VERSION ${GITREV})
  string(STRIP ${VERSION} VERSION)
  message(STATUS "Found Git repository, ${PROJECT_NAME} version tag: ${VERSION}")
ENDIF()

ADD_DEFINITIONS(-DGIT_VERSION=\"${VERSION}\")

#
# Set up RPATH for the project
#
option(ENABLE_RPATH "Enable rpath support on Linux and Mac" ON)
if(NOT CMAKE_INSTALL_RPATH)
  # the RPATH to be used when installing, but only if it's not a system directory
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}" isSystemDir)
  IF("${isSystemDir}" STREQUAL "-1")
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
  ENDIF("${isSystemDir}" STREQUAL "-1")
endif()
if(APPLE AND NOT CMAKE_INSTALL_NAME_DIR)
  set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
endif()
if(UNIX AND ENABLE_RPATH)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if (WITH_MPI)
	ADD_DEFINITIONS(-DMPI_SUPPORT)
	# Require MPI for this project:
	find_package(MPI REQUIRED)
	INCLUDE_DIRECTORIES( ${MPI_INCLUDE_PATH} )

	message(STATUS "Found MPI. INCLUDE " ${MPI_INCLUDE_PATH})
endif()

option(WITH_GPU "Build Vulkan GPU FDTD engine (requires Vulkan SDK with glslc)" OFF)
if (WITH_GPU)
	find_package(Vulkan REQUIRED)
	find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
	ADD_DEFINITIONS(-DWITH_GPU)
	INCLUDE_DIRECTORIES(${Vulkan_INCLUDE_DIRS})

	message(STATUS "Found Vulkan: ${Vulkan_LIBRARIES}")
	message(STATUS "glslc: ${GLSLC}")

	# Compile GLSL shaders to SPIR-V
	set(GPU_SHADER_SRC ${openEMS_SOURCE_DIR}/FDTD/gpu/shaders)
	set(GPU_SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_shaders)
	file(MAKE_DIRECTORY ${GPU_SHADER_OUT})

	set(GPU_SHADER_NAMES
		update_voltages update_currents apply_excitation
		upml_pre_voltage upml_post_voltage upml_pre_current upml_post_current
		dispersive_pre_voltage dispersive_pre_current
		dispersive_apply_voltage dispersive_apply_current
		tfsf_voltage tfsf_current
		mur_pre_voltage mur_post_voltage mur_apply_voltage
		rlc_pre_voltage rlc_apply_voltage
		gather_probes reduce_energy
		fused_update_voltages fused_update_currents
	)
	set(GPU_SPV_FILES)
	foreach(S ${GPU_SHADER_NAMES})
		add_custom_command(
			OUTPUT ${GPU_SHADER_OUT}/${S}.spv
			COMMAND ${GLSLC} -O ${GPU_SHADER_SRC}/${S}.comp -o ${GPU_SHADER_OUT}/${S}.spv
			DEPENDS ${GPU_SHADER_SRC}/${S}.comp
			COMMENT "Compiling shader ${S}.comp -> SPIR-V"
		)
		list(APPEND GPU_SPV_FILES ${GPU_SHADER_OUT}/${S}.spv)
	endforeach()

	# Generate C++ header with embedded SPIR-V byte arrays
	add_custom_command(
		OUTPUT ${GPU_SHADER_OUT}/gpu_shader_spirv.h
		COMMAND ${CMAKE_COMMAND}
			-DSPV_DIR=${GPU_SHADER_OUT}
			-DSHADERS="${GPU_SHADER_NAMES}"
			-DOUTPUT=${GPU_SHADER_OUT}/gpu_shader_spirv.h
			-P ${GPU_SHADER_SRC}/embed_spirv.cmake
		DEPENDS ${GPU_SPV_FILES} ${GPU_SHADER_SRC}/embed_spirv.cmake
		COMMENT "Generating embedded SPIR-V header"
	)
	add_custom_target(gpu_shaders DEPENDS ${GPU_SHADER_OUT}/gpu_shader_spirv.h)
	INCLUDE_DIRECTORIES(${GPU_SHADER_OUT})
endif()

# Set locations of extra CMake modules
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${openEMS_SOURCE_DIR}/cmake/Modules/")

# fparser
# $ cmake -D FPARSER_ROOT_DIR=~/opt/openEMS .
# SET(FPARSER_ROOT_DIR ~/opt/openEMS)
find_library(fparser_LIBRARIES 
  NAMES fparser
  HINTS ${FPARSER_ROOT_DIR}/lib${LIB_SUFFIX}
  NO_CMAKE_FIND_ROOT_PATH
)
message(STATUS "fparser: ${fparser_LIBRARIES}" )
#TODO test if fparser was found
INCLUDE_DIRECTORIES( ${FPARSER_ROOT_DIR}/include ) 

# CSXCAD
# $ cmake -D CSXCAD_ROOT_DIR=~/opt/openEMS .
# SET(CSXCAD_ROOT_DIR ~/opt/openEMS)
find_library(CSXCAD_LIBRARIES 
  NAMES CSXCAD
  HINTS ${CSXCAD_ROOT_DIR}/lib${LIB_SUFFIX}
  NO_CMAKE_FIND_ROOT_PATH
)
message(STATUS "CSXCAD_LIBRARIES: ${CSXCAD_LIBRARIES}" )
#TODO test if CSXCADs was found
find_path(CSXCAD_INCLUDE_DIR 
    NAMES ContinuousStructure.h 
    HINTS ${CSXCAD_ROOT_DIR}/include
    PATH_SUFFIXES "CSXCAD" ${CSXCAD_INCLUDE_DIR}
    NO_CMAKE_FIND_ROOT_PATH
)
message(STATUS "CSXCAD_INCLUDE_DIR: ${CSXCAD_INCLUDE_DIR}" )
INCLUDE_DIRECTORIES( ${CSXCAD_INCLUDE_DIR} ) 

# TinyXML module from https://github.com/ros/cmake_modules
find_package(TinyXML REQUIRED)
ADD_DEFINITIONS( -DTIXML_USE_STL )
message(STATUS "TinyXML_INCLUDE_DIR: ${TinyXML_INCLUDE_DIR}")
message(STATUS "TinyXML_LIBRARY: ${TinyXML_LIBRARY}")

INCLUDE_DIRECTORIES( ${TinyXML_INCLUDE_DIR} )

# hdf5
find_package(HDF5 COMPONENTS C HL REQUIRED)
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})
# hdf5 compat
#ADD_DEFINITIONS( -DH5_USE_16_API )
#ADD_DEFINITIONS( -DH5_BUILT_AS_DYNAMIC_LIB )

# boost
find_package(Boost 1.46 COMPONENTS
  thread
  date_time
  serialization
  chrono
  program_options
  REQUIRED
)
if (${Boost_MINOR_VERSION} LESS 69)
    # Since Boost 1.69, "system" is header only, so a check
    # is needed only in Boost 1.68 and earlier.
    find_package(Boost 1.46 COMPONENTS system thread date_time serialization chrono program_options REQUIRED)
endif()
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
INCLUDE_DIRECTORIES (${Boost_INCLUDE_DIRS})

# std::thread / std::mutex support
find_package(Threads REQUIRED)

# vtk
find_package(VTK COMPONENTS vtkCommonCore NO_MODULE QUIET)
if ("${VTK_VERSION}" VERSION_GREATER "9")
    find_package(VTK REQUIRED COMPONENTS IOXML IOGeometry IOLegacy IOPLY NO_MODULE REQUIRED)
else()
    find_package(VTK REQUIRED COMPONENTS vtkIOXML vtkIOGeometry vtkIOLegacy vtkIOPLY NO_MODULE REQUIRED)
    include(${VTK_USE_FILE})
endif()

message(STATUS "Found package VTK. Using version " ${VTK_VERSION})
set( vtk_LIBS ${VTK_LIBRARIES} )
message(STATUS "vtk libraries " ${vtk_LIBS})

INCLUDE_DIRECTORIES (${VTK_INCLUDE_DIR})

# independent tool
ADD_SUBDIRECTORY( nf2ff )

set(SOURCES
  openems.cpp
)

set(PUB_HEADERS openems.h openems_global.h)

# libs
ADD_SUBDIRECTORY( tools )
ADD_SUBDIRECTORY( FDTD )
ADD_SUBDIRECTORY( FDTD/extensions )
ADD_SUBDIRECTORY( Common )

INCLUDE_DIRECTORIES( ${openEMS_SOURCE_DIR} )    #find tools

message(STATUS "Sources:"  ${SOURCES})

if (${MPI_CXX_FOUND})
	set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
	set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
	SET(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

# AVX2 + FMA compile flags for the AVX2 engine sources (must be set here, not in
# subdirectory, because set_source_files_properties is directory-scoped in CMake)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2_FMA)
if(COMPILER_SUPPORTS_AVX2_FMA)
  set_source_files_properties(
    FDTD/operator_avx2.cpp
    FDTD/engine_avx2.cpp
    FDTD/operator_avx2_multithread.cpp
    FDTD/engine_avx2_multithread.cpp
    FDTD/engine_interface_avx2_fdtd.cpp
    PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
  )
  message(STATUS "AVX2+FMA engine: enabled")
else()
  message(WARNING "AVX2+FMA engine: compiler does not support -mavx2 -mfma, engine will not work at runtime on non-AVX2 CPUs")
endif()

add_library( openEMS SHARED ${SOURCES})
#ADD_EXECUTABLE( openEMS main.cpp ${SOURCES})
set_target_properties(openEMS PROPERTIES VERSION ${LIB_VERSION_STRING} SOVERSION ${LIB_VERSION_MAJOR} )
TARGET_LINK_LIBRARIES( openEMS
  ${CSXCAD_LIBRARIES}
  ${fparser_LIBRARIES}
  ${TinyXML_LIBRARY}
  ${HDF5_LIBRARIES}
  ${HDF5_HL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${vtk_LIBS}
  ${MPI_LIBRARIES}
  Threads::Threads
)

if (WITH_GPU)
  TARGET_LINK_LIBRARIES(openEMS Vulkan::Vulkan)
  add_dependencies(openEMS gpu_shaders)
endif()

if (WIN32)
    # make sure "BUILD_OPENEMS_LIB" is only set for openEMS (dll) not openEMS_bin (exe)
    target_compile_definitions(openEMS PRIVATE -DBUILD_OPENEMS_LIB )
endif (WIN32)

# main program
ADD_EXECUTABLE( openEMS_bin main.cpp )
SET_TARGET_PROPERTIES(openEMS_bin PROPERTIES OUTPUT_NAME openEMS)
TARGET_LINK_LIBRARIES(openEMS_bin openEMS)
 
if (WIN32)
    INSTALL(TARGETS openEMS RUNTIME DESTINATION bin ARCHIVE DESTINATION lib${LIB_SUFFIX})
else()
    INSTALL(TARGETS openEMS DESTINATION lib${LIB_SUFFIX})
endif()
INSTALL(TARGETS openEMS_bin DESTINATION bin)

if (UNIX)
    INSTALL( FILES openEMS.sh 
         DESTINATION bin 
         PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
	if (WITH_MPI)
	INSTALL( FILES openEMS_MPI.sh
		DESTINATION bin
		PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)
	endif()
endif ()
INSTALL(FILES ${PUB_HEADERS} DESTINATION include/openEMS)
INSTALL( DIRECTORY matlab DESTINATION share/openEMS )
# TODO mpi, tarball, debug, release
