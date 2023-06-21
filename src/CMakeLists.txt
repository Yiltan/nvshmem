# this requires cmake 3.19
include(CheckCompilerFlag)

check_compiler_flag(CUDA -t4 NVCC_THREADS)

set(BOOTSTRAP_VERSION_MAJOR 2)
set(BOOTSTRAP_VERSION_MINOR 8)
set(BOOTSTRAP_VERSION_PATCH 0)

# Start bootstrap definitions
add_library(
  nvshmem_bootstrap_pmi SHARED
)
set(ACTIVE_BOOTSTRAPS nvshmem_bootstrap_pmi)
target_sources(nvshmem_bootstrap_pmi PRIVATE bootstrap/bootstrap_pmi.cpp pmi/simple-pmi/simple_pmi.cpp pmi/simple-pmi/simple_pmiutil.cpp)
target_include_directories(nvshmem_bootstrap_pmi
                           PRIVATE include
                                   bootstrap
                                   pmi/simple-pmi
)

add_library(
  nvshmem_bootstrap_pmi2 SHARED
)
set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmi2)
target_sources(nvshmem_bootstrap_pmi2 PRIVATE bootstrap/bootstrap_pmi.cpp pmi/pmi-2/pmi2_api.c pmi/pmi-2/pmi2_util.c)
target_compile_definitions(nvshmem_bootstrap_pmi2 PRIVATE NVSHMEM_BUILD_PMI2)
target_link_options(nvshmem_bootstrap_pmi2 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_bootstrap.sym")
target_include_directories(nvshmem_bootstrap_pmi2
                           PRIVATE include
                                   bootstrap
                                   pmi/pmi-2
)

if(NVSHMEM_PMIX_SUPPORT)
  add_library(
    nvshmem_bootstrap_pmix SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmix)
  target_sources(nvshmem_bootstrap_pmix PRIVATE bootstrap/bootstrap_pmix.c)
  find_library(PMIX_lib NAMES pmix HINTS "${PMIX_HOME}/lib")
  target_link_libraries(nvshmem_bootstrap_pmix PRIVATE ${PMIX_lib})
  target_include_directories(nvshmem_bootstrap_pmix
                             PRIVATE ${PMIX_HOME}/include
                             include
                             bootstrap
  )
else()
set(NVSHMEM_DEFAULT_PMIX OFF)
endif()

if(NVSHMEM_MPI_SUPPORT)
  add_library(
    nvshmem_bootstrap_mpi SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_mpi)
  set_target_properties(nvshmem_bootstrap_mpi PROPERTIES C_STANDARD 11)
  target_sources(nvshmem_bootstrap_mpi PRIVATE bootstrap/bootstrap_mpi.c)
  target_link_libraries(nvshmem_bootstrap_mpi PRIVATE MPI::MPI_C)
  target_include_directories(nvshmem_bootstrap_mpi
                             PRIVATE include
                             bootstrap
  )
endif()

if(NVSHMEM_SHMEM_SUPPORT)
  add_library(
    nvshmem_bootstrap_shmem SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_shmem)
  set_target_properties(nvshmem_bootstrap_shmem PROPERTIES C_STANDARD 11)
  target_sources(nvshmem_bootstrap_shmem PRIVATE bootstrap/bootstrap_shmem.c)
  target_link_libraries(nvshmem_bootstrap_shmem PRIVATE shmem)
  target_include_directories(nvshmem_bootstrap_shmem
                             PRIVATE include
                             bootstrap
  )
endif()

set_target_properties(${ACTIVE_BOOTSTRAPS}
                      PROPERTIES PREFIX ""
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      VERSION ${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH}
                      SOVERSION ${BOOTSTRAP_VERSION_MAJOR})

# End bootstrap definitions

# Start nvshmem lib definitions
add_library(
  nvshmem STATIC
)

add_library(
  nvshmem_device STATIC
)

add_library(
  nvshmem_host SHARED
)

## Start generic variable configuration
set_target_properties(nvshmem nvshmem_host nvshmem_device
                      PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CXX_STANDARD_REQUIRED ON
                      CUDA_STANDARD_REQUIRED ON
                      CXX_STANDARD 11
                      CUDA_STANDARD 11
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
		      VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
		      SOVERSION ${PROJECT_VERSION_MAJOR}
)

macro(nvshmem_library_set_base_link LIBNAME)
  target_include_directories(
    ${LIBNAME}
    PRIVATE include
            util
            bootstrap
            comm
            coll/host
            coll/device
            coll
            comm/transports/common
            topo
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
              $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(${LIBNAME} PRIVATE CUDA::cudart CUDA::cuda_driver nvidia-ml)
endmacro()

macro(nvshmem_library_set_base_config LIBNAME)
  target_compile_definitions(${LIBNAME}
    PRIVATE $<$<CONFIG:Debug>:_NVSHMEM_DEBUG;NVSHMEM_TRACE;NVSHMEM_IBGDA_DEBUG>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","x86_64">,NVSHMEM_X86_64,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","ppc64le">,__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS;NVSHMEM_PPC64LE,>
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
  )

  target_compile_options(${LIBNAME}
    INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVSHMEM_VERBOSE}>>:-Xptxas -v>
    PRIVATE $<IF:$<CONFIG:Debug>,-O0;-g;,-O3>
    $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    $<IF:$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>,-O0;-g;-G;,-O3>
    $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},"x86_64">,-msse,>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  )

  if(NVSHMEM_DEVEL)
    target_compile_options(
      ${LIBNAME}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror
              all-warnings>
              $<$<COMPILE_LANGUAGE:CXX>:-Werror
              -Wall
              -Wextra
              -Wno-unused-function
              -Wno-unused-parameter>
    )
  endif()
endmacro()

set(NVSHMEM_DEVICE_SOURCES
    coll/device/gpu_coll.cu
    coll/device/recexchalgo.cu
    comm/device/proxy_device.cu
    init/init_device.cu
    launch/collective_launch_device.cu
)

set(NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT
    coll/device/kernels/alltoall.cu
    coll/device/kernels/barrier.cu
    coll/device/kernels/broadcast.cu
    coll/device/kernels/fcollect.cu
    coll/device/kernels/reduce.cu
    comm/host/cuda_interface_sync.cu
    comm/host/proxy/rma.cu
    comm/host/quiet_on_stream.cu
)

set(NVSHMEM_HOST_SOURCES
    bootstrap/bootstrap.cpp
    bootstrap/bootstrap_loader.cpp
    coll/host/cpu_coll.cpp
    coll/host/alltoall.cpp
    coll/host/alltoall_on_stream.cpp
    coll/host/barrier.cpp
    coll/host/barrier_on_stream.cpp
    coll/host/broadcast.cpp
    coll/host/broadcast_on_stream.cpp
    coll/host/fcollect.cpp
    coll/host/fcollect_on_stream.cpp
    coll/host/rdxn.cpp
    coll/host/rdxn_on_stream.cpp
    comm/host/putget.cpp
    comm/host/fence.cpp
    comm/host/quiet.cpp
    comm/host/sync.cpp
    comm/host/amo.cpp
    comm/proxy/proxy.cpp
    comm/transport.cpp
    comm/transports/p2p/p2p.cpp
    init/cudawrap.cpp
    init/init.cu
    init/init_nvtx.cpp
    init/query_host.cpp
    launch/collective_launch.cpp
    mem/mem.cpp
    team/team.cu
    team/team_internal.cu
    topo/topo.cpp
    util/cs.cpp
    util/debug.cpp
    util/env_vars.cpp
    util/util.cpp
    util/sockets.cpp
)

nvshmem_library_set_base_link(nvshmem)
nvshmem_library_set_base_link(nvshmem_host)
nvshmem_library_set_base_link(nvshmem_device)
nvshmem_library_set_base_config(nvshmem)
nvshmem_library_set_base_config(nvshmem_host)
nvshmem_library_set_base_config(nvshmem_device)
## End generic variable configuration

## Start transports
set(TRANSPORT_VERSION_MAJOR 1)
set(TRANSPORT_VERSION_MINOR 0)
set(TRANSPORT_VERSION_PATCH 0)

set(ACTIVE_TRANSPORTS)

if(NVSHMEM_UCX_SUPPORT)
  add_library(
    nvshmem_transport_ucx SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ucx)
  target_sources(nvshmem_transport_ucx PRIVATE
                 comm/transports/ucx/ucx.cpp
                 comm/transports/common/transport_common.cpp
  )
  target_link_options(nvshmem_transport_ucx PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ucx PRIVATE include comm/transports/common)
  target_link_libraries(nvshmem_transport_ucx PRIVATE ucx::ucs ucx::ucp CUDA::cudart)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ucx PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ucx PRIVATE
                   comm/transports/common/transport_gdr_common.cpp
    )
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ucx)
endif()

if(NVSHMEM_IBRC_SUPPORT)
  add_library(
    nvshmem_transport_ibrc SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibrc)
  target_sources(nvshmem_transport_ibrc PRIVATE
                 comm/transports/ibrc/ibrc.cpp
                 comm/transports/common/transport_common.cpp
                 comm/transports/common/transport_ib_common.cpp
  )
  target_link_options(nvshmem_transport_ibrc PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibrc PRIVATE include comm/transports/common)
  target_link_libraries(nvshmem_transport_ibrc PRIVATE CUDA::cudart)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ibrc PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ibrc PRIVATE
                   comm/transports/common/transport_gdr_common.cpp
    )
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibrc)
endif()

if(NVSHMEM_IBDEVX_SUPPORT OR NVSHMEM_IBGDA_SUPPORT)
  find_library(MLX5_lib NAMES mlx5)
endif()

if(NVSHMEM_IBDEVX_SUPPORT)
  add_library(
    nvshmem_transport_ibdevx SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibdevx)
  target_sources(nvshmem_transport_ibdevx PRIVATE
                 comm/transports/ibdevx/ibdevx.cpp
                 comm/transports/common/transport_common.cpp
                 comm/transports/common/transport_ib_common.cpp
                 comm/transports/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_ibdevx PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibdevx PRIVATE include comm/transports/common)
  target_link_libraries(nvshmem_transport_ibdevx PRIVATE ${MLX5_lib} CUDA::cudart)
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibdevx)
endif()

if(NVSHMEM_IBGDA_SUPPORT)
  set(NVSHMEM_DEVICE_SOURCES
      ${NVSHMEM_DEVICE_SOURCES}
      init/ibgda_init_device.cu
  )
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      init/ibgda_init.cu
  )
  add_library(
    nvshmem_transport_ibgda SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibgda)
  target_sources(nvshmem_transport_ibgda PRIVATE
                 comm/transports/ibgda/ibgda.cpp
                 comm/transports/common/transport_common.cpp
                 comm/transports/common/transport_ib_common.cpp
                 comm/transports/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_ibgda PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibgda PRIVATE include comm/transports/common)
  target_link_libraries(nvshmem_transport_ibgda PRIVATE ${MLX5_lib} CUDA::cudart)
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibgda)
endif()

if(NVSHMEM_LIBFABRIC_SUPPORT)
  find_library(FABRIC_lib NAMES fabric HINTS "${LIBFABRIC_HOME}/lib" "${LIBFABRIC_HOME}/lib64")
  add_library(
    nvshmem_transport_libfabric SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_libfabric)
  target_sources(nvshmem_transport_libfabric PRIVATE
                 comm/transports/libfabric/libfabric.cpp
                 comm/transports/common/transport_common.cpp
  )
  target_link_options(nvshmem_transport_libfabric PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_libfabric PRIVATE include comm/transports/common "${LIBFABRIC_HOME}/include/")
  target_link_libraries(nvshmem_transport_libfabric PRIVATE ${FABRIC_lib} CUDA::cudart)
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_libfabric)
endif()

set_target_properties(${ACTIVE_TRANSPORTS}
                      PROPERTIES PREFIX ""
                      CXX_STANDARD_REQUIRED ON
                      CXX_STANDARD 11
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      VERSION ${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH}
                      SOVERSION ${TRANSPORT_VERSION_MAJOR})
## End transports

## Start dependencies
if(NVSHMEM_USE_NCCL)
  target_include_directories(nvshmem PRIVATE ${NCCL_INCLUDE})
  target_include_directories(nvshmem_host PRIVATE ${NCCL_INCLUDE})
  target_include_directories(nvshmem_device PRIVATE ${NCCL_INCLUDE})
endif()

if(NVSHMEM_USE_DLMALLOC)
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      mem/dlmalloc.cpp
  )
else()
set(NVSHMEM_HOST_SOURCES
    ${NVSHMEM_HOST_SOURCES}
    mem/custom_malloc.cpp
  )
endif()
## End dependecies

## Start inlining requirements
if(NVSHMEM_ENABLE_ALL_DEVICE_INLINING)
set(INCEXPORTS
    ${INCEXPORTS}
    device/pt-to-pt/transfer_device.cuh
)
configure_file(include/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_BINARY_DIR}/include/device/pt-to-pt/transfer_device.cuh COPYONLY)
else()
set(INCEXPORTS
    ${INCEXPORTS}
    device/pt-to-pt/nvshmemi_transfer_api.cuh
)
set(NVSHMEM_DEVICE_SOURCES
    ${NVSHMEM_DEVICE_SOURCES}
    comm/device/transfer_device.cu
)
configure_file(include/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_SOURCE_DIR}/comm/device/transfer_device.cu COPYONLY)
endif()
## End inlining requirements

## Start final lib prep
target_link_libraries(nvshmem_host PRIVATE nvshmem_device)
target_link_options(nvshmem_host PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_host.sym")
set_source_files_properties(${NVSHMEM_HOST_SOURCES} ${NVSHMEM_DEVICE_SOURCES} PROPERTIES COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>)
target_sources(nvshmem PRIVATE ${NVSHMEM_HOST_SOURCES} ${NVSHMEM_DEVICE_SOURCES} ${NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT})
target_sources(nvshmem_host PRIVATE ${NVSHMEM_HOST_SOURCES})
target_sources(nvshmem_device PRIVATE ${NVSHMEM_DEVICE_SOURCES} ${NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT})

add_library(NVSHMEM::nvshmem ALIAS nvshmem)

## End final lib prep
# End nvshmem lib definitions

# Start nvshmem-info definitions.
add_executable(nvshmem-info)
target_sources(nvshmem-info PRIVATE util/nvshmem-info.cpp)
set_target_properties(nvshmem-info PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD 11 RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
target_link_libraries(nvshmem-info CUDA::cudart CUDA::cuda_driver nvidia-ml nvshmem)
target_include_directories(nvshmem-info PRIVATE include)
# End nvshmem-info definitions.

# Start header configuration and export
configure_file(include/nvshmem_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/nvshmem_version.h @ONLY)
configure_file(include/nvshmem_build_options.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/nvshmem_build_options.h)

set(INCEXPORTS
    ${INCEXPORTS}
    device/coll/alltoall.cuh
    device/coll/barrier.cuh
    device/coll/broadcast.cuh
    device/coll/defines.cuh
    device/coll/fcollect.cuh
    device/coll/reduce.cuh
    device/coll/utils.cuh
    device/pt-to-pt/ibgda_device.cuh
    device/pt-to-pt/proxy_device.cuh
    device/pt-to-pt/utils_device.h
    device/team/team_device.cuh
    device/init/query_device.cuh
    nvshmem.h
    nvshmem_api.h
    nvshmem_bootstrap_defines.h
    nvshmem_bootstrap.h
    nvshmem_build_options.h
    nvshmem_coll_api.h
    nvshmem_common.cuh
    nvshmem_constants.h
    nvshmem_defines.h
    nvshmem_types.h
    nvshmem_version.h
    nvshmemi_constants.h
    nvshmemi_transport_defines.h
    nvshmemi_proxy.h
    nvshmemi_ibgda.h
    nvshmemi_team.h
    nvshmemi_util.h
    nvshmemx.h
    nvshmemx_api.h
    nvshmemx_coll_api.h
    nvshmemx_defines.h
    nvshmemx_error.h
)

foreach(INC ${INCEXPORTS})
  configure_file(include/${INC} ${CMAKE_CURRENT_BINARY_DIR}/include/${INC} COPYONLY)
endforeach(INC ${INCEXPORTS})
# End header configuration and export

# start share configuration
set(SHAREEXPORTS
    bootstrap_loader.cpp
    bootstrap_mpi.c
    bootstrap_pmi.cpp
    bootstrap_pmix.c
    bootstrap_shmem.c
    bootstrap_util.h
    bootstrap.cpp
)

foreach(SHARE ${SHAREEXPORTS})
  configure_file(bootstrap/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/nvshmem/src/bootstrap-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${SHAREEXPORTS})

set(TRANSPORTINCEXPORTS
    transport.h
    env_defs.h
    cudawrap.h
)

foreach(SHARE ${TRANSPORTINCEXPORTS})
  configure_file(include/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/nvshmem/src/transport-plugins/common/${SHARE} COPYONLY)
endforeach(SHARE ${SHAREEXPORTS})

set(TRANSPORTEXPORTS
    common/mlx5_ifc.h
    common/mlx5_prm.h
    common/transport_common.h
    common/transport_gdr_common.h
    common/transport_ib_common.h
    common/transport_mlx5_common.h
    common/transport_common.cpp
    common/transport_gdr_common.cpp
    common/transport_ib_common.cpp
    common/transport_mlx5_common.cpp
    ibdevx/ibdevx.cpp
    ibdevx/ibdevx.h
    ibgda/ibgda.cpp
    ibrc/ibrc.cpp
    libfabric/libfabric.cpp
    libfabric/libfabric.h
    ucx/ucx.cpp
    ucx/ucx.h
)

foreach(SHARE ${TRANSPORTEXPORTS})
  configure_file(comm/transports/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/nvshmem/src/transport-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${SHAREEXPORTS})

# end share configuration

# start custom builds for nvshmem packages
add_custom_target(git_commit ALL COMMAND test -f ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt || git rev-parse HEAD > ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt || echo "not built from a git repo" > ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt
                  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt)

if(NVSHMEM_BUILD_PACKAGES)
  add_custom_target(hydra
                    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/install_hydra.sh ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build
		    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_nameserver
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_persist
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_pmi_proxy
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun.hydra
                  )
  add_dependencies(nvshmem git_commit hydra)
endif()

# Start Installation definitions
install(
	TARGETS nvshmem nvshmem_host nvshmem_device nvshmem-info ${ACTIVE_BOOTSTRAPS} ${ACTIVE_TRANSPORTS}
  EXPORT NVSHMEMTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/
  DESTINATION bin
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/share/nvshmem/src/
  DESTINATION share/nvshmem/src/
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include/
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/
  DESTINATION examples/
  PATTERN "examples_install" EXCLUDE
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../perftest/
  DESTINATION perftest/
  PATTERN "perftest_install" EXCLUDE
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/install_hydra.sh
  DESTINATION scripts
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_cxx11.sh
  DESTINATION scripts
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../changelog
  DESTINATION ./
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../common.mk
  DESTINATION ./
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../COPYRIGHT.txt
  DESTINATION ./
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../NVSHMEM-SLA.txt
  DESTINATION ./
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt
  DESTINATION ./
)
include(CMakePackageConfigHelpers)

list(APPEND CPACK_GENERATOR "TXZ")

if(NVSHMEM_BUILD_RPM_PACKAGE)
  list(APPEND CPACK_GENERATOR "RPM")
endif()

if(NVSHMEM_BUILD_DEB_PACKAGE)
  list(APPEND CPACK_GENERATOR "DEB")
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(PACKAGE_SUFFIX "amd64")
else()
  set(PACKAGE_SUFFIX "ppc64le")
endif()

set(CPACK_SOURCE_INSTALLED_DIRECTORIES "${CMAKE_SOURCE_DIR}/;/")


set(CPACK_SOURCE_IGNORE_FILES
"/install/;\
/build/;\
/.vscode;\
/include_gdrcopy;\
/include_nccl;\
/.clang-format;\
/.gitignore;\
/.git/;\
/.gitlab-ci.yml;\
/bind.sh;\
/README;\
/testcommon.py;\
/examples/examples_install/;\
/include/nvshmem_build_options.h$;\
/include/nvshmem_version.h$;\
/perftest/perftest_install/;\
/scripts/cronjob_peak.sh;\
/scripts/cronjob_prom.sh;\
/scripts/cronjob_psg.sh;\
/scripts/cronjob_pwr0x.sh;\
/scripts/dependency_installer.sh;\
/scripts/run_dev_coll.sh;\
/scripts/run_perf.sh;\
/scripts/run_tests.sh;\
/scripts/testrunner.pl;\
/src/build/;\
/test/;")


set(CPACK_PACKAGE_NAME "libnvshmem")
set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://developer.nvidia.com/nvshmem")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../pkg/nvshmem_package_description.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NVSHMEM - An implementation of OpenSHMEM for NVIDIA GPUs.")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME "libnvshmem_${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${PROJECT_VERSION_TWEAK}+cuda${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}_${PACKAGE_SUFFIX}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "nvshmem_src_${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${PROJECT_VERSION_TWEAK}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../NVSHMEM-SLA.txt")
set(CPACK_PACKAGE_CONTACT "nvshmem@nvidia.com")

# RPM-Specific Configurations
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_LICENSE "LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS")
set(CPACK_RPM_PACKAGE_AUTOREQ 0)
set(CPACK_RPM_PACKAGE_REQUIRES "/sbin/ldconfig")
set(CPACK_RPM_PACKAGE_REQUIRES_PRE "/sbin/ldconfig")
set(CPACK_RPM_PACKAGE_REQUIRES_PREUN "/sbin/ldconfig")

include(CPack)
# End Installation definitions
