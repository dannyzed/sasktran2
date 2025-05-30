add_executable(
  sasktran2_tests
  test_main.cpp
  dual/test_dual.cpp
  viewinggeometry/test_tangentaltitude.cpp
  viewinggeometry/test_groundviewing.cpp
  unitsphere/test_lebedev.cpp
  geometry/test_geometry.cpp
  grids/test_grid.cpp
  raytracing/test_spherical_shell.cpp
  raytracing/test_plane_parallel.cpp
  engine/test_engine_basic.cpp
  engine/wf/test_wf_singlescatter_scalar.cpp
  engine/wf/test_wf_vector.cpp
  engine/hr/test_hr_basic.cpp
  sktran_disco/test_util.cpp
  sktran_disco/tests_lowlevel.cpp
  sktran_disco/legacy/test_scalar.cpp
  sktran_disco/legacy/test_vector.cpp
  sktran_disco/test_crash.cpp
  twostream/test_twostream.cpp
  twostream/test_backprop.cpp
  sktran_disco/benchmark/bench_lowlevel.cpp
  engine/benchmark/bench_singlescatter.cpp
  engine/benchmark/bench_do_wf.cpp
  engine/benchmark/bench_hr.cpp
  engine/benchmark/bench_occ.cpp
  atmosphere/surface.cpp)

# Some special logic to support both CATCH2 version 2 and 3 for now
find_package(Catch2 2 CONFIG QUIET)

target_link_libraries(
  sasktran2_tests
  PRIVATE sasktran2 $<$<PLATFORM_ID:Windows>:bcrypt>
          $<$<PLATFORM_ID:Linux,Darwin>:dl> $<$<PLATFORM_ID:Linux>:gfortran>
          $<$<PLATFORM_ID:Linux>:rt>)
if(Catch2_FOUND)
  # Found Catch2 Version 2
  target_link_libraries(sasktran2_tests PRIVATE Catch2::Catch2)
else()
  # Look for Catch2 Version 3
  message("Could not find Catch2 Version 2, Looking for Version 3")
  find_package(Catch2 3 CONFIG REQUIRED)
  target_link_libraries(sasktran2_tests PRIVATE Catch2::Catch2WithMain)
  target_compile_definitions(sasktran2_tests PUBLIC SKTRAN_CATCH2_VERSION3)
endif()

if(INCLUDE_TRACY)
  target_link_libraries(sasktran2_tests INTERFACE TracyClient)
endif()

add_test(NAME TestSASKTRAN2 COMMAND sasktran2_tests)

# This is just a hack to copy dependent DLLs on windows when building inside a
# conda environment.  We copy all the DLLs in CMAKE_PREFIX_PATH/bin over to the
# tests directory if this option is set.
option(COPY_TESTING_DLLS "Test to on to copy DLLs to the testing directory" OFF)

if(COPY_TESTING_DLLS)
  file(GLOB DLLS "${CMAKE_PREFIX_PATH}/bin/*.dll")
  foreach(DLL IN LISTS DLLS)
    add_custom_command(
      TARGET sasktran2_tests
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL}
              $<TARGET_FILE_DIR:sasktran2_tests>)
  endforeach()
endif()
