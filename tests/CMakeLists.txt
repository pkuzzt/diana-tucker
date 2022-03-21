project(diana-tests)

# Add DIANA
include_directories("../include")

# Add Google Test
add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


add_executable(${PROJECT_NAME} main.cpp testcases/function/distributed/ttm.cpp testcases/function/distributed/gram.cpp testcases/function/distributed/FunctionDistributedTest.cpp testcases/function/distributed/FunctionDistributedTest.hpp)
target_link_libraries(${PROJECT_NAME} gtest gtest_main)
target_link_libraries(${PROJECT_NAME} ${DIANA_LIBRARIES_LINKED} diana-tucker-lib)