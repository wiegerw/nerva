project(NERVA_TOOLS)

add_executable(mlp mlp.cpp)
target_link_libraries(mlp LINK_PUBLIC nervalib MKL::MKL Eigen3::Eigen)
if (TBB_FOUND)
    target_link_libraries (mlp LINK_PUBLIC -ltbb)
endif()

add_executable(mkl mkl.cpp)
target_link_libraries(mkl LINK_PUBLIC nervalib MKL::MKL Eigen3::Eigen)
if (TBB_FOUND)
    target_link_libraries (mkl LINK_PUBLIC -ltbb)
endif()
