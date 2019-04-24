# name of deal.II library directory
DEAL=matrixfree

rm -rf CMakeFiles/ CMakeCache.txt

# issue the build - must usually not be modified
cmake \
    -D CMAKE_CXX_FLAGS="-march=native -Wno-array-bounds -Wno-literal-suffix -pthread" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D DEAL_II_CXX_FLAGS_DEBUG="-Og" \
    -D CMAKE_C_FLAGS="-march=native -Wno-array-bounds" \
    -D DEAL_II_WITH_MPI:BOOL="ON" \
    -D DEAL_II_LINKER_FLAGS="-lpthread" \
    -D DEAL_II_WITH_TRILINOS:BOOL="OFF" \
    -D TRILINOS_DIR:FILEPATH="$WORKING_DIRECTORY/sw/trilinos-install" \
    -D DEAL_II_WITH_METIS:BOOL="OFF" \
    -D METIS_DIR:FILEPATH="$WORKING_DIRECTORY/sw/metis" \
    -D DEAL_II_FORCE_BUNDLED_BOOST="OFF" \
    -D DEAL_II_WITH_GSL="OFF" \
    -D DEAL_II_WITH_NETCDF="OFF" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D P4EST_DIR="$WORKING_DIRECTORY/sw" \
    -D DEAL_II_COMPONENT_DOCUMENTATION="OFF" \
    $WORKING_DIRECTORY/sw/$DEAL
