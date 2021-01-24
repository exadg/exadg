rm -rf CMakeFiles/ CMakeCache.txt include/exadg/configuration/config.h

# issue the build - must usually not be modified
cmake \
    -D DEAL_II_DIR="$WORKING_DIRECTORY/sw/dealii-build" \
    -D USE_DEAL_SPECTRUM=ON \
    -D FFTW_INC="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/include" \
    -D FFTW_LIB="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/lib/combined" \
    -D LIKWID_LIB="/path_to_likwid-install/lib" \
    -D LIKWID_INCLUDE="/path_to_likwid-install/include" \
    ../
