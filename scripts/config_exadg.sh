#######               ######  #######
##                    ##   ## ##
#####   ##  ## #####  ##   ## ## ####
##       ####  ## ##  ##   ## ##   ##
####### ##  ## ###### ######  #######

#rm -rf CMakeFiles/ CMakeCache.txt libexadg.so libexadg.a include/exadg/configuration/config.h

FFTW_INSTALL=$WORKING_DIRECTORY/path/to/fftw-install
LIKWID_INSTALL=$WORKING_DIRECTORY/path/to/likwid-install

cmake \
    -D DEGREE_MAX=15 \
    -D DEAL_II_DIR="$WORKING_DIRECTORY/sw/dealii-build" \
    -D USE_FFTW=ON \
    -D FFTW_LIB="$FFTW_INSTALL/lib" \
    -D FFTW_INCLUDE="$FFTW_INSTALL/include" \
    -D LIKWID_LIB="$LIKWID_INSTALL/lib" \
    -D LIKWID_INCLUDE="$LIKWID_INSTALL/include" \
    -D BUILD_SHARED_LIBS=ON \
    ../
