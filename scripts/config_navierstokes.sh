# path to infrastructure folders
WORKING_DIRECTORY=/home/fehn

# issue the build - must usually not be modified
cmake \
    -D DEAL_II_DIR="$WORKING_DIRECTORY/sw/build" \
    -D FFTW_INC="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/include" \
    -D FFTW_LIB="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/lib/combined" \
    -D USE_DEAL_SPECTRUM=ON \
    .
