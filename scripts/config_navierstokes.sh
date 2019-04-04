rm -rf CMakeFiles/ CMakeCache.txt

# issue the build - must usually not be modified
cmake \
    -D DEAL_II_DIR="$WORKING_DIRECTORY/sw/build" \
    -D USE_DEAL_SPECTRUM=ON \
    .
