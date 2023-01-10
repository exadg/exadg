FROM dealii/dealii:master-focal

ARG branch_name

RUN  git clone https://github.com/bergbauer/exadg \
     && cd exadg \
     && git fetch origin \
     && git checkout $branch_name \
     && mkdir build \
     && cd build \
     && cmake ../ -DCMAKE_BUILD_TYPE="Release" \
     && make -j2 \
     && cd .. \
     && mkdir build-debug \
     && cd build-debug \
     && cmake ../ -DCMAKE_BUILD_TYPE="Debug" \
     && make -j2
