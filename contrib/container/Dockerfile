FROM ubuntu:20.04

RUN apt-get -q update && \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    apt-get -q -y --no-install-recommends install \
    git \
    gcc \
    g++ \
    gfortran \
    numdiff \
    python \
    libblas-dev \
    liblapack-dev \
    libboost-all-dev \
    doxygen \
    ca-certificates \
    wget \
    make \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/exa-suite

ENV WORKING_DIRECTORY="/opt/exa-suite"

RUN cd "$WORKING_DIRECTORY" && \
    git clone https://github.com/scibuilder/metis.git && \
    cd metis && \
    cmake . && \
    make

RUN cd "$WORKING_DIRECTORY" && \
    mkdir petsc && \
    cd petsc && \
    wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.14.5.tar.gz && \
    wget https://github.com/hypre-space/hypre/archive/refs/tags/v2.20.0.tar.gz -O hypre-v2.20.0.tar.gz  && \
    tar xf petsc-lite-3.14.5.tar.gz
COPY ./scripts/config_petsc.sh "$WORKING_DIRECTORY/petsc/petsc-3.14.5/"
RUN cd "$WORKING_DIRECTORY/petsc/petsc-3.14.5" && \
    bash config_petsc.sh && \
    make -j4

RUN cd "$WORKING_DIRECTORY" && \
    mkdir trilinos && \
    cd trilinos && \
    wget https://github.com/trilinos/Trilinos/archive/trilinos-release-12-12-1.tar.gz && \
    tar xf trilinos-release-12-12-1.tar.gz  && \
    cd Trilinos-trilinos-release-12-12-1/ && \
    mkdir build && \
    cd build/
COPY ./scripts/config_trilinos.sh "$WORKING_DIRECTORY/trilinos/Trilinos-trilinos-release-12-12-1/build/"
RUN cd "$WORKING_DIRECTORY/trilinos/Trilinos-trilinos-release-12-12-1/build" && \
    bash ./config_trilinos.sh && \
    make -j4 && \
    make install

RUN cd "$WORKING_DIRECTORY" && \
    git clone https://github.com/dealii/dealii.git

RUN cd "$WORKING_DIRECTORY" && \
    mkdir p4est && \
    cd p4est && \
    wget http://p4est.github.io/release/p4est-2.0.tar.gz
COPY ./scripts/config_p4est.sh "$WORKING_DIRECTORY/p4est/"
RUN cd "$WORKING_DIRECTORY/p4est" && \
    bash config_p4est.sh

RUN cd "$WORKING_DIRECTORY/dealii" && \
    mkdir build && \
    cd build/
COPY ./scripts/config_dealii.sh "$WORKING_DIRECTORY/dealii/build/"
RUN cd "$WORKING_DIRECTORY/dealii/build" && \
    bash ./config_dealii.sh && \
    make -j4 && \
    make install

RUN cd $WORKING_DIRECTORY && \
    mkdir fftw && \
    cd fftw && \
    wget http://fftw.org/fftw-3.3.7.tar.gz && \
    tar -xf fftw-3.3.7.tar.gz && \
    cd fftw-3.3.7
COPY ./scripts/config_fftw.sh "$WORKING_DIRECTORY/fftw/fftw-3.3.7/"
RUN cd "$WORKING_DIRECTORY/fftw/fftw-3.3.7" && \
    bash config_fftw.sh && \
    make && \
    make install

COPY . "$WORKING_DIRECTORY/exadg"

RUN cd "$WORKING_DIRECTORY/exadg" && \
    mkdir build && \
    cd build/ && \
    cp $WORKING_DIRECTORY/exadg/scripts/config_exadg.sh . && \
    bash ./config_exadg.sh -D EXADG_WITH_LIKWID=OFF && \
    make release && \
    make -j4
