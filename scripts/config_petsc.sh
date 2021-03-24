./configure \
--with-cc=mpicc --with-fc=mpif90 \
--with-batch \
--with-cxx=mpicxx \
--known-mpi-shared-libraries=1 \
--known-64-bit-blas-indices=0 \
--with-64-bit-indices=1 \
--with-mpi \
--with-shared-libraries=1 \
--download-hypre=../hypre-v2.20.0.tar.gz \
--with-debugging=0 \
CXXOPTFLAGS="-O3 -march=native" \
COPTFLAGS="-O3 -march=native -funroll-all-loops" \
FOPTFLAGS="-O3 -march=native -funroll-all-loops -malign-double" 

