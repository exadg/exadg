#cd /C/src/fftw-3.3.3/build/lib
mkdir libfftw3_obj
cd libfftw3_obj
ar -x ../libfftw3.a
cd ..
mkdir libfftw3_mpi_obj
cd libfftw3_mpi_obj
ar -x ../libfftw3_mpi.a
cd ..
mkdir combined
ar cru combined/libfftw3.a ./libfftw3_obj/*.o ./libfftw3_mpi_obj/*.o
ranlib combined/libfftw3.a
