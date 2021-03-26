#!/bin/sh
#########################################################################
#
#                 #######               ######  #######
#                 ##                    ##   ## ##
#                 #####   ##  ## #####  ##   ## ## ####
#                 ##       ####  ## ##  ##   ## ##   ##
#                 ####### ##  ## ###### ######  #######
#
#  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
#
#  Copyright (C) 2021 by the ExaDG authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#########################################################################

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
  --with-blas-lapack-lib=[$MKLROOT/lib/intel64_lin/libmkl_gf_lp64.so,libmkl_sequential.so,libmkl_core.so] \
  CXXOPTFLAGS="-O3 -march=skylake-avx512" \
  COPTFLAGS="-O3 -march=skylake-avx512 -funroll-all-loops" \
  FOPTFLAGS="-O3 -march=skylake-avx512 -funroll-all-loops -malign-double"
