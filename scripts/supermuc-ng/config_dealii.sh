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

DEAL=$WORKING_DIRECTORY/sw/dealii

rm -rf CMakeFiles/ CMakeCache.txt

cmake \
  -D CMAKE_BUILD_TYPE="DebugRelease" \
  -D CMAKE_CXX_FLAGS="-std=c++17 -march=skylake-avx512 -Wno-array-bounds -Wno-literal-suffix -pthread -DFE_EVAL_FACTORY_DEGREE_MAX=15" \
  -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
  -D DEAL_II_CXX_FLAGS_DEBUG="-Og" \
  -D CMAKE_C_FLAGS="-march=skylake-avx512 -Wno-array-bounds" \
  -D DEAL_II_WITH_MPI:BOOL="ON" \
  -D DEAL_II_LINKER_FLAGS="-lpthread" \
  -D DEAL_II_WITH_64BIT_INDICES="ON" \
  -D DEAL_II_WITH_TRILINOS:BOOL="ON" \
  -D TRILINOS_DIR:FILEPATH="$WORKING_DIRECTORY/sw/trilinos-install" \
  -D DEAL_II_WITH_METIS:BOOL="ON" \
  -D METIS_DIR:FILEPATH="$WORKING_DIRECTORY/sw/metis" \
  -D DEAL_II_FORCE_BUNDLED_BOOST="OFF" \
  -D DEAL_II_WITH_GSL="OFF" \
  -D DEAL_II_WITH_NETCDF="OFF" \
  -D DEAL_II_WITH_P4EST="ON" \
  -D DEAL_II_WITH_PETSC="ON" \
  -D PETSC_DIR="$WORKING_DIRECTORY/sw/petsc-3.14.5" \
  -D PETSC_ARCH="arch-linux-c-opt" \
  -D DEAL_II_WITH_LAPACK="ON" \
  -D P4EST_DIR="$WORKING_DIRECTORY/sw" \
  -D LAPACK_LIBRARIES="$MKL_LIBDIR/libmkl_gf_lp64.so;$MKL_LIBDIR/libmkl_sequential.so;$MKL_LIBDIR/libmkl_core.so;$MKL_LIBDIR/libmkl_gf_lp64.so;$MKL_LIBDIR/libmkl_sequential.so;$MKL_LIBDIR/libmkl_core.so;$MKL_LIBDIR/libmkl_sequential.so;/lib64/libpthread.so.0;/lib64/libm.so.6;/lib64/libdl.so.2" \
  -D DEAL_II_COMPONENT_DOCUMENTATION="OFF" \
  -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
  $DEAL
