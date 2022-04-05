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

rm -rf CMakeFiles/ CMakeCache.txt libexadg.so libexadg.a include/exadg/configuration/config.h

cmake \
  -D DEGREE_MAX=15 \
  -D DEAL_II_DIR="$WORKING_DIRECTORY/sw/dealii-build" \
  -D EXADG_WITH_FFTW=ON \
  -D FFTW_LIB="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/lib/" \
  -D FFTW_INCLUDE="$WORKING_DIRECTORY/sw/fftw-3.3.7-install/include" \
  -D EXADG_WITH_LIKWID=ON \
  -D LIKWID_LIB="/dss/dsshome1/lrz/sys/spack/release/19.1/opt/x86_avx512/likwid/4.3.3-gcc-axo3q7s/lib" \
  -D LIKWID_INCLUDE="/dss/dsshome1/lrz/sys/spack/release/19.1/opt/x86_avx512/likwid/4.3.3-gcc-axo3q7s/include" \
  -D BUILD_SHARED_LIBS=ON \
  -D PICKUP_TESTS=OFF \
  ../
