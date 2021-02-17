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
    -D PICKUP_TESTS=ON \
    ../
