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

rm -rf CMakeFiles/ CMakeCache.txt

DEAL=$WORKING_DIRECTORY/dealii
DEAL_INSTALL=$WORKING_DIRECTORY/dealii/install

# TPLs
P4EST=$WORKING_DIRECTORY/p4est
METIS=$WORKING_DIRECTORY/metis
TRILINOS=$WORKING_DIRECTORY/trilinos/install
PETSC=$WORKING_DIRECTORY/petsc/petsc-3.14.5

# Note on compiler flags: Note that "-march=native" requires that the hardware 
# on which you compile the code is consistent with the hardware on which you 
# execute the code. If this is not the case, consider to specify the target 
# hardware for compilation, e.g. "-march=haswell" or "-march=skylake-avx512"
# in case of Intel Hardware.
# For more details, see https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html.

cmake \
    -D CMAKE_BUILD_TYPE="DebugRelease" \
    -D CMAKE_C_FLAGS="-march=native -Wno-array-bounds" \
    -D CMAKE_CXX_FLAGS="-std=c++17 -march=native -Wno-array-bounds -Wno-literal-suffix -pthread" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D DEAL_II_CXX_FLAGS_DEBUG="-Og" \
    -D DEAL_II_WITH_MPI:BOOL="ON" \
    -D DEAL_II_LINKER_FLAGS="-lpthread" \
    -D DEAL_II_WITH_64BIT_INDICES="ON" \
    -D DEAL_II_WITH_P4EST:BOOL="ON" \
    -D CMAKE_INSTALL_PREFIX="$DEAL_INSTALL" \
    -D P4EST_DIR="$P4EST" \
    -D DEAL_II_WITH_METIS:BOOL="ON" \
    -D METIS_DIR:FILEPATH="$METIS" \
    -D DEAL_II_WITH_TRILINOS:BOOL="ON" \
    -D TRILINOS_DIR:FILEPATH="$TRILINOS" \
    -D DEAL_II_WITH_PETSC:BOOL="ON" \
    -D PETSC_DIR="$PETSC" \
    -D PETSC_ARCH="arch-linux-c-opt" \
    -D DEAL_II_COMPONENT_DOCUMENTATION="OFF" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    $DEAL
