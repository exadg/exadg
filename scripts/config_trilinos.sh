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

TRILINOS=../
TRILINOS_INSTALL=$WORKING_DIRECTORY/trilinos/install

MPIDIR=/usr
#MPIDIR=/usr/lib64/openmpi

EXTRA_ARGS=$@

cmake \
    -D CMAKE_BUILD_TYPE:STRING="RELEASE" \
    -D CMAKE_CXX_COMPILER:FILEPATH="$MPIDIR/bin/mpicxx" \
    -D CMAKE_C_COMPILER:FILEPATH="$MPIDIR/bin/mpicc" \
    -D CMAKE_Fortran_COMPILER:FILEPATH="$MPIDIR/bin/mpif90" \
    -D CMAKE_CXX_FLAGS="-march=native -O3" \
    -D CMAKE_C_FLAGS="-march=native -O3" \
    -D CMAKE_FORTRAN_FLAGS="-march=native" \
    -D CMAKE_INSTALL_PREFIX:STRING="$TRILINOS_INSTALL" \
    -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
    -D CMAKE_COLOR_MAKEFILE:BOOL=ON \
    -D BUILD_SHARED_LIBS:BOOL=ON \
    -D MPI_INCLUDE_PATH:FILEPATH="$MPIDIR/include" \
    -D MPI_LIBRARY:FILEPATH="$MPIDIR" \
    -D Gtest_SKIP_INSTALL:BOOL=TRUE \
    -D Trilinos_ASSERT_MISSING_PACKAGES=OFF \
    -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=ON \
    -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
    -D Trilinos_ENABLE_SECONDARY_STABLE_CODE:BOOL=ON \
    -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_TESTS:BOOL=OFF \
    -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
    -D Trilinos_ENABLE_AMESOS_SuperLUDist:BOOL=ON \
    -D Trilinos_ENABLE_AMESOS_SuperLU:BOOL=ON \
    -D Trilinos_ENABLE_AMESOS_UMFPACK:BOOL=ON \
    -D Trilinos_ENABLE_Amesos2:BOOL=OFF \
    -D Trilinos_ENABLE_ANASAZI:BOOL=ON \
    -D Trilinos_ENABLE_ANASAZI_AMESOS:BOOL=ON \
    -D Trilinos_ENABLE_ANASAZI_AZTECOO:BOOL=ON \
    -D Trilinos_ENABLE_ANASAZI_BELOS:BOOL=ON \
    -D Trilinos_ENABLE_ANASAZI_EPETRAEXT:BOOL=ON \
    -D Trilinos_ENABLE_ANASAZI_IFPACK:BOOL=ON \
    -D Trilinos_ENABLE_ANASAZI_THYRA:BOOL=OFF \
    -D Trilinos_ENABLE_ANASAZI_TRIUTILS:BOOL=ON \
    -D Trilinos_ENABLE_AZTECOO:BOOL=ON \
    -D Trilinos_ENABLE_Amesos:BOOL=ON \
    -D Trilinos_ENABLE_Anasazi:BOOL=ON \
    -D Trilinos_ENABLE_AztecOO:BOOL=ON \
    -D Trilinos_ENABLE_Belos:BOOL=ON \
    -D Trilinos_ENABLE_Epetra:BOOL=ON \
    -D Trilinos_ENABLE_EpetraExt:BOOL=ON \
    -D Trilinos_ENABLE_EpetraExt_all:BOOL=ON \
    -D Trilinos_ENABLE_FEI:BOOL=OFF \
    -D Trilinos_ENABLE_GALERI:BOOL=OFF \
    -D Trilinos_ENABLE_Galeri:BOOL=OFF \
    -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
    -D Trilinos_ENABLE_Ifpack:BOOL=ON \
    -D Trilinos_ENABLE_Intrepid:BOOL=ON \
    -D Trilinos_ENABLE_Isorropia:BOOL=ON \
    -D Trilinos_ENABLE_Kokkos:BOOL=ON \
    -D Trilinos_ENABLE_ML:BOOL=ON \
    -D Trilinos_ENABLE_ML_MLAPI:BOOL=ON \
    -D Trilinos_ENABLE_ML_NOX:BOOL=ON \
    -D Trilinos_ENABLE_ML_METIS:BOOL=ON \
    -D Trilinos_ENABLE_ML_PARMETIS3X:BOOL=ON \
    -D Trilinos_ENABLE_ML_SUPERLUDIST:BOOL=ON \
    -D Trilinos_ENABLE_NOX:BOOL=ON \
    -D Trilinos_ENABLE_NOX_EPETRA:BOOL=ON \
    -D Trilinos_ENABLE_NOX_THYRA:BOOL=OFF \
    -D Trilinos_ENABLE_Pamgen:BOOL=ON \
    -D Trilinos_ENABLE_Phalanx:BOOL=ON \
    -D Trilinos_ENABLE_Phdmesh:BOOL=ON \
    -D Trilinos_ENABLE_RTOp:BOOL=ON \
    -D Trilinos_ENABLE_Rythmos:BOOL=OFF \
    -D Trilinos_ENABLE_STK:BOOL=OFF \
    -D Trilinos_ENABLE_Sacado:BOOL=ON \
    -D Trilinos_ENABLE_SEACAS:BOOL=ON \
    -D Trilinos_ENABLE_SEACASExodus:BOOL=ON \
    -D Trilinos_ENABLE_SEACASNemesis:BOOL=OFF \
    -D Trilinos_ENABLE_SEACASExo2mat:BOOL=OFF \
    -D Trilinos_ENABLE_SEACASMat2exo:BOOL=OFF \
    -D Trilinos_ENABLE_Shards:BOOL=ON \
    -D Trilinos_ENABLE_Stokhos:BOOL=OFF \
    -D Trilinos_ENABLE_Stratimikos:BOOL=ON \
    -D Trilinos_ENABLE_Stratimikos_Amesos:BOOL=ON \
    -D Trilinos_ENABLE_Stratimikos_Aztecoo:BOOL=ON \
    -D Trilinos_ENABLE_Stratimikos_Belos:BOOL=ON \
    -D Trilinos_ENABLE_Stratimikos_Ifpack:BOOL=ON \
    -D Trilinos_ENABLE_Stratimikos_Ml:BOOL=ON \
    -D Trilinos_ENABLE_Teuchos:BOOL=ON \
    -D Trilinos_ENABLE_ThreadPool:BOOL=OFF \
    -D Trilinos_ENABLE_Thyra:BOOL=ON \
    -D Trilinos_ENABLE_Tifpack:BOOL=ON \
    -D Trilinos_ENABLE_Tpetra:BOOL=ON \
    -D Trilinos_ENABLE_TriKota:BOOL=OFF \
    -D TriKota_ENABLE_TESTS:BOOL=OFF \
    -D Trilinos_ENABLE_TrilinosCouplings:BOOL=ON \
    -D Trilinos_ENABLE_Teko:BOOL=OFF \
    -D Teko_ENABLE_TESTS:BOOL=OFF \
    -D Trilinos_ENABLE_MueLu:BOOL=ON \
    -D Tpetra_INST_INT_LONG_LONG:BOOL=OFF \
    -D Tpetra_INST_INT_LONG:BOOL=ON \
    -D MueLu_ENABLE_TESTS:BOOL=OFF \
    -D MueLu_ENABLE_EXAMPLES:BOOL=OFF \
    -D MueLu_ENABLE_Experimental:BOOL=ON \
    -D Xpetra_ENABLE_EXAMPLES:BOOL=OFF \
    -D Xpetra_ENABLE_TESTS:BOOL=OFF \
    -D Xpetra_ENABLE_Experimental=ON \
    -D Trilinos_ENABLE_Triutils:BOOL=ON \
    -D Trilinos_ENABLE_Zoltan:BOOL=ON \
    -D Trilinos_ENABLE_Zoltan2:BOOL=OFF \
    -D TPL_ENABLE_Boost:BOOL=ON \
    -D TPL_ENABLE_HDF5:BOOL=OFF \
    -D TPL_ENABLE_MPI:BOOL=ON \
    -D TPL_ENABLE_ParMETIS:BOOL=OFF \
    -D TPL_ENABLE_Pthread:BOOL=ON \
    -D TPL_ENABLE_UMFPACK:BOOL=OFF \
    -D TPL_ENABLE_Netcdf:BOOL=ON \
    -D TPL_Netcdf_PARALLEL:BOOL=OFF \
    -D TPL_ENABLE_MATLAB:BOOL=OFF \
    -D TPL_ENABLE_SuperLU:BOOL=OFF \
    -D TPL_ENABLE_SuperLUDist:BOOL=OFF \
    -D TPL_ENABLE_X11:BOOL=OFF \
    -D EpetraExt_BUILD_BDF:BOOL=ON \
    -D EpetraExt_BUILD_GRAPH_REORDERINGS:BOOL=ON \
    -D EpetraExt_ENABLE_HDF5:BOOL=OFF \
    -D Phdmesh_ENABLE_ExodusII:BOOL=ON \
    -D Phdmesh_ENABLE_Nemesis:BOOL=ON \
    -D Phdmesh_ENABLE_Netcdf:BOOL=ON \
    -D Phdmesh_ENABLE_Pthread:BOOL=ON \
    $EXTRA_ARGS \
    ${TRILINOS}

