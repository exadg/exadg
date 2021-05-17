/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_PETSCOPERATION_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_PETSCOPERATION_H_

#include <deal.II/lac/petsc_vector.h>

namespace ExaDG
{
using namespace dealii;

#ifdef DEAL_II_WITH_PETSC
/*
 * A typedef to make use of PETSc data structure clearer
 */
typedef Vec VectorTypePETSc;

/*
 *  This function wraps the copy of a PETSc object (sparse matrix,
 *  preconditioner) with a dealii::LinearAlgebra::distributed::Vector, taking
 *  pre-allocated PETSc vector objects (with struct name `Vec`, aka
 *  VectorTypePETSc) for the temporary operations
 */
template<typename VectorType>
void
apply_petsc_operation(VectorType &                                           dst,
                      VectorType const &                                     src,
                      VectorTypePETSc &                                      petsc_vector_dst,
                      VectorTypePETSc &                                      petsc_vector_src,
                      std::function<void(PETScWrappers::VectorBase &,
                                         PETScWrappers::VectorBase const &)> petsc_operation)
{
  {
    // copy to PETSc internal vector type because there is currently no such
    // function in deal.II (and the transition via ReadWriteVector is too
    // slow/poorly tested)
    PetscInt       begin, end;
    PetscErrorCode ierr = VecGetOwnershipRange(petsc_vector_src, &begin, &end);
    AssertThrow(ierr == 0, ExcPETScError(ierr));

    PetscScalar * ptr;
    ierr = VecGetArray(petsc_vector_src, &ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));

    const PetscInt local_size = src.get_partitioner()->local_size();
    AssertDimension(local_size, static_cast<unsigned int>(end - begin));
    for(PetscInt i = 0; i < local_size; ++i)
    {
      ptr[i] = src.local_element(i);
    }

    ierr = VecRestoreArray(petsc_vector_src, &ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }

  // wrap `Vec` (aka VectorTypePETSc) into VectorBase (without copying data)
  PETScWrappers::VectorBase petsc_dst(petsc_vector_dst);
  PETScWrappers::VectorBase petsc_src(petsc_vector_src);

  petsc_operation(petsc_dst, petsc_src);

  {
    PetscInt       begin, end;
    PetscErrorCode ierr = VecGetOwnershipRange(petsc_vector_dst, &begin, &end);
    AssertThrow(ierr == 0, ExcPETScError(ierr));

    PetscScalar * ptr;
    ierr = VecGetArray(petsc_vector_dst, &ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));

    const PetscInt local_size = dst.get_partitioner()->local_size();
    AssertDimension(local_size, static_cast<unsigned int>(end - begin));

    for(PetscInt i = 0; i < local_size; ++i)
    {
      dst.local_element(i) = ptr[i];
    }

    ierr = VecRestoreArray(petsc_vector_dst, &ptr);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
  }
}

/*
 *  This function wraps the copy of a PETSc object (sparse matrix,
 *  preconditioner) with a dealii::LinearAlgebra::distributed::Vector,
 *  allocating a PETSc vectors and then calling the other function
 */
template<typename VectorType>
void
apply_petsc_operation(VectorType &                                           dst,
                      VectorType const &                                     src,
                      MPI_Comm const &                                       petsc_mpi_communicator,
                      std::function<void(PETScWrappers::VectorBase &,
                                         PETScWrappers::VectorBase const &)> petsc_operation)
{
  VectorTypePETSc petsc_vector_dst, petsc_vector_src;
  VecCreateMPI(petsc_mpi_communicator,
               dst.get_partitioner()->local_size(),
               PETSC_DETERMINE,
               &petsc_vector_dst);
  VecCreateMPI(petsc_mpi_communicator,
               src.get_partitioner()->local_size(),
               PETSC_DETERMINE,
               &petsc_vector_src);

  apply_petsc_operation(dst, src, petsc_vector_dst, petsc_vector_src, petsc_operation);

  PetscErrorCode ierr = VecDestroy(&petsc_vector_dst);
  AssertThrow(ierr == 0, ExcPETScError(ierr));
  ierr = VecDestroy(&petsc_vector_src);
  AssertThrow(ierr == 0, ExcPETScError(ierr));
}
#endif

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERTDIAGONAL_H_ */
