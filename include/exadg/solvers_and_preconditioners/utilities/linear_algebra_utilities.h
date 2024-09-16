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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_LINEAR_ALGEBRA_UTILITIES_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_LINEAR_ALGEBRA_UTILITIES_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_vector.h>

namespace ExaDG
{
#ifdef DEAL_II_WITH_PETSC
/*
 *  This function wraps the copy of a PETSc object (sparse matrix,
 *  preconditioner) with a dealii::LinearAlgebra::distributed::Vector, taking
 *  pre-allocated PETSc vector objects (with struct name `Vec`) for the temporary operations
 */
template<typename VectorType>
void
apply_petsc_operation(
  VectorType &                                                   dst,
  VectorType const &                                             src,
  Vec &                                                          petsc_vector_dst,
  Vec &                                                          petsc_vector_src,
  std::function<void(dealii::PETScWrappers::VectorBase &,
                     dealii::PETScWrappers::VectorBase const &)> petsc_operation)
{
  {
    // copy to PETSc internal vector type because there is currently no such
    // function in deal.II (and the transition via ReadWriteVector is too
    // slow/poorly tested)
    PetscInt       begin, end;
    PetscErrorCode ierr = VecGetOwnershipRange(petsc_vector_src, &begin, &end);
    AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

    PetscScalar * ptr;
    ierr = VecGetArray(petsc_vector_src, &ptr);
    AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

    const PetscInt local_size = src.get_partitioner()->locally_owned_size();
    AssertDimension(local_size, static_cast<unsigned int>(end - begin));
    for(PetscInt i = 0; i < local_size; ++i)
    {
      ptr[i] = src.local_element(i);
    }

    ierr = VecRestoreArray(petsc_vector_src, &ptr);
    AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  }

  // wrap `Vec` into VectorBase (without copying data)
  dealii::PETScWrappers::VectorBase petsc_dst(petsc_vector_dst);
  dealii::PETScWrappers::VectorBase petsc_src(petsc_vector_src);

  petsc_operation(petsc_dst, petsc_src);

  {
    PetscInt       begin, end;
    PetscErrorCode ierr = VecGetOwnershipRange(petsc_vector_dst, &begin, &end);
    AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

    PetscScalar * ptr;
    ierr = VecGetArray(petsc_vector_dst, &ptr);
    AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));

    const PetscInt local_size = dst.get_partitioner()->locally_owned_size();
    AssertDimension(local_size, static_cast<unsigned int>(end - begin));

    for(PetscInt i = 0; i < local_size; ++i)
    {
      dst.local_element(i) = ptr[i];
    }

    ierr = VecRestoreArray(petsc_vector_dst, &ptr);
    AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  }
}

/*
 *  This function wraps the copy of a PETSc object (sparse matrix,
 *  preconditioner) with a dealii::LinearAlgebra::distributed::Vector,
 *  allocating PETSc vectors and then calling the other function
 */
template<typename VectorType>
void
apply_petsc_operation(
  VectorType &                                                   dst,
  VectorType const &                                             src,
  MPI_Comm const &                                               petsc_mpi_communicator,
  std::function<void(dealii::PETScWrappers::VectorBase &,
                     dealii::PETScWrappers::VectorBase const &)> petsc_operation)
{
  Vec petsc_vector_dst, petsc_vector_src;
  VecCreateMPI(petsc_mpi_communicator,
               dst.get_partitioner()->locally_owned_size(),
               PETSC_DETERMINE,
               &petsc_vector_dst);
  VecCreateMPI(petsc_mpi_communicator,
               src.get_partitioner()->locally_owned_size(),
               PETSC_DETERMINE,
               &petsc_vector_src);

  apply_petsc_operation(dst, src, petsc_vector_dst, petsc_vector_src, petsc_operation);

  PetscErrorCode ierr = VecDestroy(&petsc_vector_dst);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  ierr = VecDestroy(&petsc_vector_src);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
}
#endif


/**
 * Certain functionality of e.g. Trilinos is only available in double precision. The aim of this
 * function is to hide this circumstance from the user code (that uses the template parameter
 * Number) and to perform Number -> double conversions and then call the double-version of a certain
 * Trilinos functionality.
 */
template<typename Number>
void
apply_function_in_double_precision(
  dealii::LinearAlgebra::distributed::Vector<Number> &                            dst,
  dealii::LinearAlgebra::distributed::Vector<Number> const &                      src,
  std::function<void(dealii::LinearAlgebra::distributed::Vector<double> &,
                     dealii::LinearAlgebra::distributed::Vector<double> const &)> operation)
{
  if constexpr(std::is_same_v<Number, double>)
  {
    operation(dst, src);
  }
  else
  {
    // create temporal vectors of type double
    dealii::LinearAlgebra::distributed::Vector<double> dst_double, src_double;
    dst_double.reinit(dst, true); // do not zero entries
    src_double.reinit(src, true); // do not zero entries
    src_double.copy_locally_owned_data_from(src);

    operation(dst_double, src_double);

    // convert: double -> Number
    dst.copy_locally_owned_data_from(dst_double);
  }
}

} // namespace ExaDG

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_LINEAR_ALGEBRA_UTILITIES_H_ */
