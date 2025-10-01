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

#ifndef EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONERS_PRECONDITIONER_AMG_H_
#define EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONERS_PRECONDITIONER_AMG_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

// Trilinos
#ifdef DEAL_II_WITH_TRILINOS
#  include <ml_MultiLevelPreconditioner.h>
#endif

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>
#include <exadg/solvers_and_preconditioners/utilities/linear_algebra_utilities.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
template<int dim, int spacedim>
std::unique_ptr<MPI_Comm, void (*)(MPI_Comm *)>
create_subcommunicator(dealii::DoFHandler<dim, spacedim> const & dof_handler)
{
  unsigned int n_locally_owned_cells = 0;
  for(auto const & cell : dof_handler.active_cell_iterators())
    if(cell->is_locally_owned())
      ++n_locally_owned_cells;

  MPI_Comm const mpi_comm = dof_handler.get_mpi_communicator();

  // In case some of the MPI ranks do not have cells, we create a
  // sub-communicator to exclude all those processes from the MPI
  // communication in the matrix-based operations and hence speed up those
  // operations. Note that we have to free the communicator again, which is
  // done by a custom deleter of the unique pointer that is run when it goes
  // out of scope.
  if(dealii::Utilities::MPI::min(n_locally_owned_cells, mpi_comm) == 0)
  {
    std::unique_ptr<MPI_Comm, void (*)(MPI_Comm *)> subcommunicator(new MPI_Comm,
                                                                    [](MPI_Comm * comm) {
                                                                      MPI_Comm_free(comm);
                                                                      delete comm;
                                                                    });
    MPI_Comm_split(mpi_comm,
                   n_locally_owned_cells > 0,
                   dealii::Utilities::MPI::this_mpi_process(mpi_comm),
                   subcommunicator.get());

    return subcommunicator;
  }
  else
  {
    std::unique_ptr<MPI_Comm, void (*)(MPI_Comm *)> communicator(new MPI_Comm, [](MPI_Comm * comm) {
      delete comm;
    });
    *communicator = mpi_comm;

    return communicator;
  }
}

#ifdef DEAL_II_WITH_TRILINOS
inline Teuchos::ParameterList
get_ML_parameter_list(dealii::TrilinosWrappers::PreconditionAMG::AdditionalData const & ml_data,
                      int const                                                         dimension)
{
  Teuchos::ParameterList parameter_list;

  // Slightly modified from deal::PreconditionAMG::AdditionalData::set_parameters().
  if(ml_data.elliptic == true)
  {
    ML_Epetra::SetDefaults("SA", parameter_list);
  }
  else
  {
    ML_Epetra::SetDefaults("NSSA", parameter_list);
    parameter_list.set("aggregation: block scaling", true);
  }
  parameter_list.set("aggregation: type", "Uncoupled");
  parameter_list.set("smoother: type", ml_data.smoother_type);
  parameter_list.set("coarse: type", ml_data.coarse_type);

// Force re-initialization of the random seed to make ML deterministic
// (only supported in trilinos >12.2):
#  if DEAL_II_TRILINOS_VERSION_GTE(12, 4, 0)
  parameter_list.set("initialize random seed", true);
#  endif

  parameter_list.set("smoother: sweeps", static_cast<int>(ml_data.smoother_sweeps));
  parameter_list.set("cycle applications", static_cast<int>(ml_data.n_cycles));
  if(ml_data.w_cycle == true)
  {
    parameter_list.set("prec type", "MGW");
  }
  else
  {
    parameter_list.set("prec type", "MGV");
  }

  parameter_list.set("smoother: Chebyshev alpha", 10.);
  parameter_list.set("smoother: ifpack overlap", static_cast<int>(ml_data.smoother_overlap));
  parameter_list.set("aggregation: threshold", ml_data.aggregation_threshold);

  // Minimum size of the coarse problem, i.e. no coarser problems
  // smaller than `coarse: max size` are constructed.
  parameter_list.set("coarse: max size", 2000);

  // This extends the settings in deal::PreconditionAMG::AdditionalData::set_parameters().
  parameter_list.set("repartition: enable", 1);
  parameter_list.set("repartition: max min ratio", 1.3);
  parameter_list.set("repartition: min per proc", 300);
  parameter_list.set("repartition: partitioner", "Zoltan");
  parameter_list.set("repartition: Zoltan dimensions", dimension);

  if(ml_data.output_details)
  {
    parameter_list.set("ML output", 10);
  }
  else
  {
    parameter_list.set("ML output", 0);
  }

  return parameter_list;
}

template<typename Operator>
class PreconditionerML : public PreconditionerBase<double>
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorType;

  typedef dealii::TrilinosWrappers::PreconditionAMG::AdditionalData MLData;

public:
  // distributed sparse system matrix
  dealii::TrilinosWrappers::SparseMatrix system_matrix;

private:
  dealii::TrilinosWrappers::PreconditionAMG amg;

public:
  PreconditionerML(Operator const & op, bool const initialize, MLData ml_data = MLData())
    : pde_operator(op), ml_data(ml_data)
  {
    // initialize system matrix
    pde_operator.init_system_matrix(system_matrix,
                                    op.get_matrix_free().get_dof_handler().get_mpi_communicator());

    if(initialize)
    {
      this->update();
    }
  }

  void
  vmult(VectorType & dst, VectorType const & src) const override
  {
    amg.vmult(dst, src);
  }

  void
  apply_krylov_solver_with_amg_preconditioner(VectorType &                      dst,
                                              VectorType const &                src,
                                              MultigridCoarseGridSolver const & solver_type,
                                              SolverData const &                solver_data) const
  {
    dealii::ReductionControl solver_control(solver_data.max_iter,
                                            solver_data.abs_tol,
                                            solver_data.rel_tol);

    if(solver_type == MultigridCoarseGridSolver::CG)
    {
      dealii::SolverCG<VectorType> solver(solver_control);
      solver.solve(system_matrix, dst, src, *this);
    }
    else if(solver_type == MultigridCoarseGridSolver::GMRES)
    {
      typename dealii::SolverGMRES<VectorType>::AdditionalData gmres_data;
      gmres_data.max_n_tmp_vectors     = solver_data.max_krylov_size;
      gmres_data.right_preconditioning = true;

      dealii::SolverGMRES<VectorType> solver(solver_control, gmres_data);
      solver.solve(system_matrix, dst, src, *this);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  void
  update() override
  {
    // Clear content of matrix since calculate_system_matrix() adds the result.
    system_matrix *= 0.0;

    // Re-calculate the system matrix.
    pde_operator.calculate_system_matrix(system_matrix);

    // Construct AMG preconditioner based on `Teuchos::ParameterList`.
    unsigned int const     dimension      = pde_operator.get_matrix_free().dimension;
    Teuchos::ParameterList parameter_list = get_ML_parameter_list(ml_data, dimension);

    // Add near null space basis vectors to `Teuchos::ParameterList`.
    // If the `std::vector<std::vector<double>> constant_modes_values`
    // were filled, use these, otherwise use `std::vector<std::vector<bool>> constant_modes`.
    std::vector<std::vector<bool>>   constant_modes;
    std::vector<std::vector<double>> constant_modes_values;
    pde_operator.get_constant_modes(constant_modes, constant_modes_values);
    MPI_Comm const & mpi_comm = system_matrix.get_mpi_communicator();
    bool const       some_constant_mode_set =
      dealii::Utilities::MPI::logical_or(constant_modes.size() > 0, mpi_comm);
    if(not some_constant_mode_set)
    {
      bool const some_constant_mode_values_set =
        dealii::Utilities::MPI::logical_or(constant_modes_values.size() > 0, mpi_comm);
      AssertThrow(some_constant_mode_values_set > 0,
                  dealii::ExcMessage(
                    "Neither `constant_modes` nor `constant_modes_values` were provided. "
                    "AMG setup requires near null space basis vectors."));
      ml_data.constant_modes_values = constant_modes_values;
    }
    else
    {
      ml_data.constant_modes = constant_modes;
    }

    // Add near null space basis vectors to Teuchos::ParameterList.
    // `ptr_distributed_modes` must stay alive for amg.initialize()
    std::unique_ptr<Epetra_MultiVector> ptr_operator_modes;
    ml_data.set_operator_null_space(parameter_list,
                                    ptr_operator_modes,
                                    system_matrix.trilinos_matrix());

    // Initialize with the `Teuchos::ParameterList`.
    amg.initialize(system_matrix, parameter_list);

    this->update_needed = false;
  }

private:
  // reference to matrix-free operator
  Operator const & pde_operator;

  MLData ml_data;
};
#endif

#ifdef DEAL_II_WITH_PETSC
/*
 * Wrapper class for BoomerAMG from Hypre
 */
template<typename Operator, typename Number>
class PreconditionerBoomerAMG : public PreconditionerBase<Number>
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData BoomerData;

  // subcommunicator; declared before the matrix to ensure that it gets
  // deleted after the matrix and preconditioner depending on it
  std::unique_ptr<MPI_Comm, void (*)(MPI_Comm *)> subcommunicator;

public:
  // distributed sparse system matrix
  dealii::PETScWrappers::MPI::SparseMatrix system_matrix;

  // amg preconditioner for access by PETSc solver
  dealii::PETScWrappers::PreconditionBoomerAMG amg;

  PreconditionerBoomerAMG(Operator const & op,
                          bool const       initialize,
                          BoomerData       boomer_data = BoomerData())
    : subcommunicator(
        create_subcommunicator(op.get_matrix_free().get_dof_handler(op.get_dof_index()))),
      pde_operator(op),
      boomer_data(boomer_data)
  {
    // initialize system matrix
    pde_operator.init_system_matrix(system_matrix, *subcommunicator);

    if(initialize)
    {
      this->update();
    }
  }

  ~PreconditionerBoomerAMG()
  {
    if(system_matrix.m() > 0)
    {
      PetscErrorCode ierr = VecDestroy(&petsc_vector_dst);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
      ierr = VecDestroy(&petsc_vector_src);
      AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
    }
  }

  void
  vmult(VectorType & dst, VectorType const & src) const override
  {
    if(system_matrix.m() > 0)
      apply_petsc_operation(dst,
                            src,
                            petsc_vector_dst,
                            petsc_vector_src,
                            [&](dealii::PETScWrappers::VectorBase &       petsc_dst,
                                dealii::PETScWrappers::VectorBase const & petsc_src) {
                              amg.vmult(petsc_dst, petsc_src);
                            });
  }

  void
  apply_krylov_solver_with_amg_preconditioner(VectorType &                      dst,
                                              VectorType const &                src,
                                              MultigridCoarseGridSolver const & solver_type,
                                              SolverData const &                solver_data) const
  {
    apply_petsc_operation(dst,
                          src,
                          system_matrix.get_mpi_communicator(),
                          [&](dealii::PETScWrappers::VectorBase &       petsc_dst,
                              dealii::PETScWrappers::VectorBase const & petsc_src) {
                            dealii::ReductionControl solver_control(solver_data.max_iter,
                                                                    solver_data.abs_tol,
                                                                    solver_data.rel_tol);

                            if(solver_type == MultigridCoarseGridSolver::CG)
                            {
                              dealii::PETScWrappers::SolverCG solver(solver_control);
                              solver.solve(system_matrix, petsc_dst, petsc_src, amg);
                            }
                            else if(solver_type == MultigridCoarseGridSolver::GMRES)
                            {
                              dealii::PETScWrappers::SolverGMRES solver(solver_control);
                              solver.solve(system_matrix, petsc_dst, petsc_src, amg);
                            }
                            else
                            {
                              AssertThrow(false, dealii::ExcMessage("Not implemented."));
                            }
                          });
  }

  void
  update() override
  {
    // clear content of matrix since the next calculate_system_matrix calls
    // add their result; since we might run this on a sub-communicator, we
    // skip the processes that do not participate in the matrix and have size
    // zero
    if(system_matrix.m() > 0)
      system_matrix = 0.0;

    calculate_preconditioner();

    this->update_needed = false;
  }

private:
  void
  calculate_preconditioner()
  {
    // calculate_matrix in case the current MPI rank participates in the PETSc communicator
    if(system_matrix.m() > 0)
    {
      pde_operator.calculate_system_matrix(system_matrix);

      amg.initialize(system_matrix, boomer_data);

      // get vector partitioner
      dealii::LinearAlgebra::distributed::Vector<typename Operator::value_type> vector;
      pde_operator.initialize_dof_vector(vector);
      VecCreateMPI(system_matrix.get_mpi_communicator(),
                   vector.get_partitioner()->locally_owned_size(),
                   PETSC_DETERMINE,
                   &petsc_vector_dst);
      VecCreateMPI(system_matrix.get_mpi_communicator(),
                   vector.get_partitioner()->locally_owned_size(),
                   PETSC_DETERMINE,
                   &petsc_vector_src);
    }
  }

  // reference to MultigridOperator
  Operator const & pde_operator;

  BoomerData boomer_data;

  // PETSc vector objects to avoid re-allocation in every vmult() operation
  mutable Vec petsc_vector_src;
  mutable Vec petsc_vector_dst;
};
#endif

/**
 * Implementation of AMG preconditioner unifying PreconditionerML and PreconditionerBoomerAMG.
 */
template<typename Operator, typename Number>
class PreconditionerAMG : public PreconditionerBase<Number>
{
private:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

public:
  PreconditionerAMG(Operator const & pde_operator, bool const initialize, AMGData const & data)
  {
    (void)pde_operator;
    (void)initialize;
    this->data = data;

    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      preconditioner_boomer =
        std::make_shared<PreconditionerBoomerAMG<Operator, Number>>(pde_operator,
                                                                    initialize,
                                                                    data.boomer_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
    }
    else if(data.amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      preconditioner_ml =
        std::make_shared<PreconditionerML<Operator>>(pde_operator, initialize, data.ml_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      preconditioner_boomer->vmult(dst, src);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
    }
    else if(data.amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      apply_function_in_double_precision(
        dst,
        src,
        [&](dealii::LinearAlgebra::distributed::Vector<double> &       dst_double,
            dealii::LinearAlgebra::distributed::Vector<double> const & src_double) {
          preconditioner_ml->vmult(dst_double, src_double);
        });
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  apply_krylov_solver_with_amg_preconditioner(VectorType &                      dst,
                                              VectorType const &                src,
                                              MultigridCoarseGridSolver const & solver_type,
                                              SolverData const &                solver_data) const
  {
    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      std::shared_ptr<PreconditionerBoomerAMG<Operator, Number>> preconditioner =
        std::dynamic_pointer_cast<PreconditionerBoomerAMG<Operator, Number>>(preconditioner_boomer);

      preconditioner->apply_krylov_solver_with_amg_preconditioner(dst,
                                                                  src,
                                                                  solver_type,
                                                                  solver_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
    }
    else if(data.amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      std::shared_ptr<PreconditionerML<Operator>> preconditioner =
        std::dynamic_pointer_cast<PreconditionerML<Operator>>(preconditioner_ml);

      apply_function_in_double_precision(
        dst,
        src,
        [&](dealii::LinearAlgebra::distributed::Vector<double> &       dst_double,
            dealii::LinearAlgebra::distributed::Vector<double> const & src_double) {
          preconditioner->apply_krylov_solver_with_amg_preconditioner(dst_double,
                                                                      src_double,
                                                                      solver_type,
                                                                      solver_data);
        });
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }

  void
  update() final
  {
    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      preconditioner_boomer->update();
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
    }
    else if(data.amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      preconditioner_ml->update();
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    this->update_needed = false;
  }

private:
  AMGData data;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner_boomer;

  std::shared_ptr<PreconditionerBase<double>> preconditioner_ml;
};

} // namespace ExaDG

#endif /* EXADG_SOLVERS_AND_PRECONDITIONERS_PRECONDITIONERS_PRECONDITIONER_AMG_H_ */
