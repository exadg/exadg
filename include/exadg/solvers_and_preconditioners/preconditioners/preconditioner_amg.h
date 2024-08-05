/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors and Marco Feder
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

#ifndef PRECONDITIONER_AMG
#define PRECONDITIONER_AMG

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ml_MultiLevelPreconditioner.h>

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

  MPI_Comm const mpi_comm = dof_handler.get_communicator();

  // In case some of the MPI ranks do not have cells, we create a
  // sub-communicator to exclude all those processes from the MPI
  // communication in the matrix-based operation sand hence speed up those
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

// Constant modes for scalar/vector Laplace and Elasticity.
template<int dim>
class ConstantModes : public dealii::Function<dim>
{
public:
  ConstantModes(AMGOperatorType operator_type, unsigned int const mode_index);

  virtual double
  value(dealii::Point<dim> const & p, unsigned int const component) const override;

private:
  AMGOperatorType    amg_operator_type;
  unsigned int const mode_index;
};

template<int dim>
ConstantModes<dim>::ConstantModes(AMGOperatorType amg_operator_type, unsigned int const mode_index)
  : dealii::Function<dim>(dim), amg_operator_type(amg_operator_type), mode_index(mode_index)
{
  if(amg_operator_type == AMGOperatorType::ScalarLaplace)
  {
    AssertThrow(mode_index == 0,
                dealii::ExcMessage("AMGOperatorType::ScalarLaplace has one constant mode."));
  }
  else if(amg_operator_type == AMGOperatorType::VectorLaplace)
  {
    AssertThrow(mode_index < dim,
                dealii::ExcMessage("AMGOperatorType::VectorLaplace has three constant modes."));
  }
  else if(amg_operator_type == AMGOperatorType::Elasticity)
  {
    if constexpr(dim == 1)
    {
      AssertThrow(mode_index == 0,
                  dealii::ExcMessage("AMGOperatorType::Elasticity has one constant mode in 3D."));
    }
    else if constexpr(dim == 2)
    {
      AssertThrow(mode_index < 3,
                  dealii::ExcMessage(
                    "AMGOperatorType::Elasticity has three constant modes in 2D."));
    }
    else
    {
      AssertThrow(mode_index < 6,
                  dealii::ExcMessage("AMGOperatorType::Elasticity has six constant modes in 3D."));
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("AMGOperatorType not defined."));
  }
}

template<int dim>
double
ConstantModes<dim>::value(dealii::Point<dim> const & p, unsigned int const component) const
{
  if(amg_operator_type == AMGOperatorType::ScalarLaplace)
  {
    return 1.0;
  }
  else if(amg_operator_type == AMGOperatorType::VectorLaplace)
  {
    std::array<double, 3> const modes{{static_cast<double>(component == 0),
                                       static_cast<double>(component == 1),
                                       static_cast<double>(component == 2)}};
    return modes[mode_index];
  }
  else if(amg_operator_type == AMGOperatorType::Elasticity)
  {
    if constexpr(dim == 1)
    {
      return 1.0;
    }
    else if constexpr(dim == 2)
    {
      // two translations and
      // 90 degree rotation: [x y] -> [y -x]
      std::array<double, 3> const modes{{static_cast<double>(component == 0),
                                         static_cast<double>(component == 1),
                                         (component == 0) ? p[1] : -p[0]}};
      return modes[mode_index];
    }
    else
    {
      // [ 0,  z, -y]
      // [-z,  0,  x]
      // [ y, -x,  0]
      // see [Baker et al., Numer. Linear Algebra Appl. 17, 2010]
      // https://onlinelibrary.wiley.com/doi/epdf/10.1002/nla.688
      std::array<double, 6> const modes{{static_cast<double>(component == 0),
                                         static_cast<double>(component == 1),
                                         static_cast<double>(component == 2),
                                         (component == 0) ? 0.0 :
                                         (component == 1) ? p[2] :
                                                            -p[1],
                                         (component == 0) ? -p[2] :
                                         (component == 1) ? 0.0 :
                                                            p[0],
                                         (component == 0) ? p[1] :
                                         (component == 1) ? -p[0] :
                                                            0.0}};
      return modes[mode_index];
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("AMGOperatorType not defined."));
    return 0.0;
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Operator>
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
  PreconditionerML(Operator const &             op,
                   bool const                   initialize,
                   AMGOperatorType              operator_type,
                   dealii::Mapping<dim> const & mapping,
                   MLData                       ml_data = MLData())
    : pde_operator(op), ml_data(ml_data), operator_type(operator_type), mapping(mapping)
  {
    // initialize system matrix
    pde_operator.init_system_matrix(system_matrix,
                                    op.get_matrix_free().get_dof_handler().get_communicator());

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
    // clear content of matrix since calculate_system_matrix() adds the result
    system_matrix *= 0.0;

    // re-calculate matrix
    pde_operator.calculate_system_matrix(system_matrix);

    if(operator_type == AMGOperatorType::Default)
    {
      // Default setup.
      amg.initialize(system_matrix, ml_data);
    }
    else
    {
      // get Teuchos::ParameterList to provide custom near null space basis
      Teuchos::ParameterList parameter_list = get_parameter_list();

      std::vector<VectorType> constant_modes = get_constant_modes();

      // Add constant modes to Teuchos::ParameterList.
      std::unique_ptr<Epetra_MultiVector>
        ptr_distributed_modes; // has to stay alive until after amg.initialize();
      set_operator_nullspace(parameter_list,
                             ptr_distributed_modes,
                             system_matrix.trilinos_matrix(),
                             constant_modes);

      // Initialize with the Teuchos::ParameterList.
      amg.initialize(system_matrix, parameter_list);
    }

    this->update_needed = false;
  }

private:
  Teuchos::ParameterList
  get_parameter_list()
  {
    Teuchos::ParameterList parameter_list;

    // Slightly modified from deal::PreconditionAMG::AdditionalData::set_parameters().
    if(ml_data.elliptic == true)
    {
      ML_Epetra::SetDefaults("SA", parameter_list);
      if(ml_data.higher_order_elements)
      {
        parameter_list.set("aggregation: type", "Uncoupled");
      }
    }
    else
    {
      ML_Epetra::SetDefaults("NSSA", parameter_list);
      parameter_list.set("aggregation: type", "Uncoupled");
      parameter_list.set("aggregation: block scaling", true);
    }

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
    parameter_list.set("repartition: Zoltan dimensions", static_cast<int>(dim));

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

  std::vector<VectorType>
  get_constant_modes()
  {
    unsigned int n_constant_modes = dealii::numbers::invalid_unsigned_int;
    if(operator_type == AMGOperatorType::ScalarLaplace)
    {
      n_constant_modes = 1;
    }
    else if(operator_type == AMGOperatorType::VectorLaplace)
    {
      n_constant_modes = dim;
    }
    else if(operator_type == AMGOperatorType::Elasticity)
    {
      n_constant_modes = dim == 1 ? 1 : (dim == 2 ? 3 : 6);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("AMGOperatorType not defined"));
    }

    std::vector<VectorType> constant_modes(n_constant_modes);

    // The AMG Preconditioner might be used as a coarse-level solver/preconditioner within
    // multigrid.
    bool const on_coarse_grid = pde_operator.get_matrix_free().get_dof_handler().has_level_dofs();

    for(unsigned int i = 0; i < n_constant_modes; ++i)
    {
      constant_modes[i].reinit(pde_operator.get_matrix_free().get_vector_partitioner(),
                               system_matrix.get_mpi_communicator());

      // Fill vector with rigid body modes.
      ConstantModes<dim> const mode_generator(operator_type, i);
      dealii::VectorTools::interpolate(mapping,
                                       pde_operator.get_matrix_free().get_dof_handler(),
                                       mode_generator,
                                       constant_modes[i],
                                       dealii::ComponentMask(),
                                       on_coarse_grid ? 0 : dealii::numbers::invalid_unsigned_int);
    }

    return constant_modes;
  }

  void
  set_operator_nullspace(Teuchos::ParameterList &              parameter_list,
                         std::unique_ptr<Epetra_MultiVector> & ptr_distributed_modes,
                         Epetra_RowMatrix const &              matrix,
                         std::vector<VectorType> const &       constant_modes)
  {
    using size_type               = dealii::TrilinosWrappers::PreconditionAMG::size_type;
    Epetra_Map const & domain_map = matrix.OperatorDomainMap();
    size_type const    my_size    = domain_map.NumMyElements();

    // Avoid Trilinos ML warning when vectors are empty on threads with no owned rows.
    if(my_size > 0)
    {
      AssertThrow(constant_modes.size() > 0,
                  dealii::ExcMessage("Provide constant modes for near null space"));
      ptr_distributed_modes.reset(new Epetra_MultiVector(domain_map, constant_modes.size()));
      AssertThrow(ptr_distributed_modes, dealii::ExcNotInitialized());
      Epetra_MultiVector & distributed_modes = *ptr_distributed_modes;

      size_type const global_size = dealii::TrilinosWrappers::n_global_rows(matrix);

      AssertThrow(global_size == static_cast<size_type>(
                                   dealii::TrilinosWrappers::global_length(distributed_modes)),
                  dealii::ExcDimensionMismatch(
                    global_size, dealii::TrilinosWrappers::global_length(distributed_modes)));

      // Reshape null space as a contiguous vector of doubles so that
      // Trilinos can read from it.
      [[maybe_unused]] size_type const expected_mode_size = global_size;
      for(size_type i = 0; i < constant_modes.size(); ++i)
      {
        AssertThrow(constant_modes[i].size() == expected_mode_size,
                    dealii::ExcDimensionMismatch(constant_modes[i].size(), expected_mode_size));
        for(size_type row = 0; row < my_size; ++row)
        {
          distributed_modes[i][row] = static_cast<double>(
            constant_modes[i][dealii::TrilinosWrappers::global_index(domain_map, row)]);
        }
      }

      parameter_list.set("null space: type", "pre-computed");
      parameter_list.set("null space: dimension", distributed_modes.NumVectors());
      parameter_list.set("null space: vectors", distributed_modes.Values());
    }
  }

  // reference to matrix-free operator
  Operator const & pde_operator;

  MLData ml_data;

  AMGOperatorType operator_type;

  dealii::Mapping<dim> const & mapping;
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
template<int dim, typename Operator, typename Number>
class PreconditionerAMG : public PreconditionerBase<Number>
{
private:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

public:
  PreconditionerAMG(Operator const &             pde_operator,
                    bool const                   initialize,
                    AMGData const &              data,
                    dealii::Mapping<dim> const & mapping)
  {
    (void)pde_operator;
    (void)initialize;
    (void)mapping;
    this->data = data;

    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      AssertThrow(data.amg_operator_type == AMGOperatorType::Default,
                  dealii::ExcMessage("AMGType::BoomerAMG incorporates "
                                     "only AMGOperatorType::Default."));
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
      preconditioner_ml = std::make_shared<PreconditionerML<dim, Operator>>(
        pde_operator, initialize, data.amg_operator_type, mapping, data.ml_data);
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
      std::shared_ptr<PreconditionerML<dim, Operator>> preconditioner =
        std::dynamic_pointer_cast<PreconditionerML<dim, Operator>>(preconditioner_ml);

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

#endif
