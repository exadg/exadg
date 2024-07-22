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
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <ml_MultiLevelPreconditioner.h>

#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/utilities/petsc_operation.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
// Rigid body motions for elasticity
template<int dim>
class RigidBodyMotion : public dealii::Function<dim>
{
public:
  RigidBodyMotion(unsigned int const type);

  virtual double
  value(dealii::Point<dim> const & p, unsigned int const component) const override;

private:
  unsigned int const type;
};

template<int dim>
RigidBodyMotion<dim>::RigidBodyMotion(unsigned int const type)
  : dealii::Function<dim>(dim), type(type)
{
  if constexpr(dim == 1)
  {
    AssertThrow(type == 0, dealii::ExcMessage("Requested invalid mode type."));
  }
  else if constexpr(dim == 2)
  {
    AssertThrow(type <= 2, dealii::ExcMessage("Requested invalid mode type."));
  }
  else if constexpr(dim == 3)
  {
    AssertThrow(type <= 5, dealii::ExcMessage("Requested invalid mode type."));
  }
  else
  {
    AssertThrow(dim > 0 and dim < 4, dealii::ExcMessage("Dimension 1 <= dim <= 3 implemented."));
  }
}

template<int dim>
double
RigidBodyMotion<dim>::value(dealii::Point<dim> const & p, unsigned int const component) const
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
    return modes[type];
  }
  else
  {
    // three translations and three 90 degree rotations:
    // [x, y, z] -> [ 0,  z, -y]
    // [x, y, z] -> [-z,  0,  x]
    // [x, y, z] -> [ y, -x,  0]
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
    return modes[type];
  }
}

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

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Operator, typename Number>
class PreconditionerML : public PreconditionerBase<Number>
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::TrilinosWrappers::PreconditionAMG::AdditionalData MLData;

public:
  // distributed sparse system matrix
  dealii::TrilinosWrappers::SparseMatrix system_matrix;

private:
  dealii::TrilinosWrappers::PreconditionAMG amg;

public:
  PreconditionerML(Operator const &                op,
                   bool const                      initialize,
                   AMGOperatorType                 operator_type,
                   dealii::DoFHandler<dim> const & dof_handler,
                   MLData                          ml_data = MLData())
    : pde_operator(op), ml_data(ml_data), operator_type(operator_type), dof_handler(dof_handler)
  {
    // initialize system matrix
    pde_operator.init_system_matrix(system_matrix,
                                    op.get_matrix_free().get_dof_handler().get_communicator());

    if(initialize)
    {
      this->update();
    }
  }

  dealii::TrilinosWrappers::SparseMatrix const &
  get_system_matrix()
  {
    return system_matrix;
  }

  void
  vmult(VectorType & dst, VectorType const & src) const override
  {
    amg.vmult(dst, src);
  }

  void
  update() override
  {
    // clear content of matrix since calculate_system_matrix() adds the result
    system_matrix *= 0.0;

    // re-calculate matrix
    pde_operator.calculate_system_matrix(system_matrix);

    // setup constant modes
    if(operator_type == AMGOperatorType::Ignore)
    {
      // Ignore the operator type, that is, do not setup a near nullspace.
      amg.initialize(system_matrix, ml_data);
    }
    else if(operator_type == AMGOperatorType::Unknown or operator_type == AMGOperatorType::Laplace)
    {
      // Treat AMGOperatorType like Laplace to at least provide trivial nullspace.
      std::vector<std::vector<bool>> constant_modes;
      dealii::DoFTools::extract_constant_modes(dof_handler,
                                               dealii::ComponentMask(),
                                               constant_modes);
      ml_data.constant_modes = constant_modes;

      amg.initialize(system_matrix, ml_data);
    }
    else if(operator_type == AMGOperatorType::Elasticity)
    {
      if(pde_operator.get_matrix_free().get_dof_handler().dimension == 1)
      {
        // Initialize directly, as default is scalar Laplace, which is suitable in this case.
        amg.initialize(system_matrix, ml_data);
      }
      else
      {
        // Translational and rotational rigid body modes need to be provided.
        this->initialize_amg_elasticity();
      }
    }

    this->update_needed = false;
  }

private:
  void
  initialize_amg_elasticity()
  {
    // To provide constant modes for space_dim space dimensions,
    // we need to switch to a Teuchos::ParameterList.
    AssertThrow(dim == dof_handler.space_dimension,
                dealii::ExcMessage("Mixed-dimensional problems are not implemented."));
    AssertThrow(
      dim > 1,
      dealii::ExcMessage(
        "One-dimensional elasticity is identical to scalar Laplace, which is the default."));
    Teuchos::ParameterList parameter_list;
    {
      // This block is copied from deal::PreconditionAMG::AdditionalData::set_parameters(),
      // but does not setup the default constant_modes suitable for, e.g.,
      // a Laplace problem with dim components. Setting up the near nullspace is the most
      // expensive part, which is why we copy paste the code ommitting the last step, which
      // we replace below.
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
      parameter_list.set("coarse: max size", 2000);

      if(ml_data.output_details)
      {
        parameter_list.set("ML output", 10);
      }
      else
      {
        parameter_list.set("ML output", 0);
      }
    }

    // Compute constant modes for elasticity.
    unsigned int constexpr n_constant_modes = dim == 2 ? 3 : 6;
    std::vector<VectorType> constant_modes(n_constant_modes);
    dealii::IndexSet        locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    for(unsigned int i = 0; i < n_constant_modes; ++i)
    {
      constant_modes[i].reinit(dof_handler.locally_owned_dofs(),
                               locally_relevant_dofs,
                               system_matrix.get_mpi_communicator());

      // Fill vector with rigid body modes.
      RigidBodyMotion<dim> const rbm(i);
      dealii::VectorTools::interpolate(dof_handler, rbm, constant_modes[i]);
    }

    // Add constant modes to Teuchos::ParameterList.
    std::unique_ptr<Epetra_MultiVector>
      ptr_distributed_modes; // has to stay alive until after amg.initialize();
    this->set_elasticity_operator_nullspace(parameter_list,
                                            ptr_distributed_modes,
                                            system_matrix.trilinos_matrix(),
                                            constant_modes);

    // Initialize with the Teuchos::ParameterList.
    amg.initialize(system_matrix, parameter_list);
  }

  void
  set_elasticity_operator_nullspace(Teuchos::ParameterList &              parameter_list,
                                    std::unique_ptr<Epetra_MultiVector> & ptr_distributed_modes,
                                    Epetra_RowMatrix const &              matrix,
                                    std::vector<VectorType> const &       constant_modes)
  {
    using size_type               = dealii::TrilinosWrappers::PreconditionAMG::size_type;
    Epetra_Map const & domain_map = matrix.OperatorDomainMap();

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

    size_type const my_size = domain_map.NumMyElements();

    bool const constant_modes_are_global = constant_modes[0].size() == global_size;

    // Reshape null space as a contiguous vector of doubles so that
    // Trilinos can read from it.
    [[maybe_unused]] size_type const expected_mode_size =
      constant_modes_are_global ? global_size : my_size;
    for(size_type i = 0; i < constant_modes.size(); ++i)
    {
      AssertThrow(constant_modes[i].size() == expected_mode_size,
                  dealii::ExcDimensionMismatch(constant_modes[i].size(), expected_mode_size));
      for(size_type row = 0; row < my_size; ++row)
      {
        const dealii::TrilinosWrappers::types::int_type mode_index =
          constant_modes_are_global ? dealii::TrilinosWrappers::global_index(domain_map, row) : row;
        distributed_modes[i][row] = static_cast<double>(constant_modes[i][mode_index]);
      }
    }

    parameter_list.set("null space: type", "pre-computed");
    parameter_list.set("null space: dimension", distributed_modes.NumVectors());
    parameter_list.set("null space: vectors", distributed_modes.Values());
  }

  // reference to matrix-free operator
  Operator const & pde_operator;

  MLData ml_data;

  AMGOperatorType operator_type;

  dealii::DoFHandler<dim> const & dof_handler;
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

  dealii::PETScWrappers::MPI::SparseMatrix const &
  get_system_matrix()
  {
    return system_matrix;
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

  // reference to matrix-free operator
  Operator const & pde_operator;

  BoomerData boomer_data;

  // PETSc vector objects to avoid re-allocation in every vmult() operation
  mutable VectorTypePETSc petsc_vector_src;
  mutable VectorTypePETSc petsc_vector_dst;
};
#endif

/**
 * Implementation of AMG preconditioner unifying PreconditionerML and PreconditionerBoomerAMG.
 */
template<int dim, typename Operator, typename Number>
class PreconditionerAMG : public PreconditionerBase<Number>
{
private:
  typedef double                                                NumberAMG;
  typedef typename PreconditionerBase<Number>::VectorType       VectorType;
  typedef dealii::LinearAlgebra::distributed::Vector<NumberAMG> VectorTypeAMG;

public:
  PreconditionerAMG(Operator const &                pde_operator,
                    bool const                      initialize,
                    AMGData const &                 data,
                    dealii::DoFHandler<dim> const & dof_handler)
  {
    (void)pde_operator;
    (void)initialize;
    (void)data;
    (void)dof_handler;

    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      preconditioner_amg =
        std::make_shared<PreconditionerBoomerAMG<Operator, double>>(pde_operator,
                                                                    initialize,
                                                                    data.boomer_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
    }
    else if(data.amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      preconditioner_amg = std::make_shared<PreconditionerML<dim, Operator, double>>(
        pde_operator, initialize, data.amg_operator_type, dof_handler, data.ml_data);
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
    if constexpr(std::is_same_v<Number, NumberAMG>)
    {
      preconditioner_amg->vmult(dst, src);
    }
    else
    {
      // create temporal vectors of type NumberAMG
      VectorTypeAMG dst_amg;
      dst_amg.reinit(dst, false);
      VectorTypeAMG src_amg;
      src_amg.reinit(src, true);
      src_amg = src;

      preconditioner_amg->vmult(dst_amg, src_amg);

      // convert: NumberAMG -> Number
      dst.copy_locally_owned_data_from(dst_amg);
    }
  }

private:
  void
  update() final
  {
    preconditioner_amg->update();

    this->update_needed = false;
  }

  std::shared_ptr<PreconditionerBase<NumberAMG>> preconditioner_amg;
};

} // namespace ExaDG

#endif
