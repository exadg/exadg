/*
 * multigrid_preconditioner.cpp
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

// ExaDG
#include <exadg/convection_diffusion/preconditioners/multigrid_preconditioner.h>
#include <exadg/operators/mapping_flags.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim, typename Number>
MultigridPreconditioner<dim, Number>::MultigridPreconditioner(MPI_Comm const & mpi_comm)
  : Base(mpi_comm),
    pde_operator(nullptr),
    mg_operator_type(MultigridOperatorType::Undefined),
    mesh_is_moving(false)
{
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize(MultigridData const &                    mg_data,
                                                 parallel::TriangulationBase<dim> const * tria,
                                                 FiniteElement<dim> const &               fe,
                                                 Mapping<dim> const &                     mapping,
                                                 PDEOperator const &           pde_operator,
                                                 MultigridOperatorType const & mg_operator_type,
                                                 bool const                    mesh_is_moving,
                                                 Map const *                   dirichlet_bc,
                                                 PeriodicFacePairs *           periodic_face_pairs)
{
  this->pde_operator     = &pde_operator;
  this->mg_operator_type = mg_operator_type;
  this->mesh_is_moving   = mesh_is_moving;

  data = this->pde_operator->get_data();

  // When solving the reaction-convection-diffusion equations, it might be possible
  // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
  // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
  // reaction-convection-diffusion operator. Accordingly, we have to reset which
  // operators should be "active" for the multigrid preconditioner, independently of
  // the actual equation type that is solved.
  AssertThrow(this->mg_operator_type != MultigridOperatorType::Undefined,
              ExcMessage("Invalid parameter mg_operator_type."));

  if(this->mg_operator_type == MultigridOperatorType::ReactionDiffusion)
  {
    // deactivate convective term for multigrid preconditioner
    data.convective_problem = false;
    data.diffusive_problem  = true;
  }
  else if(this->mg_operator_type == MultigridOperatorType::ReactionConvection)
  {
    data.convective_problem = true;
    // deactivate viscous term for multigrid preconditioner
    data.diffusive_problem = false;
  }
  else if(this->mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    data.convective_problem = true;
    data.diffusive_problem  = true;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  Base::initialize(
    mg_data, tria, fe, mapping, data.operator_is_singular, dirichlet_bc, periodic_face_pairs);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update()
{
  if(mesh_is_moving)
  {
    this->update_matrix_free();
  }

  update_operators();

  update_smoothers();

  this->update_coarse_solver(data.operator_is_singular);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level)
{
  matrix_free_data.data.mg_level = this->level_info[level].h_level();
  matrix_free_data.data.tasks_parallel_scheme =
    MatrixFree<dim, MultigridNumber>::AdditionalData::none;

  MappingFlags flags;
  if(data.unsteady_problem)
    matrix_free_data.append_mapping_flags(MassMatrixKernel<dim, Number>::get_mapping_flags());
  if(data.convective_problem)
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());
  if(data.diffusive_problem)
    matrix_free_data.append_mapping_flags(
      Operators::DiffusiveKernel<dim, Number>::get_mapping_flags(this->level_info[level].is_dg(),
                                                                 this->level_info[level].is_dg()));

  if(data.use_cell_based_loops && this->level_info[level].is_dg())
  {
    auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
      &this->dof_handlers[level]->get_triangulation());
    Categorization::do_cell_based_loops(*tria,
                                        matrix_free_data.data,
                                        this->level_info[level].h_level());
  }

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "std_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "std_dof_handler");
  matrix_free_data.insert_quadrature(QGauss<1>(this->level_info[level].degree() + 1),
                                     "std_quadrature");

  if(data.convective_problem)
  {
    if(data.convective_kernel_data.velocity_type == TypeVelocityField::Function)
    {
      // do nothing
    }
    else if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
    {
      matrix_free_data.insert_dof_handler(&(*dof_handlers_velocity[level]), "velocity_dof_handler");
      matrix_free_data.insert_constraint(&(*constraints_velocity[level]), "velocity_dof_handler");
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  // initialize pde_operator in a first step
  std::shared_ptr<PDEOperatorMG> pde_operator_level(new PDEOperatorMG());

  // set dof and quad indices after matrix_free_data has been filled
  data.dof_index = this->matrix_free_data_objects[level]->get_dof_index("std_dof_handler");
  if(data.convective_problem &&
     data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
  {
    data.convective_kernel_data.dof_index_velocity =
      this->matrix_free_data_objects[level]->get_dof_index("velocity_dof_handler");
  }
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("std_quadrature");

  pde_operator_level->initialize(*this->matrix_free_objects[level],
                                 *this->constraints[level],
                                 data);

  // make sure that scaling factor of time derivative term has been set before the smoothers are
  // initialized
  pde_operator_level->set_scaling_factor_mass_matrix(
    pde_operator->get_scaling_factor_mass_matrix());

  // initialize MGOperator which is a wrapper around the PDEOperatorMG
  std::shared_ptr<MGOperator> mg_operator_level(new MGOperator(pde_operator_level));

  return mg_operator_level;
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize_dof_handler_and_constraints(
  bool const                               operator_is_singular,
  PeriodicFacePairs *                      periodic_face_pairs,
  FiniteElement<dim> const &               fe,
  parallel::TriangulationBase<dim> const * tria,
  Map const *                              dirichlet_bc)
{
  Base::initialize_dof_handler_and_constraints(
    operator_is_singular, periodic_face_pairs, fe, tria, dirichlet_bc);

  if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
  {
    FESystem<dim> fe_velocity(FE_DGQ<dim>(fe.degree), dim);
    Map           dirichlet_bc_velocity;
    this->do_initialize_dof_handler_and_constraints(false,
                                                    *periodic_face_pairs,
                                                    fe_velocity,
                                                    tria,
                                                    dirichlet_bc_velocity,
                                                    this->level_info,
                                                    this->p_levels,
                                                    this->dof_handlers_velocity,
                                                    this->constrained_dofs_velocity,
                                                    this->constraints_velocity);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize_transfer_operators()
{
  Base::initialize_transfer_operators();

  if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
    this->transfers_velocity.reinit(*this->mapping,
                                    this->matrix_free_objects,
                                    this->constraints_velocity,
                                    this->constrained_dofs_velocity,
                                    1);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update_operators()
{
  if(mesh_is_moving)
  {
    update_operators_after_mesh_movement();
  }

  set_time(pde_operator->get_time());
  set_scaling_factor_mass_matrix(pde_operator->get_scaling_factor_mass_matrix());

  TypeVelocityField velocity_type = data.convective_kernel_data.velocity_type;

  if(velocity_type == TypeVelocityField::DoFVector &&
     (mg_operator_type == MultigridOperatorType::ReactionConvection ||
      mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion))
  {
    VectorType const & velocity = pde_operator->get_velocity();

    // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
    VectorTypeMG         vector_multigrid_type_copy;
    VectorTypeMG const * vector_multigrid_type_ptr;
    if(std::is_same<MultigridNumber, Number>::value)
    {
      vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&velocity);
    }
    else
    {
      vector_multigrid_type_copy = velocity;
      vector_multigrid_type_ptr  = &vector_multigrid_type_copy;
    }

    set_velocity(*vector_multigrid_type_ptr);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_velocity(VectorTypeMG const & velocity)
{
  // copy velocity to finest level
  this->get_operator(this->fine_level)->set_velocity_copy(velocity);

  // interpolate velocity from fine to coarse level
  for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
  {
    auto & vector_fine_level   = this->get_operator(level - 0)->get_velocity();
    auto   vector_coarse_level = this->get_operator(level - 1)->get_velocity();
    transfers_velocity.interpolate(level, vector_coarse_level, vector_fine_level);
    this->get_operator(level - 1)->set_velocity_copy(vector_coarse_level);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update_operators_after_mesh_movement()
{
  for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
  {
    this->get_operator(level)->update_after_mesh_movement();
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_time(double const & time)
{
  for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    this->get_operator(level)->set_time(time);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_scaling_factor_mass_matrix(double const & scaling_factor)
{
  for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    this->get_operator(level)->set_scaling_factor_mass_matrix(scaling_factor);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update_smoothers()
{
  // Skip coarsest level
  for(unsigned int level = this->coarse_level + 1; level <= this->fine_level; ++level)
    this->update_smoother(level);
}

template<int dim, typename Number>
std::shared_ptr<Operator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::get_operator(unsigned int level) const
{
  std::shared_ptr<MGOperator> mg_operator =
    std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template class MultigridPreconditioner<2, float>;
template class MultigridPreconditioner<3, float>;

template class MultigridPreconditioner<2, double>;
template class MultigridPreconditioner<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
