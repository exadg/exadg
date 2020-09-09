/*
 * multigrid_preconditioner_momentum.cpp
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
MultigridPreconditioner<dim, Number>::MultigridPreconditioner(MPI_Comm const & comm)
  : Base(comm),
    pde_operator(nullptr),
    mg_operator_type(MultigridOperatorType::ReactionDiffusion),
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
  this->pde_operator = &pde_operator;

  this->mg_operator_type = mg_operator_type;

  this->mesh_is_moving = mesh_is_moving;

  data = this->pde_operator->get_data();

  // When solving the reaction-convection-diffusion problem, it might be possible
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
  }
  else if(this->mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    AssertThrow(data.convective_problem == true, ExcMessage("Invalid parameter."));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  Base::initialize(
    mg_data, tria, fe, mapping, false /*operator_is_singular*/, dirichlet_bc, periodic_face_pairs);
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

  this->update_smoothers();

  // singular operators do not occur for this operator
  this->update_coarse_solver(false /* operator_is_singular */);
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

  if(data.unsteady_problem)
    matrix_free_data.append_mapping_flags(MassMatrixKernel<dim, Number>::get_mapping_flags());
  if(data.convective_problem)
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());
  if(data.viscous_problem)
    matrix_free_data.append_mapping_flags(
      Operators::ViscousKernel<dim, Number>::get_mapping_flags(this->level_info[level].is_dg(),
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
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  // initialize pde_operator in a first step
  std::shared_ptr<PDEOperatorMG> pde_operator_level(new PDEOperatorMG());

  data.dof_index  = this->matrix_free_data_objects[level]->get_dof_index("std_dof_handler");
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("std_quadrature");

  pde_operator_level->initialize(*this->matrix_free_objects[level],
                                 *this->constraints[level],
                                 data);

  // make sure that scaling factor of time derivative term has been set before the smoothers are
  // initialized
  pde_operator_level->set_scaling_factor_mass_matrix(
    pde_operator->get_scaling_factor_mass_matrix());

  // initialize MGOperator which is a wrapper around the PDEOperatorMG
  std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator_level));

  return mg_operator;
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update_operators()
{
  if(mesh_is_moving)
  {
    update_operators_after_mesh_movement();
  }

  if(data.unsteady_problem)
  {
    set_time(pde_operator->get_time());
    set_scaling_factor_time_derivative_term(pde_operator->get_scaling_factor_mass_matrix());
  }

  if(mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    VectorType const & vector_linearization = pde_operator->get_velocity();

    // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
    VectorTypeMG         vector_multigrid_type_copy;
    VectorTypeMG const * vector_multigrid_type_ptr;
    if(std::is_same<MultigridNumber, Number>::value)
    {
      vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&vector_linearization);
    }
    else
    {
      vector_multigrid_type_copy = vector_linearization;
      vector_multigrid_type_ptr  = &vector_multigrid_type_copy;
    }

    set_vector_linearization(*vector_multigrid_type_ptr);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_vector_linearization(
  VectorTypeMG const & vector_linearization)
{
  // copy velocity to finest level
  this->get_operator(this->fine_level)->set_velocity_copy(vector_linearization);

  // interpolate velocity from fine to coarse level
  for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
  {
    auto & vector_fine_level   = this->get_operator(level - 0)->get_velocity();
    auto   vector_coarse_level = this->get_operator(level - 1)->get_velocity();
    this->transfers.interpolate(level, vector_coarse_level, vector_fine_level);
    this->get_operator(level - 1)->set_velocity_copy(vector_coarse_level);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_time(double const & time)
{
  for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
  {
    get_operator(level)->set_time(time);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update_operators_after_mesh_movement()
{
  for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
  {
    get_operator(level)->update_after_mesh_movement();
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_scaling_factor_time_derivative_term(
  double const & scaling_factor_time_derivative_term)
{
  for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
  {
    get_operator(level)->set_scaling_factor_mass_matrix(scaling_factor_time_derivative_term);
  }
}

template<int dim, typename Number>
std::shared_ptr<
  MomentumOperator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::get_operator(unsigned int level)
{
  std::shared_ptr<MGOperator> mg_operator =
    std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template class MultigridPreconditioner<2, float>;
template class MultigridPreconditioner<3, float>;

template class MultigridPreconditioner<2, double>;
template class MultigridPreconditioner<3, double>;

} // namespace IncNS
} // namespace ExaDG
