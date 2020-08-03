/*
 * compatible_laplace_multigrid_preconditioner.cpp
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

#include "compatible_laplace_multigrid_preconditioner.h"

#include <deal.II/fe/fe_system.h>

namespace IncNS
{
template<int dim, typename Number>
CompatibleLaplaceMultigridPreconditioner<dim, Number>::CompatibleLaplaceMultigridPreconditioner(
  MPI_Comm const & mpi_comm)
  : Base(mpi_comm), mesh_is_moving(false)
{
}

template<int dim, typename Number>
void
CompatibleLaplaceMultigridPreconditioner<dim, Number>::initialize(
  MultigridData const &                      mg_data,
  const parallel::TriangulationBase<dim> *   tria,
  const FiniteElement<dim> &                 fe,
  Mapping<dim> const &                       mapping,
  CompatibleLaplaceOperatorData<dim> const & data_in,
  bool const                                 mesh_is_moving,
  Map const *                                dirichlet_bc,
  PeriodicFacePairs *                        periodic_face_pairs)
{
  data = data_in;

  this->mesh_is_moving = mesh_is_moving;

  Base::initialize(
    mg_data, tria, fe, mapping, data.operator_is_singular, dirichlet_bc, periodic_face_pairs);
}

template<int dim, typename Number>
void
CompatibleLaplaceMultigridPreconditioner<dim, Number>::update()
{
  // update of this multigrid preconditioner is only needed
  // if the mesh is moving
  if(mesh_is_moving)
  {
    this->update_matrix_free();

    this->update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(data.operator_is_singular);
  }
}

template<int dim, typename Number>
void
CompatibleLaplaceMultigridPreconditioner<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level)
{
  matrix_free_data.data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  if(this->level_info[level].is_dg())
  {
    matrix_free_data.data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    matrix_free_data.data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
  }

  matrix_free_data.data.mg_level = this->level_info[level].h_level();

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "pressure_dof_handler");
  matrix_free_data.insert_dof_handler(&(*this->dof_handlers_velocity[level]),
                                      "velocity_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "std_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints_velocity[level]), "velocity_dof_handler");
  // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature
  // formula that is used for the gradient operator and the divergence operator (and the inverse
  // velocity mass matrix operator)
  matrix_free_data.insert_quadrature(QGauss<1>(this->level_info[level].degree() + 1 +
                                               (data.degree_u - data.degree_p)),
                                     "velocity_quadrature");
  // quadrature formula with (fe_degree_p+1) quadrature points: this is the quadrature
  // that is needed for p-transfer
  matrix_free_data.insert_quadrature(QGauss<1>(this->level_info[level].degree() + 1),
                                     "pressure_quadrature");
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
CompatibleLaplaceMultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  data.dof_index_velocity =
    this->matrix_free_data_objects[level]->get_dof_index("velocity_dof_handler");
  data.dof_index_pressure =
    this->matrix_free_data_objects[level]->get_dof_index("pressure_dof_handler");
  data.quad_index_velocity =
    this->matrix_free_data_objects[level]->get_quad_index("velocity_quadrature");

  data.gradient_operator_data.dof_index_velocity = data.dof_index_velocity;
  data.gradient_operator_data.dof_index_pressure = data.dof_index_pressure;
  data.gradient_operator_data.quad_index         = data.quad_index_velocity;

  data.divergence_operator_data.dof_index_velocity = data.dof_index_velocity;
  data.divergence_operator_data.dof_index_pressure = data.dof_index_pressure;
  data.divergence_operator_data.quad_index         = data.quad_index_velocity;

  // initialize pde_operator in a first step
  std::shared_ptr<PDEOperatorMG> pde_operator(new PDEOperatorMG());
  pde_operator->initialize(*this->matrix_free_objects[level], *this->constraints[level], data);

  // initialize MGOperator which is a wrapper around the PDEOperatorMG
  std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

  return mg_operator;
}


template<int dim, typename Number>
void
CompatibleLaplaceMultigridPreconditioner<dim, Number>::initialize_dof_handler_and_constraints(
  bool const                               operator_is_singular,
  PeriodicFacePairs *                      periodic_face_pairs,
  FiniteElement<dim> const &               fe,
  parallel::TriangulationBase<dim> const * tria,
  Map const *                              dirichlet_bc)
{
  Base::initialize_dof_handler_and_constraints(
    operator_is_singular, periodic_face_pairs, fe, tria, dirichlet_bc);

  // do setup required for derived class

  std::vector<MGLevelInfo>            level_info_velocity;
  std::vector<MGDoFHandlerIdentifier> p_levels_velocity;

  // setup global velocity levels
  for(auto & level : this->level_info)
    level_info_velocity.push_back(
      {level.h_level(), level.degree() + data.degree_u - data.degree_p, level.is_dg()});

  // setup p velocity levels
  for(auto level : level_info_velocity)
    p_levels_velocity.push_back(level.dof_handler_id());

  sort(p_levels_velocity.begin(), p_levels_velocity.end());
  p_levels_velocity.erase(unique(p_levels_velocity.begin(), p_levels_velocity.end()),
                          p_levels_velocity.end());
  std::reverse(std::begin(p_levels_velocity), std::end(p_levels_velocity));

  // setup dofhandler and constraint matrices
  FE_DGQ<dim>   temp(data.degree_u);
  FESystem<dim> fe_velocity(temp, dim);

  Map dirichlet_bc_velocity;
  this->do_initialize_dof_handler_and_constraints(false,
                                                  *periodic_face_pairs,
                                                  fe_velocity,
                                                  tria,
                                                  dirichlet_bc_velocity,
                                                  level_info_velocity,
                                                  p_levels_velocity,
                                                  this->dof_handlers_velocity,
                                                  this->constrained_dofs_velocity,
                                                  this->constraints_velocity);
}


template<int dim, typename Number>
std::shared_ptr<
  CompatibleLaplaceOperator<dim,
                            typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
CompatibleLaplaceMultigridPreconditioner<dim, Number>::get_operator(unsigned int level)
{
  std::shared_ptr<MGOperator> mg_operator =
    std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template class CompatibleLaplaceMultigridPreconditioner<2, float>;
template class CompatibleLaplaceMultigridPreconditioner<3, float>;

template class CompatibleLaplaceMultigridPreconditioner<2, double>;
template class CompatibleLaplaceMultigridPreconditioner<3, double>;

} // namespace IncNS
