/*
 * multigrid_preconditioner.cpp
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

#include <exadg/structure/preconditioners/multigrid_preconditioner.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
MultigridPreconditioner<dim, Number>::MultigridPreconditioner(MPI_Comm const & mpi_comm)
  : Base(mpi_comm), pde_operator(nullptr), nonlinear(true)
{
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize(
  MultigridData const &                       mg_data,
  parallel::TriangulationBase<dim> const *    tria,
  FiniteElement<dim> const &                  fe,
  Mapping<dim> const &                        mapping,
  ElasticityOperatorBase<dim, Number> const & pde_operator,
  bool const                                  nonlinear_operator,
  Map const *                                 dirichlet_bc,
  PeriodicFacePairs *                         periodic_face_pairs)
{
  this->pde_operator = &pde_operator;

  this->data = this->pde_operator->get_data();

  this->nonlinear = nonlinear_operator;

  Base::initialize(
    mg_data, tria, fe, mapping, false /*operator_is_singular*/, dirichlet_bc, periodic_face_pairs);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update()
{
  if(nonlinear)
  {
    update_operators();

    this->update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(false /* operator_is_singular */);
  }
  else
  {
    AssertThrow(false,
                ExcMessage(
                  "Update of multigrid preconditioner is not implemented for linear elasticity."));
  }
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

  if(nonlinear)
    matrix_free_data.append_mapping_flags(PDEOperatorNonlinear::get_mapping_flags());
  else // linear
    matrix_free_data.append_mapping_flags(PDEOperatorLinear::get_mapping_flags());

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "elasticity_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "elasticity_dof_handler");
  matrix_free_data.insert_quadrature(QGauss<1>(this->level_info[level].degree() + 1),
                                     "elasticity_quadrature");
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize_constrained_dofs(
  DoFHandler<dim> const & dof_handler,
  MGConstrainedDoFs &     constrained_dofs,
  Map const &             dirichlet_bc)
{
  // We use data.bc->dirichlet_bc since we also need dirichlet_bc_component_mask,
  // but the argument dirichlet_bc could be used as well
  (void)dirichlet_bc;

  constrained_dofs.initialize(dof_handler);
  for(auto it : data.bc->dirichlet_bc)
  {
    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(it.first);

    ComponentMask mask    = ComponentMask();
    auto          it_mask = data.bc->dirichlet_bc_component_mask.find(it.first);
    if(it_mask != data.bc->dirichlet_bc_component_mask.end())
      mask = it_mask->second;

    constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary, mask);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update_operators()
{
  PDEOperatorNonlinear const * pde_operator_nonlinear =
    dynamic_cast<PDEOperatorNonlinear const *>(pde_operator);

  VectorType const & vector_linearization = pde_operator_nonlinear->get_solution_linearization();

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

  set_solution_linearization(*vector_multigrid_type_ptr);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_solution_linearization(
  VectorTypeMG const & vector_linearization)
{
  // copy velocity to finest level
  this->get_operator_nonlinear(this->fine_level)->set_solution_linearization(vector_linearization);

  // interpolate velocity from fine to coarse level
  for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
  {
    auto & vector_fine_level =
      this->get_operator_nonlinear(level - 0)->get_solution_linearization();
    auto vector_coarse_level =
      this->get_operator_nonlinear(level - 1)->get_solution_linearization();
    this->transfers.interpolate(level, vector_coarse_level, vector_fine_level);
    this->get_operator_nonlinear(level - 1)->set_solution_linearization(vector_coarse_level);
  }
}

template<int dim, typename Number>
std::shared_ptr<
  NonLinearOperator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::get_operator_nonlinear(unsigned int level)
{
  std::shared_ptr<MGOperatorNonlinear> mg_operator =
    std::dynamic_pointer_cast<MGOperatorNonlinear>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  std::shared_ptr<MGOperatorBase> mg_operator_level;

  data.dof_index  = this->matrix_free_data_objects[level]->get_dof_index("elasticity_dof_handler");
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("elasticity_quadrature");

  if(nonlinear)
  {
    std::shared_ptr<PDEOperatorNonlinearMG> pde_operator_level(new PDEOperatorNonlinearMG());
    pde_operator_level->initialize(*this->matrix_free_objects[level],
                                   *this->constraints[level],
                                   data);

    mg_operator_level.reset(new MGOperatorNonlinear(pde_operator_level));
  }
  else // linear
  {
    std::shared_ptr<PDEOperatorLinearMG> pde_operator_level(new PDEOperatorLinearMG());
    pde_operator_level->initialize(*this->matrix_free_objects[level],
                                   *this->constraints[level],
                                   data);

    mg_operator_level.reset(new MGOperatorLinear(pde_operator_level));
  }

  return mg_operator_level;
}

template class MultigridPreconditioner<2, float>;
template class MultigridPreconditioner<3, float>;

template class MultigridPreconditioner<2, double>;
template class MultigridPreconditioner<3, double>;

} // namespace Structure
} // namespace ExaDG
