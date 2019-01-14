#ifndef OPERATOR_REINIT_MULTIGRID
#define OPERATOR_REINIT_MULTIGRID

#include "../../../include/functionalities/constraints.h"

using namespace dealii;

template<int dim, typename Number>
void
do_reinit_multigrid(
  DoFHandler<dim> const &             dof_handler,
  Mapping<dim> const &                mapping,
  PreconditionableOperatorData<dim> & operator_data,
  MGConstrainedDoFs const &           mg_constrained_dofs,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                              periodic_face_pairs,
  unsigned int const          level,
  MatrixFree<dim, Number> &   data_own,
  AffineConstraints<double> & constraint_own)
{
  // set dof_index and quad_index to 0 since we only consider a subset
  operator_data.set_dof_index(0);
  operator_data.set_quad_index(0);

  // check if DG or CG (for explanation: see above)
  bool is_dg = dof_handler.get_fe().dofs_per_vertex == 0;

  // setup MatrixFree::AdditionalData
  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  additional_data.level_mg_handler = level;

  additional_data.mapping_update_flags = operator_data.get_mapping_update_flags();

  if(is_dg)
  {
    additional_data.mapping_update_flags_inner_faces =
      operator_data.get_mapping_update_flags_inner_faces();
    additional_data.mapping_update_flags_boundary_faces =
      operator_data.get_mapping_update_flags_boundary_faces();
  }

  if(operator_data.do_use_cell_based_loops() && is_dg)
  {
    auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
      &dof_handler.get_triangulation());
    Categorization::do_cell_based_loops(*tria, additional_data, level);
  }


  ConstraintUtil<dim>::add_constraints(
    is_dg, false, dof_handler, constraint_own, mg_constrained_dofs, periodic_face_pairs, level);

  // // setup constraint matrix for CG
  // if(!is_dg)
  // {
  //   this->add_constraints(
  //     dof_handler, constraint_own, mg_constrained_dofs, periodic_face_pairs, level);
  // }

  constraint_own.close();

  QGauss<1> const quad(dof_handler.get_fe().degree + 1);

  data_own.reinit(mapping, dof_handler, constraint_own, quad, additional_data);
}


#endif