/*
 * elasticity_operator_base.cpp
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

#include "elasticity_operator_base.h"

// deal.II
#include <deal.II/numerics/vector_tools.h>

namespace Structure
{
template<int dim, typename Number>
ElasticityOperatorBase<dim, Number>::ElasticityOperatorBase() : scaling_factor_mass(1.0)
{
}

template<int dim, typename Number>
IntegratorFlags
ElasticityOperatorBase<dim, Number>::get_integrator_flags(bool const unsteady) const
{
  IntegratorFlags flags;

  flags.cell_evaluate  = CellFlags(unsteady, true, false);
  flags.cell_integrate = CellFlags(unsteady, true, false);

  // evaluation of Neumann BCs
  flags.face_evaluate  = FaceFlags(false, false);
  flags.face_integrate = FaceFlags(true, false);

  return flags;
}

template<int dim, typename Number>
MappingFlags
ElasticityOperatorBase<dim, Number>::get_mapping_flags()
{
  MappingFlags flags;

  flags.cells = update_gradients | update_JxW_values;

  flags.boundary_faces =
    update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points;

  return flags;
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::initialize(MatrixFree<dim, Number> const &   matrix_free,
                                                AffineConstraints<double> const & constraint_matrix,
                                                OperatorData<dim> const &         data)
{
  operator_data = data;

  Base::reinit(matrix_free, constraint_matrix, data);

  this->integrator_flags = this->get_integrator_flags(data.unsteady);

  material_handler.initialize(data.material_descriptor);
}

template<int dim, typename Number>
OperatorData<dim> const &
ElasticityOperatorBase<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::set_scaling_factor_mass(double const factor) const
{
  scaling_factor_mass = factor;
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::set_dirichlet_values_continuous(VectorType & dst,
                                                                     double const time) const
{
  std::map<types::global_dof_index, double> boundary_values;
  fill_dirichlet_values_continuous(boundary_values, time);

  // set Dirichlet values in solution vector
  for(auto m : boundary_values)
    if(dst.get_partitioner()->in_local_range(m.first))
      dst[m.first] = m.second;
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  this->material_handler.reinit(*this->matrix_free, cell);
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::fill_dirichlet_values_continuous(
  std::map<types::global_dof_index, double> & boundary_values,
  double const                                time) const
{
  for(auto dbc : operator_data.bc->dirichlet_bc)
  {
    dbc.second->set_time(time);
    ComponentMask mask = operator_data.bc->dirichlet_bc_component_mask.find(dbc.first)->second;

    VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
                                             this->matrix_free->get_dof_handler(
                                               operator_data.dof_index),
                                             dbc.first,
                                             *dbc.second,
                                             boundary_values,
                                             mask);
  }
}

template class ElasticityOperatorBase<2, float>;
template class ElasticityOperatorBase<2, double>;

template class ElasticityOperatorBase<3, float>;
template class ElasticityOperatorBase<3, double>;
} // namespace Structure
