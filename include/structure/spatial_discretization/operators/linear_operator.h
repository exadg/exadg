/*
 * linear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_LINEAR_OPERATOR_H_

#include "../../../functionalities/evaluate_functions.h"

#include "../../material/material_handler.h"
#include "continuum_mechanics_util.h"
#include "operator_data.h"

namespace Structure
{
/*
 * This function calculates the Neumann boundary value.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<Number>>
  calculate_neumann_value(unsigned int const                             q,
                          FaceIntegrator<dim, dim, Number> const &       integrator,
                          BoundaryType const &                           boundary_type,
                          types::boundary_id const                       boundary_id,
                          std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor,
                          double const &                                 time)
{
  Tensor<1, dim, VectorizedArray<Number>> normal_gradient;

  if(boundary_type == BoundaryType::Neumann)
  {
    auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
    auto q_points = integrator.quadrature_point(q);

    normal_gradient = FunctionEvaluator<1, dim, Number>::value(bc, q_points, time);
  }
  else
  {
    // do nothing

    AssertThrow(boundary_type == BoundaryType::Dirichlet,
                ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient;
}

template<int dim, typename Number>
class LinearOperator : public OperatorBase<dim, Number, OperatorData<dim>, dim>
{
public:
  typedef Number value_type;

private:
  typedef OperatorBase<dim, Number, OperatorData<dim>, dim> Base;
  typedef typename Base::VectorType                         VectorType;
  typedef typename Base::IntegratorCell                     IntegratorCell;
  typedef typename Base::IntegratorFace                     IntegratorFace;

public:
  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         OperatorData<dim> const &         operator_data)
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);

    this->integrator_flags = this->get_integrator_flags();

    material_handler.initialize(this->get_data().material_descriptor);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

    // evaluation of Neumann BCs
    flags.face_evaluate  = FaceFlags(false, false);
    flags.face_integrate = FaceFlags(true, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_gradients | update_JxW_values;

    flags.boundary_faces =
      update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points;

    return flags;
  }

  void
  set_dirichlet_values_continuous(VectorType & dst, double const time) const
  {
    std::map<types::global_dof_index, double> boundary_values;
    fill_dirichlet_values_continuous(boundary_values, time);

    // set Dirichlet values in solution vector
    for(auto m : boundary_values)
      if(dst.get_partitioner()->in_local_range(m.first))
        dst[m.first] = m.second;
  }

private:
  void
  reinit_cell(unsigned int const cell) const
  {
    Base::reinit_cell(cell);

    this->material_handler.reinit(*this->matrix_free, cell);
  }

  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // update geometrical information
      auto const gradient = integrator.get_gradient(q);
      auto const E        = apply_l<dim, Number>(gradient);

      // update material
      material->reinit(E);
      auto const C = material->get_dSdE();

      // test with gradients
      integrator.submit_gradient(apply_l_transposed<dim, Number>(C * E), q);
    }
  }

  void
  do_boundary_integral_continuous(IntegratorFace &           integrator_m,
                                  types::boundary_id const & boundary_id) const
  {
    BoundaryType boundary_type = this->data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      auto const neumann_value = calculate_neumann_value<dim, Number>(
        q, integrator_m, boundary_type, boundary_id, this->data.bc, this->time);

      integrator_m.submit_value(-neumann_value, q);
    }
  }

  void
  fill_dirichlet_values_continuous(std::map<types::global_dof_index, double> & boundary_values,
                                   double const                                time) const
  {
    for(auto dbc : this->data.bc->dirichlet_bc)
    {
      dbc.second->set_time(time);
      ComponentMask mask = this->data.bc->dirichlet_bc_component_mask.find(dbc.first)->second;

      VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
                                               this->matrix_free->get_dof_handler(
                                                 this->data.dof_index),
                                               dbc.first,
                                               *dbc.second,
                                               boundary_values,
                                               mask);
    }
  }

  mutable MaterialHandler<dim, Number> material_handler;
};
} // namespace Structure

#endif
