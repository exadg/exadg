/*
 * elasticity_operator_base.cpp
 *
 *  Created on: 03.05.2020
 *      Author: fehn
 */

#include "elasticity_operator_base.h"

// deal.II
#include <deal.II/numerics/vector_tools.h>

#include "../../../functions_and_boundary_conditions/evaluate_functions.h"

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
ElasticityOperatorBase<dim, Number>::set_constrained_values(VectorType & dst,
                                                            double const time) const
{
  // standard Dirichlet boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
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

  // set Dirichlet values in solution vector
  for(auto m : boundary_values)
    if(dst.get_partitioner()->in_local_range(m.first))
      dst[m.first] = m.second;

  // Dirichlet mortar type boundary conditions
  if(not(operator_data.bc->dirichlet_mortar_bc.empty()))
  {
    unsigned int const dof_index  = operator_data.dof_index;
    unsigned int const quad_index = operator_data.quad_index_gauss_lobatto;

    IntegratorFace integrator(*this->matrix_free, true, dof_index, quad_index);

    for(unsigned int face = this->matrix_free->n_inner_face_batches();
        face <
        this->matrix_free->n_inner_face_batches() + this->matrix_free->n_boundary_face_batches();
        ++face)
    {
      types::boundary_id const boundary_id = this->matrix_free->get_boundary_id(face);

      BoundaryType const boundary_type = operator_data.bc->get_boundary_type(boundary_id);

      if(boundary_type == BoundaryType::DirichletMortar)
      {
        integrator.reinit(face);
        integrator.read_dof_values(dst);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          unsigned int const local_face_number =
            this->matrix_free->get_face_info(face).interior_face_no;

          unsigned int const index = this->matrix_free->get_shape_info(dof_index, quad_index)
                                       .face_to_cell_index_nodal[local_face_number][q];

          Tensor<1, dim, VectorizedArray<Number>> g;

          if(boundary_type == BoundaryType::DirichletMortar)
          {
            auto bc = operator_data.bc->dirichlet_mortar_bc.find(boundary_id)->second;

            g = FunctionEvaluator<1, dim, Number>::value(bc, face, q, quad_index);
          }
          else
          {
            AssertThrow(false, ExcMessage("Not implemented."));
          }

          integrator.submit_dof_value(g, index);
        }

        integrator.set_dof_values_plain(dst);
      }
      else
      {
        AssertThrow(boundary_type == BoundaryType::Dirichlet ||
                      boundary_type == BoundaryType::Neumann ||
                      boundary_type == BoundaryType::NeumannMortar,
                    ExcMessage("BoundaryType not implemented."));
      }
    }
  }
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  this->material_handler.reinit(*this->matrix_free, cell);
}

template class ElasticityOperatorBase<2, float>;
template class ElasticityOperatorBase<2, double>;

template class ElasticityOperatorBase<3, float>;
template class ElasticityOperatorBase<3, double>;
} // namespace Structure
