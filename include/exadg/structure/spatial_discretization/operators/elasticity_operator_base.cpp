/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
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

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/structure/spatial_discretization/operators/elasticity_operator_base.h>

namespace ExaDG
{
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

  auto const unsteady_flag =
    unsteady ? dealii::EvaluationFlags::values : dealii::EvaluationFlags::nothing;

  flags.cell_evaluate  = unsteady_flag | dealii::EvaluationFlags::gradients;
  flags.cell_integrate = unsteady_flag | dealii::EvaluationFlags::gradients;

  // evaluation of Neumann BCs
  flags.face_evaluate  = dealii::EvaluationFlags::nothing;
  flags.face_integrate = dealii::EvaluationFlags::values;

  return flags;
}

template<int dim, typename Number>
MappingFlags
ElasticityOperatorBase<dim, Number>::get_mapping_flags()
{
  MappingFlags flags;

  flags.cells =
    dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points;

  flags.boundary_faces = dealii::update_gradients | dealii::update_JxW_values |
                         dealii::update_normal_vectors | dealii::update_quadrature_points;

  return flags;
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  OperatorData<dim> const &                 data)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = this->get_integrator_flags(data.unsteady);

  material_handler.initialize(matrix_free,
                              data.dof_index,
                              data.quad_index,
                              data.material_descriptor);
}

template<int dim, typename Number>
OperatorData<dim> const &
ElasticityOperatorBase<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::set_scaling_factor_mass_operator(
  double const scaling_factor) const
{
  scaling_factor_mass = scaling_factor;
}

template<int dim, typename Number>
double
ElasticityOperatorBase<dim, Number>::get_scaling_factor_mass_operator() const
{
  return scaling_factor_mass;
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::set_inhomogeneous_boundary_values(VectorType & dst) const
{
  // standard Dirichlet boundary conditions
  std::map<dealii::types::global_dof_index, double> boundary_values;
  for(auto dbc : operator_data.bc->dirichlet_bc)
  {
    dbc.second->set_time(this->get_time());
    dealii::ComponentMask mask =
      operator_data.bc->dirichlet_bc_component_mask.find(dbc.first)->second;

    dealii::VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
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

  dst.update_ghost_values();

  // DirichletCached boundary conditions
  if(not(operator_data.bc->dirichlet_cached_bc.empty()))
  {
    unsigned int const dof_index  = operator_data.dof_index;
    unsigned int const quad_index = operator_data.quad_index_gauss_lobatto;

    IntegratorFace integrator(*this->matrix_free, true, dof_index, quad_index);

    for(unsigned int face = this->matrix_free->n_inner_face_batches();
        face <
        this->matrix_free->n_inner_face_batches() + this->matrix_free->n_boundary_face_batches();
        ++face)
    {
      dealii::types::boundary_id const boundary_id = this->matrix_free->get_boundary_id(face);

      BoundaryType const boundary_type = operator_data.bc->get_boundary_type(boundary_id);

      if(boundary_type == BoundaryType::DirichletCached)
      {
        integrator.reinit(face);
        integrator.read_dof_values(dst);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          unsigned int const local_face_number =
            this->matrix_free->get_face_info(face).interior_face_no;

          unsigned int const index = this->matrix_free->get_shape_info(dof_index, quad_index)
                                       .face_to_cell_index_nodal[local_face_number][q];

          dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> g;

          if(boundary_type == BoundaryType::DirichletCached)
          {
            auto bc = operator_data.bc->get_dirichlet_cached_data();

            g = FunctionEvaluator<1, dim, Number>::value(*bc, face, q, quad_index);
          }
          else
          {
            AssertThrow(false, dealii::ExcMessage("Not implemented."));
          }

          integrator.submit_dof_value(g, index);
        }

        integrator.set_dof_values_plain(dst);
      }
      else
      {
        AssertThrow(boundary_type == BoundaryType::Dirichlet or
                      boundary_type == BoundaryType::Neumann or
                      boundary_type == BoundaryType::NeumannCached,
                    dealii::ExcMessage("BoundaryType not implemented."));
      }
    }
  }
}

template<int dim, typename Number>
void
ElasticityOperatorBase<dim, Number>::reinit_cell_derived(IntegratorCell &   integrator,
                                                         unsigned int const cell) const
{
  (void)integrator;

  this->material_handler.reinit(*this->matrix_free, cell);
}

template class ElasticityOperatorBase<2, float>;
template class ElasticityOperatorBase<2, double>;

template class ElasticityOperatorBase<3, float>;
template class ElasticityOperatorBase<3, double>;
} // namespace Structure
} // namespace ExaDG
