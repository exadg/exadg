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

#include "operator.h"
#include <exadg/incompressible_flow_with_rans/calculator/operator.h>

namespace ExaDG
{
namespace NSRans
{
template<int dim, typename Number>
ViscosityOperator<dim, Number>::ViscosityOperator(std::shared_ptr<Grid<dim> const> grid_in,
                                                  unsigned int const               fe_degree_in)
  : grid(grid_in), fe_degree(fe_degree_in)
{
  // Use the factory function from finite_element.h
  // Arguments:
  // 1. ElementType: Hypercube (Matches previous FE_DGQ usage. Change to Simplex if needed)
  // 2. is_dg:       true      (We want discontinuous elements)
  // 3. n_components: 1        (Eddy viscosity is a scalar)
  // 4. degree:      degree_in
  fe = create_finite_element<dim>(ElementType::Hypercube, true, 1, fe_degree); //

  // Initialize DoFHandler with the created FE
  dof_handler = std::make_shared<dealii::DoFHandler<dim>>(*grid->triangulation);
  dof_handler->distribute_dofs(*fe);

  // Initialize Constraints (Closed but empty for DG)
  affine_constraints.clear();
  affine_constraints.close();
}

template<int dim, typename Number>
void
ViscosityOperator<dim, Number>::fill_matrix_free_data(MatrixFreeData<dim, Number> & mf_data) const
{
  // Register DoFHandler and Constraints
  mf_data.insert_dof_handler(dof_handler.get(), dof_name);
  mf_data.insert_constraint(&affine_constraints, dof_name);

  // Register Quadrature
  // Standard Gauss quadrature with degree + 1 precision
  auto quadrature = std::make_shared<dealii::QGauss<dim>>(fe_degree + 1);
  mf_data.insert_quadrature(*quadrature, quad_name);
}

template<int dim, typename Number>
void
ViscosityOperator<dim, Number>::setup(
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in)
{
  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;
}

template<int dim, typename Number>
unsigned int
ViscosityOperator<dim, Number>::get_dof_index() const
{
  return matrix_free_data->get_dof_index(dof_name);
}

template<int dim, typename Number>
unsigned int
ViscosityOperator<dim, Number>::get_quad_index() const
{
  return matrix_free_data->get_quad_index(quad_name);
}

template<int dim, typename Number>
void
ViscosityOperator<dim, Number>::initialize_viscosity_calculator(
  RANS::TurbulenceModelData const & turbulence_model_data)
{
  viscosity_calculator = std::make_shared<ViscosityCalculator<dim, Number>>();

  viscosity_calculator->initialize(*matrix_free,
                                   turbulence_model_data,
                                   get_dof_index(),
                                   get_quad_index());
}

template<int dim, typename Number>
void
ViscosityOperator<dim, Number>::calculate_eddy_viscosity() const
{
  viscosity_calculator->calculate_eddy_viscosity();
}

template<int dim, typename Number>
void
ViscosityOperator<dim, Number>::set_turbulent_kinetic_energy(VectorType const & tke_in) const
{
  viscosity_calculator->set_turbulent_kinetic_energy(tke_in);
}

template<int dim, typename Number>
void
ViscosityOperator<dim, Number>::set_tke_dissipation_rate(VectorType const & epsilon_in) const
{
  viscosity_calculator->set_tke_dissipation_rate(epsilon_in);
}

template<int dim, typename Number>
void
ViscosityOperator<dim, Number>::extrapolate_eddy_viscosity_to_dof(
  VectorType &       dst,
  unsigned int const dst_dof_index) const
{
  viscosity_calculator->extrapolate_eddy_viscosity_to_dof(dst, dst_dof_index);
}

template class ViscosityOperator<2, float>;
template class ViscosityOperator<3, float>;
template class ViscosityOperator<2, double>;
template class ViscosityOperator<3, double>;

} // namespace NSRans
} // namespace ExaDG
