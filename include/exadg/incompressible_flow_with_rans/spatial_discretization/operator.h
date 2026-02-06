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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_RANS_SPATIAL_DISCRETIZATION_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_RANS_SPATIAL_DISCRETIZATION_H_

#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>

#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/finite_element.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <exadg/incompressible_flow_with_rans/spatial_discretization/viscosity_calculator.h>

namespace ExaDG
{
namespace NSRans
{
/*
* @brief Class for the viscosity operator in RANS simulations
* @details This class wraps the ViscosityCalculator to provide an interface compatible with the ExaDG MatrixFree framework.
*/
template<int dim, typename Number>
class ViscosityOperator : public dealii::Subscriptor
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  ViscosityOperator(std::shared_ptr<Grid<dim> const> grid_in, unsigned int const fe_degree_in);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
        std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in);

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  void
  initialize_viscosity_calculator(RANS::TurbulenceModelData const & turbulence_model_data);

  void
  calculate_eddy_viscosity() const;

  void
  set_turbulent_kinetic_energy(VectorType const & tke_in) const;

  void
  set_tke_dissipation_rate(VectorType const & epsilon_in) const;

  void
  extrapolate_eddy_viscosity_to_dof(VectorType & dst, unsigned int const dst_dof_index) const;

private:
  std::shared_ptr<Grid<dim> const> grid;
  unsigned int                     fe_degree;

  std::shared_ptr<dealii::FiniteElement<dim>> fe;
  std::shared_ptr<dealii::DoFHandler<dim>>    dof_handler;

  dealii::AffineConstraints<Number> affine_constraints;

  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data;

  std::shared_ptr<ViscosityCalculator<dim, Number>> viscosity_calculator;

  const std::string dof_name  = "eddy_viscosity";
  const std::string quad_name = "eddy_viscosity_quad";
};

} // namespace NSRans
} // namespace ExaDG

#endif
