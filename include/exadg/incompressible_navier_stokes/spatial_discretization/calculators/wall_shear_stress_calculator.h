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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_WALL_SHEAR_STRESS_CALCULATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_WALL_SHEAR_STRESS_CALCULATOR_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * This class calculates the wall shear stress on parts of the domain's boundary identified by
 * 'write_wall_shear_stress_on_IDs' or alternatively the entire boundary iff the only provided
 * boundary ID is dealii::numbers::invalid_boundary_id.
 * Values on the (sub-)boundary are interpolated via dealii::FEFaceEvaluation and exported via
 * dealii::DataOutFaces (in 3D)
 * or dealii::DataOut (in 2D, .vtk output via dealii::DataOutFaces<2> not available ).
 * Constant density and viscosity are assumed. The face to cell index match is performed once
 * up front and checked in DEBUG mode only, despite the fact that face orientations on the
 * boundary *should* stay constant.
 */

template<int dim, typename Number>
class WallShearStressCalculator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  WallShearStressCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_in,
             unsigned int const                      quad_index_in,
             double const                            kinematic_viscosity,
             double const                            density);

  void
  compute_wall_shear_stress(
    VectorType &                                  dst,
    VectorType const &                            src,
    std::shared_ptr<dealii::Mapping<dim> const>   mapping,
    std::vector<dealii::types::boundary_id> const write_wall_shear_stress_on_IDs) const;

private:
  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int                                              dof_index;
  unsigned int                                              quad_index;
  std::vector<std::vector<dealii::types::global_dof_index>> face_to_cell_index;
  double                                                    dynamic_viscosity;
  double const                                              rel_tol = 1.0e-5;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CALCULATORS_WALL_SHEAR_STRESS_CALCULATOR_H_ \
        */
