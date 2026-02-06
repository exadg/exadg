/*  ______________________________________________________________________
 *
 *  exadg - high-order discontinuous galerkin for the exa-scale
 *
 *  copyright (c) 2021 by the exadg authors
 *
 *  this program is free software: you can redistribute it and/or modify
 *  it under the terms of the gnu general public license as published by
 *  the free software foundation, either version 3 of the license, or
 *  (at your option) any later version.
 *
 *  this program is distributed in the hope that it will be useful,
 *  but without any warranty; without even the implied warranty of
 *  merchantability or fitness for a particular purpose.  see the
 *  gnu general public license for more details.
 *
 *  you should have received a copy of the gnu general public license
 *  along with this program.  if not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_INCOMPRESSIBLE_FLOW_WITH_RANS_CALCULATOR_VISCOSITY_CALCULATOR_H
#define INCLUDE_INCOMPRESSIBLE_FLOW_WITH_RANS_CALCULATOR_VISCOSITY_CALCULATOR_H

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <exadg/matrix_free/integrators.h>

#include <exadg/rans_equations/user_interface/viscosity_model_data.h>

namespace ExaDG
{
namespace NSRans
{
/*
 * @brief Class to calculate the eddy viscosity based on turbulence model data
 * @details This class computes the eddy viscosity using the provided turbulent  model scalars
 */
template<int dim, typename Number>
class ViscosityCalculator
{
private:
  typedef ViscosityCalculator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

public:
  ViscosityCalculator();

  ~ViscosityCalculator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             RANS::TurbulenceModelData const &       turbulence_model_data_in,
             unsigned int const                      dof_index_in,
             unsigned int const                      quad_index_in);

  void
  set_turbulent_kinetic_energy(VectorType const & tke_in);

  void
  set_tke_dissipation_rate(VectorType const & epsilon_in);

  void
  calculate_eddy_viscosity();

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_eddy_viscosity() const;

  void
  extrapolate_eddy_viscosity_to_dof(VectorType & dst, unsigned int const & target_dof_index) const;

private:
  void
  cell_loop_set_viscosity(dealii::MatrixFree<dim, Number> const & data,
                          VectorType &                            dst,
                          VectorType const &,
                          Range const & cell_range) const;

  void
  spalar_allmaras_model(scalar const & tke, scalar & viscosity) const;

  void
  standard_k_epsilon_model(scalar const & tke, scalar const & epsilon, scalar & viscosity) const;

  void
  add_viscosity(scalar const & scalar_1, scalar const & scalar_2, scalar & viscosity) const;

  void
  extrapolate_to_new_dof(VectorType const &   src,
                         VectorType &         dst,
                         unsigned int const & target_dof_index) const;

public:
  VectorType const *        turbulent_kinetic_energy;
  VectorType const *        tke_dissipation_rate;
  VectorType                eddy_viscosity;
  RANS::TurbulenceModelData turbulence_model_data;

  std::vector<double> model_coefficients;

protected:
  unsigned int                            dof_index;
  unsigned int                            quad_index;
  dealii::MatrixFree<dim, Number> const * matrix_free;
};
} // namespace NSRans
} // namespace ExaDG

#endif
