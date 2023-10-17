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

#ifndef STRUCTURE_MATERIAL_LIBRARY_STVENANTKIRCHHOFF
#define STRUCTURE_MATERIAL_LIBRARY_STVENANTKIRCHHOFF

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/structure/material/material.h>

namespace ExaDG
{
namespace Structure
{
template<int dim>
struct StVenantKirchhoffData : public MaterialData
{
  StVenantKirchhoffData(MaterialType const &                         type,
                        double const &                               E,
                        double const &                               nu,
                        Type2D const &                               type_two_dim,
                        std::shared_ptr<dealii::Function<dim>> const E_function = nullptr)
    : MaterialData(type), E(E), E_function(E_function), nu(nu), type_two_dim(type_two_dim)
  {
  }

  double                                 E;
  std::shared_ptr<dealii::Function<dim>> E_function;

  double nu;
  Type2D type_two_dim;
};

template<int dim, typename Number>
class StVenantKirchhoff : public Material<dim, Number>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>              Range;
  typedef CellIntegrator<dim, dim, Number>                   IntegratorCell;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> tensor;

  StVenantKirchhoff(dealii::MatrixFree<dim, Number> const & matrix_free,
                    unsigned int const                      dof_index,
                    unsigned int const                      quad_index,
                    StVenantKirchhoffData<dim> const &      data,
                    bool const                              large_deformation,
                    unsigned int const                      check_type);

  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  second_piola_kirchhoff_stress(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_displacement,
    unsigned int const                                              cell,
    unsigned int const                                              q,
    bool const                                                      force_evaluation) const final;

  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  second_piola_kirchhoff_stress_displacement_derivative(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & gradient_increment,
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & deformation_gradient,
    unsigned int const                                              cell,
    unsigned int const                                              q) const final;

private:
  /*
   * Factor out coefficients for faster computation. Note that these factors do not contain the
   * (potentially variable) Young's modulus.
   */
  Number
  get_f0_factor() const;

  Number
  get_f1_factor() const;

  Number
  get_f2_factor() const;

  /*
   * The second Piola-Kirchhoff stress tensor S is given as S = lambda * I * tr(E) + 2 mu E, with E
   * being the Green-Lagrange strain tensor and Lamee parameters lambda and mu. This leads to
   * Sii = f0 * Eii + f1 * sum_{j = 1, ..., dim; i!=j} Eij, for i = 1, ..., dim, and
   * Sij = f2 * (Eij + Eji),    for i, j = 1, ..., dim and i != j.
   * The latter symmetrizes the off-diagonal entries in the strain argument to reduce computations.
   */
  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
  second_piola_kirchhoff_stress_symmetrize(
    dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & strain,
    unsigned int const                                              cell,
    unsigned int const                                              q) const;

  /*
   * Store factors involving (potentially variable) Young's modulus.
   */
  void
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const;

  unsigned int dof_index;
  unsigned int quad_index;

  StVenantKirchhoffData<dim> const & data;

  bool large_deformation;

  unsigned int check_type;

  mutable dealii::VectorizedArray<Number> f0;
  mutable dealii::VectorizedArray<Number> f1;
  mutable dealii::VectorizedArray<Number> f2;

  // cache coefficients for spatially varying material parameters
  bool                                                          E_is_variable;
  mutable VariableCoefficients<dealii::VectorizedArray<Number>> f0_coefficients;
  mutable VariableCoefficients<dealii::VectorizedArray<Number>> f1_coefficients;
  mutable VariableCoefficients<dealii::VectorizedArray<Number>> f2_coefficients;
};
} // namespace Structure
} // namespace ExaDG

#endif
