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

  StVenantKirchhoff(dealii::MatrixFree<dim, Number> const & matrix_free,
                    unsigned int const                      n_q_points_1d,
                    unsigned int const                      dof_index,
                    unsigned int const                      quad_index,
                    StVenantKirchhoffData<dim> const &      data);

  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
    evaluate_stress(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & E,
                    unsigned int const                                              cell,
                    unsigned int const                                              q) const;

  dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
    apply_C(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & E,
            unsigned int const                                              cell,
            unsigned int const                                              q) const;

private:
  Number
  get_f0_factor() const;

  Number
  get_f1_factor() const;

  Number
  get_f2_factor() const;

  void
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const;

  unsigned int dof_index;
  unsigned int quad_index;

  StVenantKirchhoffData<dim> const & data;

  mutable dealii::VectorizedArray<Number> f0;
  mutable dealii::VectorizedArray<Number> f1;
  mutable dealii::VectorizedArray<Number> f2;

  // cache coefficients for spatially varying material parameters
  bool                                           E_is_variable;
  mutable VariableCoefficientsCells<dim, Number> f0_coefficients;
  mutable VariableCoefficientsCells<dim, Number> f1_coefficients;
  mutable VariableCoefficientsCells<dim, Number> f2_coefficients;
};
} // namespace Structure
} // namespace ExaDG

#endif
