/*
 * st_venant_kirchhoff.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
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
using namespace dealii;

template<int dim>
struct StVenantKirchhoffData : public MaterialData
{
  StVenantKirchhoffData(MaterialType const &                 type,
                        double const &                       E,
                        double const &                       nu,
                        Type2D const &                       type_two_dim,
                        std::shared_ptr<Function<dim>> const E_function = nullptr)
    : MaterialData(type), E(E), E_function(E_function), nu(nu), type_two_dim(type_two_dim)
  {
  }

  double                         E;
  std::shared_ptr<Function<dim>> E_function;

  double nu;
  Type2D type_two_dim;
};

template<int dim, typename Number>
class StVenantKirchhoff : public Material<dim, Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef std::pair<unsigned int, unsigned int>      Range;
  typedef CellIntegrator<dim, dim, Number>           IntegratorCell;

  StVenantKirchhoff(MatrixFree<dim, Number> const &    matrix_free,
                    unsigned int const                 n_q_points_1d,
                    unsigned int const                 dof_index,
                    unsigned int const                 quad_index,
                    StVenantKirchhoffData<dim> const & data);

  Tensor<2, dim, VectorizedArray<Number>>
    evaluate_stress(Tensor<2, dim, VectorizedArray<Number>> const & E,
                    unsigned int const                              cell,
                    unsigned int const                              q) const;

  Tensor<2, dim, VectorizedArray<Number>> apply_C(Tensor<2, dim, VectorizedArray<Number>> const & E,
                                                  unsigned int const cell,
                                                  unsigned int const q) const;

private:
  Number
  get_f0_factor() const;

  Number
  get_f1_factor() const;

  Number
  get_f2_factor() const;

  void
  cell_loop_set_coefficients(MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const & src,
                             Range const &      cell_range) const;

  unsigned int dof_index;
  unsigned int quad_index;

  StVenantKirchhoffData<dim> const & data;

  mutable VectorizedArray<Number> f0;
  mutable VectorizedArray<Number> f1;
  mutable VectorizedArray<Number> f2;

  // cache coefficients for spatially varying material parameters
  bool                                           E_is_variable;
  mutable VariableCoefficientsCells<dim, Number> f0_coefficients;
  mutable VariableCoefficientsCells<dim, Number> f1_coefficients;
  mutable VariableCoefficientsCells<dim, Number> f2_coefficients;
};
} // namespace Structure
} // namespace ExaDG

#endif
