/*
 * comp_navier_stokes_calculators.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_CALCULATORS_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_CALCULATORS_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>
#include <deal.II/matrix_free/matrix_free.h>

template<int dim, typename Number>
class VorticityCalculator
{
public:
  static const unsigned int number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VorticityCalculator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;

  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  VorticityCalculator() : matrix_free(nullptr), dof_index(0), quad_index(0){};

  void
  initialize(MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const              dof_index_in,
             unsigned int const              quad_index_in)
  {
    matrix_free = &matrix_free_in;
    dof_index   = dof_index_in;
    quad_index  = quad_index_in;
  }

  void
  compute_vorticity(VectorType & dst, VectorType const & src) const
  {
    matrix_free->cell_loop(&This::local_compute_vorticity, this, dst, src);
  }

private:
  void
  local_compute_vorticity(MatrixFree<dim, Number> const &               matrix_free,
                          VectorType &                                  dst,
                          VectorType const &                            src,
                          std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorVector integrator(matrix_free, dof_index, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src);
      integrator.evaluate(false, true, false);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        Tensor<1, number_vorticity_components, VectorizedArray<Number>> omega =
          integrator.get_curl(q);

        // omega_vector is a vector with dim components
        // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
        // for dim=2: omega_vector[0] = omega,
        //            omega_vector[1] = 0
        vector omega_vector;
        for(unsigned int d = 0; d < number_vorticity_components; ++d)
          omega_vector[d] = omega[d];
        integrator.submit_value(omega_vector, q);
      }

      integrator.integrate(true, false);
      integrator.set_dof_values(dst);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;
  unsigned int                    dof_index;
  unsigned int                    quad_index;
};


template<int dim, typename Number>
class DivergenceCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef DivergenceCalculator<dim, Number> This;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;
  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;

  typedef VectorizedArray<Number> scalar;

  DivergenceCalculator()
    : matrix_free(nullptr), dof_index_vector(1), dof_index_scalar(2), quad_index(0){};

  void
  initialize(MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const              dof_index_vector_in,
             unsigned int const              dof_index_scalar_in,
             unsigned int const              quad_index_in)
  {
    matrix_free      = &matrix_free_in;
    dof_index_vector = dof_index_vector_in;
    dof_index_scalar = dof_index_scalar_in;
    quad_index       = quad_index_in;
  }

  void
  compute_divergence(VectorType & dst, VectorType const & src) const
  {
    matrix_free->cell_loop(&This::local_compute_divergence, this, dst, src);
  }

private:
  void
  local_compute_divergence(MatrixFree<dim, Number> const &               matrix_free,
                           VectorType &                                  dst,
                           VectorType const &                            src,
                           std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
    CellIntegratorScalar integrator_scalar(matrix_free, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator_vector.reinit(cell);
      integrator_vector.read_dof_values(src);
      integrator_vector.evaluate(false, true);

      integrator_scalar.reinit(cell);

      for(unsigned int q = 0; q < integrator_vector.n_q_points; q++)
      {
        scalar divergence = integrator_vector.get_divergence(q);
        integrator_scalar.submit_value(divergence, q);
      }

      integrator_scalar.integrate(true, false);
      integrator_scalar.set_dof_values(dst);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
};


/*
 *  The DoF vector contains the vector of conserved quantities (rho, rho u, rho E).
 *  This class allows to transfer these quantities into the derived variables (p, u, T)
 *  by using L2-projections.
 */
template<int dim, typename Number>
class p_u_T_Calculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef p_u_T_Calculator<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number>   CellIntegratorScalar;
  typedef CellIntegrator<dim, dim, Number> CellIntegratorVector;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  p_u_T_Calculator()
    : matrix_free(nullptr),
      dof_index_all(0),
      dof_index_vector(1),
      dof_index_scalar(2),
      quad_index(0),
      heat_capacity_ratio(-1.0),
      specific_gas_constant(-1.0)
  {
  }

  void
  initialize(MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const              dof_index_all_in,
             unsigned int const              dof_index_vector_in,
             unsigned int const              dof_index_scalar_in,
             unsigned int const              quad_index_in,
             double const                    heat_capacity_ratio_in,
             double const                    specific_gas_constant_in)
  {
    matrix_free           = &matrix_free_in;
    dof_index_all         = dof_index_all_in;
    dof_index_vector      = dof_index_vector_in;
    dof_index_scalar      = dof_index_scalar_in;
    quad_index            = quad_index_in;
    heat_capacity_ratio   = heat_capacity_ratio_in;
    specific_gas_constant = specific_gas_constant_in;
  }

  void
  compute_pressure(VectorType & pressure, VectorType const & solution_conserved) const
  {
    AssertThrow(heat_capacity_ratio > 0.0, ExcMessage("heat capacity ratio has not been set!"));
    AssertThrow(specific_gas_constant > 0.0, ExcMessage("specific gas constant has not been set!"));

    matrix_free->cell_loop(&This::local_apply_pressure, this, pressure, solution_conserved);
  }

  void
  compute_velocity(VectorType & velocity, VectorType const & solution_conserved) const
  {
    matrix_free->cell_loop(&This::local_apply_velocity, this, velocity, solution_conserved);
  }

  void
  compute_temperature(VectorType & temperature, VectorType const & solution_conserved) const
  {
    AssertThrow(heat_capacity_ratio > 0.0, ExcMessage("heat capacity ratio has not been set!"));
    AssertThrow(specific_gas_constant > 0.0, ExcMessage("specific gas constant has not been set!"));

    matrix_free->cell_loop(&This::local_apply_temperature, this, temperature, solution_conserved);
  }

private:
  void
  local_apply_pressure(MatrixFree<dim, Number> const &               matrix_free,
                       VectorType &                                  dst,
                       VectorType const &                            src,
                       std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    CellIntegratorScalar density(matrix_free, dof_index_all, quad_index, 0);
    CellIntegratorVector momentum(matrix_free, dof_index_all, quad_index, 1);
    CellIntegratorScalar energy(matrix_free, dof_index_all, quad_index, 1 + dim);

    // dst-vector
    CellIntegratorScalar pressure(matrix_free, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      density.reinit(cell);
      density.read_dof_values(src);
      density.evaluate(true, false, false);
      momentum.reinit(cell);
      momentum.read_dof_values(src);
      momentum.evaluate(true, false, false);
      energy.reinit(cell);
      energy.read_dof_values(src);
      energy.evaluate(true, false, false);

      // dst-vector
      pressure.reinit(cell);

      for(unsigned int q = 0; q < energy.n_q_points; ++q)
      {
        // conserved variables
        scalar rho   = density.get_value(q);
        vector rho_u = momentum.get_value(q);
        scalar rho_E = energy.get_value(q);

        // compute derived quantities in quadrature point
        vector u = rho_u / rho;

        scalar p = make_vectorized_array<Number>(heat_capacity_ratio - 1.0) *
                   (rho_E - 0.5 * rho * scalar_product(u, u));

        pressure.submit_value(p, q);
      }

      pressure.integrate(true, false);
      pressure.set_dof_values(dst);
    }
  }

  void
  local_apply_velocity(MatrixFree<dim, Number> const &               matrix_free,
                       VectorType &                                  dst,
                       VectorType const &                            src,
                       std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    CellIntegratorScalar density(matrix_free, dof_index_all, quad_index, 0);
    CellIntegratorVector momentum(matrix_free, dof_index_all, quad_index, 1);

    // dst-vector
    CellIntegratorVector velocity(matrix_free, dof_index_vector, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      density.reinit(cell);
      density.read_dof_values(src);
      density.evaluate(true, false, false);
      momentum.reinit(cell);
      momentum.read_dof_values(src);
      momentum.evaluate(true, false, false);

      // dst-vector
      velocity.reinit(cell);

      for(unsigned int q = 0; q < momentum.n_q_points; ++q)
      {
        // conserved variables
        scalar rho   = density.get_value(q);
        vector rho_u = momentum.get_value(q);

        // compute derived quantities in quadrature point
        vector u = rho_u / rho;

        velocity.submit_value(u, q);
      }

      velocity.integrate(true, false);
      velocity.set_dof_values(dst);
    }
  }

  void
  local_apply_temperature(MatrixFree<dim, Number> const &               matrix_free,
                          VectorType &                                  dst,
                          VectorType const &                            src,
                          std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    CellIntegratorScalar density(matrix_free, dof_index_all, quad_index, 0);
    CellIntegratorVector momentum(matrix_free, dof_index_all, quad_index, 1);
    CellIntegratorScalar energy(matrix_free, dof_index_all, quad_index, 1 + dim);

    // dst-vector
    CellIntegratorScalar temperature(matrix_free, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      density.reinit(cell);
      density.read_dof_values(src);
      density.evaluate(true, false, false);
      momentum.reinit(cell);
      momentum.read_dof_values(src);
      momentum.evaluate(true, false, false);
      energy.reinit(cell);
      energy.read_dof_values(src);
      energy.evaluate(true, false, false);

      // dst-vector
      temperature.reinit(cell);

      for(unsigned int q = 0; q < energy.n_q_points; ++q)
      {
        // conserved variables
        scalar rho   = density.get_value(q);
        vector rho_u = momentum.get_value(q);
        scalar rho_E = energy.get_value(q);

        // compute derived quantities in quadrature point
        vector u = rho_u / rho;
        scalar E = rho_E / rho;

        scalar T =
          make_vectorized_array<Number>((heat_capacity_ratio - 1.0) / specific_gas_constant) *
          (E - 0.5 * scalar_product(u, u));

        temperature.submit_value(T, q);
      }

      temperature.integrate(true, false);
      temperature.set_dof_values(dst);
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_all;
  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;

  double heat_capacity_ratio;
  double specific_gas_constant;
};

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_CALCULATORS_H_ \
        */
