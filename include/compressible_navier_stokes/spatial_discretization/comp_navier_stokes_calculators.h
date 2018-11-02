/*
 * comp_navier_stokes_calculators.h
 *
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_CALCULATORS_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_CALCULATORS_H_

// deal.II
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

template<int dim, int fe_degree, typename value_type>
class VorticityCalculator
{
public:
  static const unsigned int number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef VorticityCalculator<dim, fe_degree, value_type> This;

  typedef FEEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type> FEEval_vectorial;

  VorticityCalculator() : data(nullptr), dof_index(0), quad_index(0){};

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             unsigned int const                  dof_index_in,
             unsigned int const                  quad_index_in)
  {
    this->data = &mf_data;
    dof_index  = dof_index_in;
    quad_index = quad_index_in;
  }

  void
  compute_vorticity(VectorType & dst, VectorType const & src) const
  {
    data->cell_loop(&This::local_compute_vorticity, this, dst, src);
  }

private:
  void
  local_compute_vorticity(MatrixFree<dim, value_type> const &           data,
                          VectorType &                                  dst,
                          VectorType const &                            src,
                          std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_vectorial fe_eval(data, dof_index, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(false, true, false);

      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        Tensor<1, number_vorticity_components, VectorizedArray<value_type>> omega =
          fe_eval.get_curl(q);
        // omega_vector is a vector with dim components
        // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
        // for dim=2: omega_vector[0] = omega,
        //            omega_vector[1] = 0
        Tensor<1, dim, VectorizedArray<value_type>> omega_vector;
        for(unsigned int d = 0; d < number_vorticity_components; ++d)
          omega_vector[d] = omega[d];
        fe_eval.submit_value(omega_vector, q);
      }

      fe_eval.integrate(true, false);
      fe_eval.set_dof_values(dst);
    }
  }

  MatrixFree<dim, value_type> const * data;
  unsigned int                        dof_index;
  unsigned int                        quad_index;
};


template<int dim, int fe_degree, typename value_type>
class DivergenceCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef DivergenceCalculator<dim, fe_degree, value_type> This;

  typedef FEEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type> FEEval_vectorial;
  typedef FEEvaluation<dim, fe_degree, fe_degree + 1, 1, value_type>   FEEval_scalar;

  DivergenceCalculator() : data(nullptr), dof_index_vector(1), dof_index_scalar(2), quad_index(0){};

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             unsigned int const                  dof_index_vector_in,
             unsigned int const                  dof_index_scalar_in,
             unsigned int const                  quad_index_in)
  {
    this->data       = &mf_data;
    dof_index_vector = dof_index_vector_in;
    dof_index_scalar = dof_index_scalar_in;
    quad_index       = quad_index_in;
  }

  void
  compute_divergence(VectorType & dst, VectorType const & src) const
  {
    data->cell_loop(&This::local_compute_divergence, this, dst, src);
  }

private:
  void
  local_compute_divergence(MatrixFree<dim, value_type> const &           data,
                           VectorType &                                  dst,
                           VectorType const &                            src,
                           std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_vectorial fe_eval_vectorial(data, dof_index_vector, quad_index, 0);
    FEEval_scalar    fe_eval_scalar(data, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_vectorial.reinit(cell);
      fe_eval_vectorial.read_dof_values(src);
      fe_eval_vectorial.evaluate(false, true);

      fe_eval_scalar.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_vectorial.n_q_points; q++)
      {
        VectorizedArray<value_type> divergence = fe_eval_vectorial.get_divergence(q);
        fe_eval_scalar.submit_value(divergence, q);
      }

      fe_eval_scalar.integrate(true, false);
      fe_eval_scalar.set_dof_values(dst);
    }
  }

  MatrixFree<dim, value_type> const * data;

  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;
};


/*
 *  The DoF vector contains the vector of conserved quantities (rho, rho u, rho E).
 *  This class allows to transfer these quantities into the derived variables (p, u, T)
 *  by using L2-projections.
 */
template<int dim, int fe_degree, int n_q_points, typename value_type>
class p_u_T_Calculator
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef p_u_T_Calculator<dim, fe_degree, n_q_points, value_type> This;

  typedef FEEvaluation<dim, fe_degree, n_q_points, 1, value_type>   FEEval_scalar;
  typedef FEEvaluation<dim, fe_degree, n_q_points, dim, value_type> FEEval_vectorial;

  p_u_T_Calculator()
    : data(nullptr),
      dof_index_all(0),
      dof_index_vector(1),
      dof_index_scalar(2),
      quad_index(0),
      heat_capacity_ratio(-1.0),
      specific_gas_constant(-1.0){};

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             unsigned int const                  dof_index_all_in,
             unsigned int const                  dof_index_vector_in,
             unsigned int const                  dof_index_scalar_in,
             unsigned int const                  quad_index_in,
             double const                        heat_capacity_ratio_in,
             double const                        specific_gas_constant_in)
  {
    data                  = &mf_data;
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

    data->cell_loop(&This::local_apply_pressure, this, pressure, solution_conserved);
  }

  void
  compute_velocity(VectorType & velocity, VectorType const & solution_conserved) const
  {
    data->cell_loop(&This::local_apply_velocity, this, velocity, solution_conserved);
  }

  void
  compute_temperature(VectorType & temperature, VectorType const & solution_conserved) const
  {
    AssertThrow(heat_capacity_ratio > 0.0, ExcMessage("heat capacity ratio has not been set!"));
    AssertThrow(specific_gas_constant > 0.0, ExcMessage("specific gas constant has not been set!"));

    data->cell_loop(&This::local_apply_temperature, this, temperature, solution_conserved);
  }

private:
  void
  local_apply_pressure(MatrixFree<dim, value_type> const &           data,
                       VectorType &                                  dst,
                       VectorType const &                            src,
                       std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    FEEval_scalar    fe_eval_density(data, dof_index_all, quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, dof_index_all, quad_index, 1);
    FEEval_scalar    fe_eval_energy(data, dof_index_all, quad_index, 1 + dim);

    // dst-vector
    FEEval_scalar fe_eval_pressure(data, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      fe_eval_density.reinit(cell);
      fe_eval_density.read_dof_values(src);
      fe_eval_density.evaluate(true, false, false);
      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.read_dof_values(src);
      fe_eval_momentum.evaluate(true, false, false);
      fe_eval_energy.reinit(cell);
      fe_eval_energy.read_dof_values(src);
      fe_eval_energy.evaluate(true, false, false);

      // dst-vector
      fe_eval_pressure.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_energy.n_q_points; ++q)
      {
        // conserved variables
        VectorizedArray<value_type>                 rho   = fe_eval_density.get_value(q);
        Tensor<1, dim, VectorizedArray<value_type>> rho_u = fe_eval_momentum.get_value(q);
        VectorizedArray<value_type>                 rho_E = fe_eval_energy.get_value(q);

        // compute derived quantities in quadrature point
        Tensor<1, dim, VectorizedArray<value_type>> u = rho_u / rho;

        VectorizedArray<value_type> p =
          make_vectorized_array<value_type>(heat_capacity_ratio - 1.0) *
          (rho_E - 0.5 * rho * scalar_product(u, u));

        fe_eval_pressure.submit_value(p, q);
      }

      fe_eval_pressure.integrate(true, false);
      fe_eval_pressure.set_dof_values(dst);
    }
  }

  void
  local_apply_velocity(MatrixFree<dim, value_type> const &           data,
                       VectorType &                                  dst,
                       VectorType const &                            src,
                       std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    FEEval_scalar    fe_eval_density(data, dof_index_all, quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, dof_index_all, quad_index, 1);

    // dst-vector
    FEEval_vectorial fe_eval_velocity(data, dof_index_vector, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      fe_eval_density.reinit(cell);
      fe_eval_density.read_dof_values(src);
      fe_eval_density.evaluate(true, false, false);
      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.read_dof_values(src);
      fe_eval_momentum.evaluate(true, false, false);

      // dst-vector
      fe_eval_velocity.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_momentum.n_q_points; ++q)
      {
        // conserved variables
        VectorizedArray<value_type>                 rho   = fe_eval_density.get_value(q);
        Tensor<1, dim, VectorizedArray<value_type>> rho_u = fe_eval_momentum.get_value(q);

        // compute derived quantities in quadrature point
        Tensor<1, dim, VectorizedArray<value_type>> u = rho_u / rho;

        fe_eval_velocity.submit_value(u, q);
      }

      fe_eval_velocity.integrate(true, false);
      fe_eval_velocity.set_dof_values(dst);
    }
  }

  void
  local_apply_temperature(MatrixFree<dim, value_type> const &           data,
                          VectorType &                                  dst,
                          VectorType const &                            src,
                          std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    // src-vector
    FEEval_scalar    fe_eval_density(data, dof_index_all, quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, dof_index_all, quad_index, 1);
    FEEval_scalar    fe_eval_energy(data, dof_index_all, quad_index, 1 + dim);

    // dst-vector
    FEEval_scalar fe_eval_temperature(data, dof_index_scalar, quad_index, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // src-vector
      fe_eval_density.reinit(cell);
      fe_eval_density.read_dof_values(src);
      fe_eval_density.evaluate(true, false, false);
      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.read_dof_values(src);
      fe_eval_momentum.evaluate(true, false, false);
      fe_eval_energy.reinit(cell);
      fe_eval_energy.read_dof_values(src);
      fe_eval_energy.evaluate(true, false, false);

      // dst-vector
      fe_eval_temperature.reinit(cell);

      for(unsigned int q = 0; q < fe_eval_energy.n_q_points; ++q)
      {
        // conserved variables
        VectorizedArray<value_type>                 rho   = fe_eval_density.get_value(q);
        Tensor<1, dim, VectorizedArray<value_type>> rho_u = fe_eval_momentum.get_value(q);
        VectorizedArray<value_type>                 rho_E = fe_eval_energy.get_value(q);

        // compute derived quantities in quadrature point
        Tensor<1, dim, VectorizedArray<value_type>> u = rho_u / rho;
        VectorizedArray<value_type>                 E = rho_E / rho;

        VectorizedArray<value_type> T =
          make_vectorized_array<value_type>((heat_capacity_ratio - 1.0) / specific_gas_constant) *
          (E - 0.5 * scalar_product(u, u));

        fe_eval_temperature.submit_value(T, q);
      }

      fe_eval_temperature.integrate(true, false);
      fe_eval_temperature.set_dof_values(dst);
    }
  }

  MatrixFree<dim, value_type> const * data;

  unsigned int dof_index_all;
  unsigned int dof_index_vector;
  unsigned int dof_index_scalar;
  unsigned int quad_index;

  double heat_capacity_ratio;
  double specific_gas_constant;
};

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_COMP_NAVIER_STOKES_CALCULATORS_H_ \
        */
