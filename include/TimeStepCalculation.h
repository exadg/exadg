/*
 * TimeStepCalculation.h
 *
 *  Created on: May 30, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMESTEPCALCULATION_H_
#define INCLUDE_TIMESTEPCALCULATION_H_

#define CFL_BASED_ON_MINIMUM_COMPONENT

#include "SpaldingsLaw.h"

double calculate_const_time_step(double const dt,
                                 unsigned int const n_refine_time)
{
  double time_step = dt/std::pow(2.,n_refine_time);

  return time_step;
}

template<int dim>
double calculate_min_cell_diameter(Triangulation<dim> const &triangulation)
{
  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();

  double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();
  for (; cell!=endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
      diameter = cell->minimum_vertex_distance();
      if (diameter < min_cell_diameter)
        min_cell_diameter = diameter;
    }
  }
  const double global_min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, MPI_COMM_WORLD);

  return global_min_cell_diameter;
}

double calculate_const_time_step_cfl(double const       cfl,
                                     double const       max_velocity,
                                     double const       global_min_cell_diameter,
                                     unsigned int const fe_degree,
                                     double const       exponent_fe_degree = 2.0)
{
  // cfl/p^{exponent_fe_degree} = || U || * time_step / h
  double time_step = cfl/pow(fe_degree,exponent_fe_degree) * global_min_cell_diameter / max_velocity;

  return time_step;
}

template<int dim, int fe_degree, typename value_type>
double calculate_adaptive_time_step_cfl(MatrixFree<dim,value_type> const                &data,
                                        unsigned int const                              dof_index,
                                        unsigned int const                              quad_index,
                                        parallel::distributed::Vector<value_type> const &velocity,
                                        double const                                    cfl,
                                        double const                                    last_time_step,
                                        bool const                                      use_limiter = true,
                                        double const                                    exponent_fe_degree = 1.5)
{
  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,dof_index,quad_index);

  value_type new_time_step = std::numeric_limits<value_type>::max();
  value_type cfl_p = cfl/pow(fe_degree,exponent_fe_degree);

  // loop over cells of processor
  for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
  {
    VectorizedArray<value_type> delta_t_cell =
        make_vectorized_array<value_type>(std::numeric_limits<value_type>::max());
    Tensor<2,dim,VectorizedArray<value_type> > invJ;
    Tensor<1,dim,VectorizedArray<value_type> > u_x;
    Tensor<1,dim,VectorizedArray<value_type> > ut_xi;

    fe_eval.reinit(cell);
    fe_eval.read_dof_values(velocity);
    fe_eval.evaluate(true,false);

    // loop over quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      u_x = fe_eval.get_value(q);
      invJ = fe_eval.inverse_jacobian(q);
      invJ = transpose(invJ);
      ut_xi = invJ*u_x;

#ifdef CFL_BASED_ON_MINIMUM_COMPONENT
      for (unsigned int d = 0; d < dim; ++d)
        delta_t_cell = std::min(delta_t_cell,cfl_p/(std::abs(ut_xi[d])));
#else
      delta_t_cell = std::min(delta_t_cell,cfl_p/ut_xi.norm());
#endif
    }

    // loop over vectorized array
    value_type dt = std::numeric_limits<value_type>::max();
    for (unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
    {
      dt = std::min(dt,delta_t_cell[v]);
    }

    new_time_step = std::min(new_time_step,dt);
  }

  // find minimum over all processors
  new_time_step = Utilities::MPI::min(new_time_step, MPI_COMM_WORLD);

  // limit the maximum increase/decrease of the time step size
  if(use_limiter == true)
  {
    double fac = 1.2;
    if (new_time_step >= fac*last_time_step)
    {
      new_time_step = fac*last_time_step;
    }

    else if (new_time_step <= last_time_step/fac)
    {
      new_time_step = last_time_step/fac;
    }
  }

  return new_time_step;
}

template<int dim, int fe_degree, int fe_degree_xwall, typename value_type>
double calculate_adaptive_time_step_cfl_xwall(MatrixFree<dim,value_type> const                &data,
                                              unsigned int const                              dof_index,
                                              unsigned int const                              dof_index_tauw,
                                              unsigned int const                              quad_index,
                                              parallel::distributed::Vector<value_type> const &velocity,
                                              parallel::distributed::Vector<value_type> const &wdist,
                                              parallel::distributed::Vector<value_type> const &tauw,
                                              double const                                    viscosity,
                                              double const                                    cfl,
                                              double const                                    last_time_step,
                                              bool const                                      use_limiter = true,
                                              double const                                    exponent_fe_degree = 1.5)
{
  //implement enriched velocity manually such that these functions are independent and can be precompiled
  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,dof_index,quad_index);
  FEEvaluation<dim,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,dof_index,quad_index,dim);
  FEEvaluation<dim,1,fe_degree+1,1,value_type> fe_eval_tauw(data,dof_index_tauw,quad_index);

  value_type new_time_step = std::numeric_limits<value_type>::max();
  value_type cfl_p = cfl/pow(fe_degree,exponent_fe_degree);

  AlignedVector<VectorizedArray<value_type> > wdist_loc;
  AlignedVector<VectorizedArray<value_type> > tauw_loc;
  // loop over cells of processor
  for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
  {
    VectorizedArray<value_type> delta_t_cell =
        make_vectorized_array<value_type>(std::numeric_limits<value_type>::max());
    Tensor<2,dim,VectorizedArray<value_type> > invJ;
    Tensor<1,dim,VectorizedArray<value_type> > u_x;
    Tensor<1,dim,VectorizedArray<value_type> > ut_xi;

    fe_eval.reinit(cell);
    fe_eval.read_dof_values(velocity);
    fe_eval.evaluate(true,false);
    fe_eval_xwall.reinit(cell);
    fe_eval_xwall.read_dof_values(velocity);
    fe_eval_xwall.evaluate(true,false);
    fe_eval_tauw.reinit(cell);
    fe_eval_tauw.read_dof_values(wdist);
    fe_eval_tauw.evaluate(true,false);
    wdist_loc.resize(fe_eval.n_q_points);
    tauw_loc.resize(fe_eval.n_q_points);
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      wdist_loc[q] = fe_eval_tauw.get_value(q);
    fe_eval_tauw.read_dof_values(tauw);
    fe_eval_tauw.evaluate(true,false);
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      tauw_loc[q] = fe_eval_tauw.get_value(q);

    SpaldingsLawEvaluation<dim, value_type, VectorizedArray<value_type> > spalding(viscosity);
    spalding.reinit(wdist_loc,tauw_loc,fe_eval.n_q_points);

    // loop over quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      u_x = fe_eval.get_value(q) + fe_eval_xwall.get_value(q)*spalding.enrichment(q);
      invJ = fe_eval.inverse_jacobian(q);
      invJ = transpose(invJ);
      ut_xi = invJ*u_x;

#ifdef CFL_BASED_ON_MINIMUM_COMPONENT
      for (unsigned int d = 0; d < dim; ++d)
        delta_t_cell = std::min(delta_t_cell,cfl_p/(std::abs(ut_xi[d])));
#else
      delta_t_cell = std::min(delta_t_cell,cfl_p/ut_xi.norm());
#endif
    }

    // loop over vectorized array
    value_type dt = std::numeric_limits<value_type>::max();
    for (unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
    {
      dt = std::min(dt,delta_t_cell[v]);
    }

    new_time_step = std::min(new_time_step,dt);
  }

  // find minimum over all processors
  new_time_step = Utilities::MPI::min(new_time_step, MPI_COMM_WORLD);

  // limit the maximum increase/decrease of the time step size
  if(use_limiter == true)
  {
    double fac = 1.2;
    if (new_time_step >= fac*last_time_step)
    {
      new_time_step = fac*last_time_step;
    }

    else if (new_time_step <= last_time_step/fac)
    {
      new_time_step = last_time_step/fac;
    }
  }

  return new_time_step;
}

template<int dim, int fe_degree, typename value_type>
double calculate_adaptive_time_step_diffusion(MatrixFree<dim,value_type> const                &data,
                                              unsigned int const                              dof_index,
                                              unsigned int const                              quad_index,
                                              parallel::distributed::Vector<value_type> const &vt,
                                              double const                                    viscosity,
                                              double const                                    d,
                                              double const                                    last_time_step,
                                              bool const                                      use_limiter = true,
                                              double const                                    exponent_fe_degree = 3)
{
  FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,dof_index,quad_index);

  value_type new_time_step = std::numeric_limits<value_type>::max();
  value_type d_p = d/pow(fe_degree,exponent_fe_degree);

  // loop over cells of processor
  for (unsigned int cell=0; cell<data.n_macro_cells(); ++cell)
  {
    VectorizedArray<value_type> h;
    for(unsigned int v = 0; v<VectorizedArray<value_type>::n_array_elements;v++)
    {
      typename DoFHandler<dim>::cell_iterator dcell = data.get_cell_iterator(cell, v);
      h[v] = dcell->minimum_vertex_distance();
    }
    VectorizedArray<value_type> delta_t_cell =
        make_vectorized_array<value_type>(std::numeric_limits<value_type>::max());
    VectorizedArray<value_type> vt_val;

    fe_eval.reinit(cell);
    fe_eval.read_dof_values(vt);
    fe_eval.evaluate(true,false);

    // loop over quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      vt_val = fe_eval.get_value(q);

      //TODO Benjamin: use local shortest length, not global one
      //vt should be larger than zero
        delta_t_cell = std::min(delta_t_cell,d_p*h*h/(std::abs((vt_val+viscosity)*1.5)));
    }

    // loop over vectorized array
    value_type dt = std::numeric_limits<value_type>::max();
    for (unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
    {
      dt = std::min(dt,delta_t_cell[v]);
    }

    new_time_step = std::min(new_time_step,dt);
  }

  // find minimum over all processors
  new_time_step = Utilities::MPI::min(new_time_step, MPI_COMM_WORLD);

  // limit the maximum increase/decrease of the time step size
  if(use_limiter == true)
  {
    double fac = 1.2;
    if (new_time_step >= fac*last_time_step)
    {
      new_time_step = fac*last_time_step;
    }

    else if (new_time_step <= last_time_step/fac)
    {
      new_time_step = last_time_step/fac;
    }
  }

  return new_time_step;
}


#endif /* INCLUDE_TIMESTEPCALCULATION_H_ */
