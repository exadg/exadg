/*
 * TimeIntBDFDualSplittingXWall.h
 *
 *  Created on: Jul 7, 2016
 *      Author: krank
 */

#ifndef INCLUDE_TIMEINTBDFDUALSPLITTINGXWALL_H_
#define INCLUDE_TIMEINTBDFDUALSPLITTINGXWALL_H_

#include "TimeIntBDFDualSplitting.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class TimeIntBDFDualSplittingXWall : public TimeIntBDFDualSplitting<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall,value_type>
{
public:
  TimeIntBDFDualSplittingXWall(std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree,
                            fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> >  ns_operation_in,
                          std_cxx11::shared_ptr<PostProcessor<dim> >              postprocessor_in,
                          InputParameters const                                   &param_in,
                          unsigned int const                                      n_refine_time_in)
    :
    TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
            (ns_operation_in,postprocessor_in,param_in,n_refine_time_in),
    ns_operation_xwall (std::dynamic_pointer_cast<DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > (this->ns_operation))
  {
    AssertThrow(this->param.start_with_low_order == true, ExcMessage("Start with low order for xwall"));
  }

  virtual ~TimeIntBDFDualSplittingXWall(){}

private:

  virtual void setup_derived();

  virtual void solve_timestep();

  std_cxx11::shared_ptr<DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> >
     ns_operation_xwall;

};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
setup_derived()
{
  if(not this->param.variabletauw)
  {
    ns_operation_xwall->precompute_inverse_mass_matrix();
  }

  this->setup_derived();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve_timestep()
{
  // set the parameters that NavierStokesOperation depends on
  ns_operation_xwall->set_time(this->time);
  ns_operation_xwall->set_time_step(this->time_steps[0]);
  ns_operation_xwall->set_scaling_factor_time_derivative_term(this->gamma0/this->time_steps[0]);

  ns_operation_xwall->update_tauw(this->velocity);
  if(this->param.variabletauw)
  {
    ns_operation_xwall->precompute_inverse_mass_matrix();
    ns_operation_xwall->xwall_projection();
    for (unsigned int o = 1; o < this->param.order; o++)
    {
      ns_operation_xwall->evaluate_convective_term(this->vec_convective_term[o],this->velocity[o],this->time - this->time_steps[o]);
      ns_operation_xwall->compute_vorticity(this->vorticity[o], this->velocity[o]);
    }
  }


  // perform the four substeps of the dual-splitting method
  this->convective_step();

  this->pressure_step();

  this->projection_step();

  this->viscous_step();
}

#endif /* INCLUDE_TIMEINTBDFDUALSPLITTINGXWALL_H_ */
