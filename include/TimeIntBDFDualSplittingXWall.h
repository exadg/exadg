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
    ns_operation_xwall (std::dynamic_pointer_cast<DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > (ns_operation_in))
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
std::cout << "test" << std::endl;
  ns_operation_xwall->precompute_inverse_mass_matrix();


  TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::setup_derived();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve_timestep()
{
  // set the parameters that NavierStokesOperation depends on

  ns_operation_xwall->update_tauw(this->velocity[0]);

  if(this->param.variabletauw)
  {
    ns_operation_xwall->precompute_inverse_mass_matrix();
    for (unsigned int o=0; o < this->order; o++)
      ns_operation_xwall->xwall_projection(this->velocity[o]);
    for (unsigned int o = 1; o < this->param.order_time_integrator; o++)
    {
      ns_operation_xwall->evaluate_convective_term(this->vec_convective_term[o],this->velocity[o],this->time - this->time_steps[o]);
      ns_operation_xwall->compute_vorticity(this->vorticity[o], this->velocity[o]);
    }
  }

  TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::solve_timestep();
}

#endif /* INCLUDE_TIMEINTBDFDUALSPLITTINGXWALL_H_ */
