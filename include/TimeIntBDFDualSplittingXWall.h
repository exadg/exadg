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
                          std_cxx11::shared_ptr<PostProcessor<dim, fe_degree,
                          fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> >    postprocessor_in,
                          InputParametersNavierStokes const                       &param_in,
                          unsigned int const                                      n_refine_time_in,
                          bool const                                              use_adaptive_time_stepping)
    :
    TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
            (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
    ns_operation_xwall (std::dynamic_pointer_cast<DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > (ns_operation_in))
  {
    AssertThrow(this->param.start_with_low_order == true, ExcMessage("Start with low order for xwall"));
  }

  virtual ~TimeIntBDFDualSplittingXWall(){}

  virtual void solve_timestep();

private:

  virtual void setup_derived();

  std_cxx11::shared_ptr<DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> >
     ns_operation_xwall;

};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
setup_derived()
{
  ns_operation_xwall->precompute_inverse_mass_matrix();


  TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::setup_derived();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve_timestep()
{
  // set the parameters that NavierStokesOperation depends on
//this->velocity[0].print(std::cout)
  Timer timer;
  timer.restart(); 
  ns_operation_xwall->update_tauw(this->velocity[0]);
//  ns_operation_xwall->get_fe_parameters().tauw->print(std::cout);
  if(this->param.variabletauw)
  {
    ns_operation_xwall->precompute_inverse_mass_matrix();
    ns_operation_xwall->xwall_projection(this->velocity_np);
    for (unsigned int o=0; o < this->order; o++)
      ns_operation_xwall->xwall_projection(this->velocity[o]);
    for (unsigned int o = 1; o < this->param.order_time_integrator; o++)
    {
      ns_operation_xwall->evaluate_convective_term(this->vec_convective_term[o],this->velocity[o],this->time - this->time_steps[o]);
      ns_operation_xwall->apply_inverse_mass_matrix(this->vec_convective_term[o],this->vec_convective_term[o]);
      ns_operation_xwall->compute_vorticity(this->vorticity[o], this->velocity[o]);
    }
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Precompute inverse mass matrix:" << std::endl
                << "                     " << std::setw(4) << std::right << " " << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }

  TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::solve_timestep();
}

#endif /* INCLUDE_TIMEINTBDFDUALSPLITTINGXWALL_H_ */
