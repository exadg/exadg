/*
 * TimeIntBDFDualSplittingXWallSpalartAllmaras.h
 *
 *  Created on: Aug 1, 2016
 *      Author: krank
 */

#ifndef INCLUDE_TIMEINTBDFDUALSPLITTINGXWALLSPALARTALLMARAS_H_
#define INCLUDE_TIMEINTBDFDUALSPLITTINGXWALLSPALARTALLMARAS_H_

#include "DGSpalartAllmarasModel.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class TimeIntBDFDualSplittingXWallSpalartAllmaras : public TimeIntBDFDualSplittingXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall,value_type>
{
public:
  TimeIntBDFDualSplittingXWallSpalartAllmaras(
      std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree,
        fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> >  ns_operation_in,
      std_cxx11::shared_ptr<PostProcessor<dim, fe_degree,
        fe_degree_p> >                                        postprocessor_in,
      InputParametersNavierStokes<dim> const                  &param_in,
      unsigned int const                                      n_refine_time_in,
      bool const                                              use_adaptive_time_stepping)
    :
    TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
            (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
    ns_operation_xwall_sa (std::dynamic_pointer_cast<DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > (ns_operation_in)),
    vt(this->order),
    vt_rhs(this->order)
  {
  }

  virtual ~TimeIntBDFDualSplittingXWallSpalartAllmaras(){}

private:
  virtual void postprocessing() const;

  virtual void solve_timestep();

  void viscous_step();

  virtual void initialize_vectors();

  void prepare_vectors_for_next_timestep();

  std_cxx11::shared_ptr<DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> >
     ns_operation_xwall_sa;

  parallel::distributed::Vector<value_type> vt_np;
  std::vector<parallel::distributed::Vector<value_type> > vt;
  std::vector<parallel::distributed::Vector<value_type> > vt_rhs;

};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vectors()
{
  TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
  initialize_vectors();
  // velocity
  for(unsigned int i=0;i<vt.size();++i)
    ns_operation_xwall_sa->initialize_vector_vt(vt[i]);
  ns_operation_xwall_sa->initialize_vector_vt(vt_np);
  for(unsigned int i=0;i<vt.size();++i)
    ns_operation_xwall_sa->prescribe_initial_condition_vt(vt[i]);
  ns_operation_xwall_sa->prescribe_initial_condition_vt(vt_np);

  for(unsigned int i=0;i<vt_rhs.size();++i)
    ns_operation_xwall_sa->initialize_vector_vt(vt_rhs[i]);

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve_timestep()
{
  ns_operation_xwall_sa->evaluate_spalart_allmaras(&(this->velocity[0]),vt[0],vt_rhs[0]);

  vt_np = 0;
  for(unsigned int i=0;i<vt_rhs.size();++i)
    vt_np.add(-this->beta[i]*this->time_steps[0],vt_rhs[i]);

  for (unsigned int i=0;i<vt.size();++i)
    vt_np.add(this->alpha[i],vt[i]);

  vt_np /= this->gamma0;
//  vt_np.print(std::cout);
  TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::solve_timestep();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
prepare_vectors_for_next_timestep()
{
  TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
  prepare_vectors_for_next_timestep();

  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=vt.size()-1; i>0; --i)
  {
    vt[i].swap(vt[i-1]);
    vt_rhs[i].swap(vt_rhs[i-1]);
  }
  vt[0].swap(vt_np);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
postprocessing() const
{
  this->postprocessor->do_postprocessing(this->velocity[0],this->pressure[0],this->vorticity[0],vt[0],this->time,this->time_step_number);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
viscous_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  ns_operation_xwall_sa->rhs_viscous(this->rhs_vec_viscous, this->velocity_np, this->vt_np);

  // extrapolate old solution to get a good initial estimate for the solver
  this->velocity_np = 0;
  for (unsigned int i=0; i<this->velocity.size(); ++i)
    this->velocity_np.add(this->beta[i],this->velocity[i]);

  // solve linear system of equations
  unsigned int iterations_viscous = ns_operation_xwall_sa->solve_viscous(this->velocity_np, this->rhs_vec_viscous);

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "Solve viscous step for velocity u:" << std::endl
              << "  PCG iterations:    " << std::setw(4) << std::right << iterations_viscous << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  this->computing_times[3] += timer.wall_time();
}

#endif /* INCLUDE_TIMEINTBDFDUALSPLITTINGXWALLSPALARTALLMARAS_H_ */
