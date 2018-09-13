/*
 * TimeIntBDFDualSplittingXWallSpalartAllmaras.h
 *
 *  Created on: Aug 1, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_TIMEINTBDFDUALSPLITTINGXWALLSPALARTALLMARAS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_TIMEINTBDFDUALSPLITTINGXWALLSPALARTALLMARAS_H_

#include "../../incompressible_navier_stokes/xwall/DGSpalartAllmarasModel.h"
#include "../../incompressible_navier_stokes/xwall/TimeIntBDFDualSplittingXWall.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class TimeIntBDFDualSplittingXWallSpalartAllmaras : virtual public TimeIntBDFDualSplittingXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>
{
public:
  TimeIntBDFDualSplittingXWallSpalartAllmaras(
      std::shared_ptr<DGNavierStokesBase<dim, fe_degree,
        fe_degree_p, fe_degree_xwall, xwall_quad_rule> >  ns_operation_in,
      std::shared_ptr<PostProcessorBase<dim> >          postprocessor_in,
      InputParametersNavierStokes<dim> const                  &param_in,
      unsigned int const                                      n_refine_time_in,
      bool const                                              use_adaptive_time_stepping)
    :
    TimeIntBDFDualSplitting<dim, fe_degree, value_type, DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >
            (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
    TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
            (ns_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
    vt(this->order),
    time_steps_sa(this->order),
    num_sa_substeps(1),
    ns_operation_xwall_sa (std::dynamic_pointer_cast<DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > (ns_operation_in)),
    vt_rhs(this->order),
    gamma0_sa(1.0),
    alpha_sa(this->order),
    beta_sa(this->order)
  {
  }

  virtual ~TimeIntBDFDualSplittingXWallSpalartAllmaras(){}

  std::vector<parallel::distributed::Vector<value_type> > vt;

protected:

  void read_restart_data_sa(boost::archive::binary_iarchive & ia)
  {
    Vector<double> tmp;
    for (unsigned int i=0; i<this->vt.size(); i++)
    {
      ia >> tmp;
      std::copy(tmp.begin(), tmp.end(),
                this->vt[i].begin());
    }

    for (unsigned int i=0; i<this->vt_rhs.size(); i++)
    {
      ia >> tmp;
      std::copy(tmp.begin(), tmp.end(),
                this->vt_rhs[i].begin());
    }

    for (unsigned int i = 0; i < this->order; i++)
      ia & time_steps_sa[i];

  }
  void write_restart_data_sa(boost::archive::binary_oarchive & oa) const
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    VectorView<double> tmp(this->vt[0].local_size(),
                           this->vt[0].begin());
#pragma GCC diagnostic pop
    oa << tmp;
    for (unsigned int i=1; i<this->vt.size(); i++)
    {
      tmp.reinit(this->vt[i].local_size(),
                 this->vt[i].begin());
      oa << tmp;
    }

    for (unsigned int i=0; i<this->vt_rhs.size(); i++)
    {
      tmp.reinit(this->vt_rhs[i].local_size(),
                 this->vt_rhs[i].begin());
      oa << tmp;
    }

    for (unsigned int i = 0; i< this->order;i++)
      oa & time_steps_sa[i];
  }

private:
  virtual void postprocessing() const;

  virtual void solve_timestep();

  void viscous_step();

  virtual void initialize_vectors();

  void prepare_vectors_for_next_timestep();

  void calculate_time_step();
  void recalculate_adaptive_time_step();

  void update_time_integrator_constants();

  std::vector<value_type> time_steps_sa;

  unsigned int num_sa_substeps;

  std::shared_ptr<DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >
     ns_operation_xwall_sa;

  parallel::distributed::Vector<value_type> vt_np;
  std::vector<parallel::distributed::Vector<value_type> > vt_rhs;

  value_type gamma0_sa;
  std::vector<value_type> alpha_sa, beta_sa;

};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
initialize_vectors()
{
  TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
solve_timestep()
{
  for(unsigned int n=0;n<num_sa_substeps; n++)
  {
    ns_operation_xwall_sa->evaluate_spalart_allmaras(&(this->velocity[0]),vt[0],vt_rhs[0]);

    vt_np = 0;
    for(unsigned int i=0;i<vt_rhs.size();++i)
      vt_np.add(-this->beta_sa[i]*this->time_steps_sa[0],vt_rhs[i]);

    for (unsigned int i=0;i<vt.size();++i)
      vt_np.add(this->alpha_sa[i],vt[i]);

    vt_np /= gamma0_sa;
    //swap if this is not the last substep
    if(n < num_sa_substeps - 1)
    {
      // solution at t_{n-i} <-- solution at t_{n-i+1}
      for(unsigned int i=vt.size()-1; i>0; --i)
      {
        vt[i].swap(vt[i-1]);
        vt_rhs[i].swap(vt_rhs[i-1]);
      }
      vt[0].swap(vt_np);
    }
    if(this->param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
    {
      for(unsigned int i=this->order-1;i>0;--i)
        time_steps_sa[i] = time_steps_sa[i-1];
      // when starting the time integrator with a low order method, ensure that
      // the time integrator constants are set properly
      if(this->time_step_number <= this->order && this->param.start_with_low_order == true)
      {
        this->set_adaptive_time_integrator_constants(this->time_step_number,time_steps_sa, alpha_sa,beta_sa,gamma0_sa);
      }
      else // otherwise, adjust time integrator constants since this is adaptive time stepping
      {
        this->set_adaptive_time_integrator_constants(this->order,time_steps_sa, alpha_sa,beta_sa,gamma0_sa);
      }
    }
  }
//  vt_np.print(std::cout);
  TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::solve_timestep();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
prepare_vectors_for_next_timestep()
{
  TimeIntBDFDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
  prepare_vectors_for_next_timestep();

  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=vt.size()-1; i>0; --i)
  {
    vt[i].swap(vt[i-1]);
    vt_rhs[i].swap(vt_rhs[i-1]);
  }
  vt[0].swap(vt_np);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
postprocessing() const
{
  this->postprocessor->do_postprocessing(this->velocity[0],this->intermediate_velocity,this->pressure[0],this->vorticity[0],vt[0],this->time,this->time_step_number);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
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

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
calculate_time_step()
{
  //there is no enrichment in the first time step
  TimeIntBDFNavierStokes<dim, fe_degree, value_type, DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >::
  calculate_time_step();

  if(this->param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    time_steps_sa[0] = calculate_adaptive_time_step_diffusion<dim, fe_degree, value_type>
                                          (ns_operation_xwall_sa->get_data(),
                                           ns_operation_xwall_sa->get_dof_index_vt(),
                                           ns_operation_xwall_sa->get_quad_index_velocity_linear(),
                                           vt[0],
                                           this->param.viscosity,
                                           this->param.diffusion_number,
                                           this->time_steps[1],
                                           false);
    time_steps_sa[0] = std::min(this->time_steps[0],time_steps_sa[0]);

    const double EPSILON = 1e-16;
    num_sa_substeps = (unsigned int)ceil(this->time_steps[0]/time_steps_sa[0]-EPSILON);
    time_steps_sa[0] = this->time_steps[0] / (double)num_sa_substeps;
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << "Calculation of time step size according to adaptive Diffusion number condition:" << std::endl << std::endl;

    print_parameter(pcout,"num_sa_substeps",num_sa_substeps);
    print_parameter(pcout,"D",this->param.diffusion_number);
    print_parameter(pcout,"Time step size",time_steps_sa[0]);

  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
recalculate_adaptive_time_step()
{
  /*
   * push back time steps
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *  time_steps-vec:   dt[2]     dt[1]    dt[0]
   *
   *                    dt[1]  <- dt[0] <- recalculate dt[0]
   *
   */

  for(unsigned int i=this->order-1;i>0;--i)
    this->time_steps[i] = this->time_steps[i-1];

  for(unsigned int i=this->order-1;i>0;--i)
    time_steps_sa[i] = time_steps_sa[i-1];

  this->time_steps[0] = calculate_adaptive_time_step_cfl_xwall<dim, fe_degree,fe_degree_xwall, value_type>(ns_operation_xwall_sa->get_data(),
                                                                                                           ns_operation_xwall_sa->get_dof_index_velocity(),
                                                                                                           2,
                                                                                                           ns_operation_xwall_sa->get_quad_index_velocity_linear(),
                                                                                                           this->velocity[0],
                                                                                                           *(ns_operation_xwall_sa->get_fe_parameters().wdist),
                                                                                                           *(ns_operation_xwall_sa->get_fe_parameters().tauw),
                                                                                                           this->param.viscosity,
                                                                                                           this->cfl,
                                                                                                           this->time_steps[0]);
  time_steps_sa[0] = calculate_adaptive_time_step_diffusion<dim, fe_degree, value_type>
                                        (ns_operation_xwall_sa->get_data(),
                                         ns_operation_xwall_sa->get_dof_index_vt(),
                                         ns_operation_xwall_sa->get_quad_index_velocity_linear(),
                                         vt[0],
                                         this->param.viscosity,
                                         this->param.diffusion_number,
                                         this->time_steps[1],
                                         false);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  print_parameter(pcout,"Time step size conv",this->time_steps[0]);
  time_steps_sa[0] = std::min(this->time_steps[0],time_steps_sa[0]);
  const double EPSILON = 1e-16;
  num_sa_substeps = (unsigned int)ceil(this->time_steps[0]/time_steps_sa[0]-EPSILON);
  time_steps_sa[0] = this->time_steps[0] / (double)num_sa_substeps;
  print_parameter(pcout,"num_sa_substeps",num_sa_substeps);
  print_parameter(pcout,"Time step size diff",time_steps_sa[0]);

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
update_time_integrator_constants()
{
  TimeIntBDFNavierStokes<dim, fe_degree, value_type, DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> >::
  update_time_integrator_constants();

  if(this->param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepUserSpecified ||
     this->param.calculation_of_time_step_size == TimeStepCalculation::ConstTimeStepCFL)
  {
    AssertThrow(false,ExcMessage("not recommended"));
  }
  else if(this->param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    // when starting the time integrator with a low order method, ensure that
    // the time integrator constants are set properly
    if(this->time_step_number <= this->order && this->param.start_with_low_order == true)
    {
      this->set_adaptive_time_integrator_constants(this->time_step_number,time_steps_sa, alpha_sa,beta_sa,gamma0_sa);
    }
    else // otherwise, adjust time integrator constants since this is adaptive time stepping
    {
      this->set_adaptive_time_integrator_constants(this->order,time_steps_sa, alpha_sa,beta_sa,gamma0_sa);
    }
  }

}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_TIMEINTBDFDUALSPLITTINGXWALLSPALARTALLMARAS_H_ */
