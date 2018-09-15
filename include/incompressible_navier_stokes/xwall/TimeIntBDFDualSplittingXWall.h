/*
 * TimeIntBDFDualSplittingXWall.h
 *
 *  Created on: Jul 7, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_TIMEINTBDFDUALSPLITTINGXWALL_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_TIMEINTBDFDUALSPLITTINGXWALL_H_

#include "../../incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename value_type>
class TimeIntBDFDualSplittingXWall
  : public virtual TimeIntBDFDualSplitting<dim,
                                           fe_degree,
                                           value_type,
                                           DGNavierStokesDualSplittingXWall<dim,
                                                                            fe_degree,
                                                                            fe_degree_p,
                                                                            fe_degree_xwall,
                                                                            xwall_quad_rule>>
{
public:
  TimeIntBDFDualSplittingXWall(
    std::shared_ptr<
      DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>>
                                             ns_operation_in,
    std::shared_ptr<PostProcessorBase<dim>>  postprocessor_in,
    InputParametersNavierStokes<dim> const & param_in,
    unsigned int const                       n_refine_time_in,
    bool const                               use_adaptive_time_stepping)
    : TimeIntBDFDualSplitting<dim,
                              fe_degree,
                              value_type,
                              DGNavierStokesDualSplittingXWall<dim,
                                                               fe_degree,
                                                               fe_degree_p,
                                                               fe_degree_xwall,
                                                               xwall_quad_rule>>(
        ns_operation_in,
        postprocessor_in,
        param_in,
        n_refine_time_in,
        use_adaptive_time_stepping),
      ns_operation_xwall(
        std::dynamic_pointer_cast<DGNavierStokesDualSplittingXWall<dim,
                                                                   fe_degree,
                                                                   fe_degree_p,
                                                                   fe_degree_xwall,
                                                                   xwall_quad_rule>>(
          ns_operation_in))
  {
  }

  virtual ~TimeIntBDFDualSplittingXWall()
  {
  }

  virtual void
  solve_timestep();

protected:
  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia)
  {
    TimeIntBDFDualSplitting<
      dim,
      fe_degree,
      value_type,
      DGNavierStokesDualSplittingXWall<dim,
                                       fe_degree,
                                       fe_degree_p,
                                       fe_degree_xwall,
                                       xwall_quad_rule>>::read_restart_vectors(ia);

    Vector<double> tmp;
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), ns_operation_xwall->get_fe_parameters().tauw->begin());
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), ns_operation_xwall->get_fe_parameters_n().tauw->begin());
    ns_operation_xwall->get_fe_parameters().tauw->update_ghost_values();
    ns_operation_xwall->get_fe_parameters_n().tauw->update_ghost_values();

    ns_operation_xwall->precompute_spaldings_law();
  }

  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const
  {
    TimeIntBDFDualSplitting<
      dim,
      fe_degree,
      value_type,
      DGNavierStokesDualSplittingXWall<dim,
                                       fe_degree,
                                       fe_degree_p,
                                       fe_degree_xwall,
                                       xwall_quad_rule>>::write_restart_vectors(oa);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    VectorView<double> tmp(ns_operation_xwall->get_fe_parameters().tauw->local_size(),
                           ns_operation_xwall->get_fe_parameters().tauw->begin());
#pragma GCC diagnostic pop
    oa << tmp;

    tmp.reinit(ns_operation_xwall->get_fe_parameters_n().tauw->local_size(),
               ns_operation_xwall->get_fe_parameters_n().tauw->begin());
    oa << tmp;
  }

protected:
  virtual void
  setup_derived();

  std::shared_ptr<
    DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule>>
    ns_operation_xwall;
};

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename value_type>
void
TimeIntBDFDualSplittingXWall<dim,
                             fe_degree,
                             fe_degree_p,
                             fe_degree_xwall,
                             xwall_quad_rule,
                             value_type>::setup_derived()
{
  ns_operation_xwall->precompute_inverse_mass_matrix();


  TimeIntBDFDualSplitting<dim,
                          fe_degree,
                          value_type,
                          DGNavierStokesDualSplittingXWall<dim,
                                                           fe_degree,
                                                           fe_degree_p,
                                                           fe_degree_xwall,
                                                           xwall_quad_rule>>::setup_derived();
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename value_type>
void
TimeIntBDFDualSplittingXWall<dim,
                             fe_degree,
                             fe_degree_p,
                             fe_degree_xwall,
                             xwall_quad_rule,
                             value_type>::solve_timestep()
{
  // set the parameters that NavierStokesOperation depends on
  // this->velocity[0].print(std::cout)
  Timer timer;
  timer.restart();
  ns_operation_xwall->update_tauw(this->velocity[0]);
  if(this->param.variabletauw)
  {
    ns_operation_xwall->precompute_inverse_mass_matrix();
    ns_operation_xwall->xwall_projection(this->velocity_np);
    for(unsigned int o = 0; o < this->order; o++)
      ns_operation_xwall->xwall_projection(this->velocity[o]);
    for(unsigned int o = 1; o < this->param.order_time_integrator; o++)
    {
      ns_operation_xwall->evaluate_convective_term(this->vec_convective_term[o],
                                                   this->velocity[o],
                                                   this->time - this->time_steps[o]);
      ns_operation_xwall->apply_inverse_mass_matrix(this->vec_convective_term[o],
                                                    this->vec_convective_term[o]);
      ns_operation_xwall->compute_vorticity(this->vorticity[o], this->velocity[o]);
    }
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
       this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl
                << "Precompute inverse mass matrix:" << std::endl
                << "                     " << std::setw(4) << std::right << " "
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }

  TimeIntBDFDualSplitting<dim,
                          fe_degree,
                          value_type,
                          DGNavierStokesDualSplittingXWall<dim,
                                                           fe_degree,
                                                           fe_degree_p,
                                                           fe_degree_xwall,
                                                           xwall_quad_rule>>::solve_timestep();
}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_TIMEINTBDFDUALSPLITTINGXWALL_H_ */
