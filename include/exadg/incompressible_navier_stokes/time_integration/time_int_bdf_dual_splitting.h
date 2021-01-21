/*
 * time_int_bdf_dual_splitting.h
 *
 *  Created on: May 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_

#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>

// TODO
//#define EXTRAPOLATE_ACCELERATION

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

// forward declarations
template<int dim, typename Number>
class DGNavierStokesDualSplitting;

template<int dim, typename Number>
class TimeIntBDFDualSplitting : public TimeIntBDF<dim, Number>
{
private:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef DGNavierStokesDualSplitting<dim, Number> Operator;

public:
  TimeIntBDFDualSplitting(std::shared_ptr<Operator>                       pde_operator_in,
                          InputParameters const &                         param_in,
                          unsigned int const                              refine_steps_time_in,
                          MPI_Comm const &                                mpi_comm_in,
                          bool const                                      print_wall_times_in,
                          std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
                          std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in = nullptr,
                          std::shared_ptr<MatrixFree<dim, Number>>        matrix_free_in = nullptr);

  virtual ~TimeIntBDFDualSplitting()
  {
  }

  void
  postprocessing_stability_analysis();

  void
  print_iterations() const;

  VectorType const &
  get_velocity_np() const;

  VectorType const &
  get_pressure_np() const;

private:
  void
  setup_derived() override;

  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia) override;

  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const override;

  void
  solve_timestep() override;

  void
  allocate_vectors() override;

  void
  prepare_vectors_for_next_timestep() override;

  void
  update_velocity_dbc();

  void
  convective_step();

  void
  evaluate_convective_term();

  void
  update_time_integrator_constants();

  void
  initialize_current_solution() override;

  void
  initialize_former_solutions() override;

  void
  initialize_acceleration_and_velocity_on_boundary();

  void
  pressure_step();

  void
  rhs_pressure(VectorType & rhs) const;

  void
  projection_step();

  void
  rhs_projection(VectorType & rhs) const;

  void
  penalty_step();

  void
  viscous_step();

  void
  rhs_viscous(VectorType & rhs) const;

  void
  solve_steady_problem();

  double
  evaluate_residual();

  VectorType const &
  get_velocity() const;

  VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const;

  VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const;

  void
  set_velocity(VectorType const & velocity, unsigned int const i /* t_{n-i} */);

  void
  set_pressure(VectorType const & pressure, unsigned int const i /* t_{n-i} */);

  std::shared_ptr<Operator> pde_operator;

  std::vector<VectorType> velocity;

  VectorType velocity_np;

  std::vector<VectorType> pressure;

  VectorType pressure_np;

#ifdef EXTRAPOLATE_ACCELERATION
  std::vector<VectorType> acceleration;
#endif
  std::vector<VectorType> velocity_dbc;
  VectorType              velocity_dbc_np;

  // required for strongly-coupled partitioned FSI
  VectorType pressure_last_iter;
  VectorType velocity_projection_last_iter;
  VectorType velocity_viscous_last_iter;

  // iteration counts
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_pressure;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
                                                                                 iterations_projection;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_viscous;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */> iterations_penalty;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_nbc;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_ \
        */
