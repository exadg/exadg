/*
 * time_int_bdf_pressure_correction.h
 *
 *  Created on: Oct 26, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_

#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

// forward declarations
template<int dim, typename Number>
class DGNavierStokesPressureCorrection;

template<int dim, typename Number>
class TimeIntBDFPressureCorrection : public TimeIntBDF<dim, Number>
{
private:
  typedef TimeIntBDF<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef DGNavierStokesPressureCorrection<dim, Number> Operator;

public:
  TimeIntBDFPressureCorrection(
    std::shared_ptr<Operator>                       operator_in,
    InputParameters const &                         param_in,
    unsigned int const                              refine_steps_time_in,
    MPI_Comm const &                                mpi_comm_in,
    std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
    std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in = nullptr,
    std::shared_ptr<MatrixFree<dim, Number>>        matrix_free_in = nullptr);

  virtual ~TimeIntBDFPressureCorrection()
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
  update_time_integrator_constants();

  void
  allocate_vectors() override;

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  setup_derived() override;

  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia) override;

  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const override;

  void
  initialize_pressure_on_boundary();

  void
  solve_timestep() override;

  void
  solve_steady_problem();

  double
  evaluate_residual();

  void
  momentum_step();

  void
  rhs_momentum(VectorType & rhs);

  void
  pressure_step(VectorType & pressure_increment);

  void
  projection_step(VectorType const & pressure_increment);

  void
  evaluate_convective_term();

  void
  rhs_projection(VectorType & rhs, VectorType const & pressure_increment) const;

  void
  pressure_update(VectorType const & pressure_increment);

  void
  calculate_chi(double & chi) const;

  void
  rhs_pressure(VectorType & rhs) const;

  void
  prepare_vectors_for_next_timestep() override;

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

  VectorType              velocity_np;
  std::vector<VectorType> velocity;

  VectorType              pressure_np;
  std::vector<VectorType> pressure;

  // incremental formulation of pressure-correction scheme
  unsigned int order_pressure_extrapolation;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_gradient;

  // stores pressure Dirichlet boundary values at previous times
  std::vector<VectorType> pressure_dbc;

  // required for strongly-coupled partitioned FSI
  VectorType pressure_increment_last_iter;
  VectorType velocity_momentum_last_iter;
  VectorType velocity_projection_last_iter;

  // iteration counts
  std::pair<
    unsigned int /* calls */,
    std::tuple<unsigned long long, unsigned long long> /* iteration counts {Newton, linear} */>
    iterations_momentum;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_pressure;
  std::pair<unsigned int /* calls */, unsigned long long /* iteration counts */>
    iterations_projection;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_PRESSURE_CORRECTION_H_ \
        */
