
#ifndef __indexa_statistics_manager_h
#define __indexa_statistics_manager_h

#include <deal.II/lac/parallel_vector.h>
#include <fluid_base_algorithm.h>
#include "../../incompressible_navier_stokes/infrastructure/fe_parameters.h"

using namespace IncNS;

template<int dim>
class StatisticsManager
{
public:
  StatisticsManager(const DoFHandler<dim> & dof_handler_velocity, const Mapping<dim> & mapping);

  // The argument grid_transform indicates how the y-direction that is initially distributed from
  // [0,1] is mapped to the actual grid. This must match the transformation applied to the
  // triangulation, otherwise the identification of data will fail
  void
  setup(std::function<double(double const &)> const & grid_tranform,
        TurbulentChannelData const &                  turb_channel_data);

  void
  evaluate(parallel::distributed::Vector<double> const & velocity,
           double const &                                time,
           unsigned int const &                          time_step_number);

  void
  evaluate(const parallel::distributed::Vector<double> & velocity);

  void
  evaluate(const std::vector<parallel::distributed::Vector<double>> & velocity);

  void
  evaluate_xwall(const parallel::distributed::Vector<double> & velocity,
                 const DoFHandler<dim> &                       dof_handler_wdist,
                 const FEParameters<dim> &                     fe_param,
                 const double                                  viscosity);

  void
  write_output(const std::string output_prefix,
               const double      dynamic_viscosity,
               const double      density);

  void
  reset();

private:
  static const unsigned int n_points_y_per_cell_linear = 11;
  unsigned int              n_points_y_per_cell;

  void
  do_evaluate(const std::vector<const parallel::distributed::Vector<double> *> & velocity);

  void
  do_evaluate_xwall(const std::vector<const parallel::distributed::Vector<double> *> & velocity,
                    const DoFHandler<dim> &   dof_handler_wdist,
                    const FEParameters<dim> & fe_param,
                    const double              viscosity);

  const DoFHandler<dim> & dof_handler;
  const Mapping<dim> &    mapping;
  MPI_Comm                communicator;

  // vector of y-coordinates at which statistical quantities are computed
  std::vector<double> y_glob;

  // mean velocity <u_i>, i=1,...,d (for all y-coordinates)
  std::vector<std::vector<double>> vel_glob;
  // square velocity <u_i²>, i=1,...,d (for all y-coordinates)
  std::vector<std::vector<double>> velsq_glob;
  // <u_1*u_2> = <u*v> (for all y-coordinates)
  std::vector<double> veluv_glob;

  // number of samples
  int number_of_samples;

  bool                 write_final_output;
  TurbulentChannelData turb_channel_data;
};


#endif
