
#ifndef __indexa_statistics_manager_PH_h
#define __indexa_statistics_manager_PH_h

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/base/std_cxx11/function.h>
#include <deal.II/lac/parallel_block_vector.h>
#include <fluid_base_algorithm.h>
#include <deal.II/fe/mapping_q.h>
#include <fstream>

#include "../../incompressible_navier_stokes/infrastructure/fe_parameters.h"

template <int dim>
class StatisticsManagerPH //: public StatisticsManager<dim>
{
public:

  StatisticsManagerPH(const DoFHandler<dim> &dof_handler_velocity, const DoFHandler<dim> &dof_handler_pressure, const Mapping<dim> &mapping, const bool dns = false):
  dof_handler (dof_handler_velocity),
  dof_handler_p (dof_handler_pressure),
  communicator (dynamic_cast<const parallel::Triangulation<dim>*>(&dof_handler_velocity.get_triangulation()) ?
                (dynamic_cast<const parallel::Triangulation<dim>*>(&dof_handler_velocity.get_triangulation())
                 ->get_communicator()) :
                MPI_COMM_SELF),
                h(0.028),
                y_max(3.036*h),
                y_min(h),
                x_max(9.0*h),
                mapping_(mapping),
              //  mapping_(dynamic_cast<MappingQGeneric<dim> const &>(mapping)),
                numchsamp(0),
                n_points_y(0),
                n_points_y_glob(0),
                n_points_x_glob(0),
                DNS(dns)
  {};

  void setup(const Function<dim> &push_forward_function, const std::string output_prefix, const bool enriched);

  void evaluate(const parallel::distributed::Vector<double> &velocity,const parallel::distributed::Vector<double> &pressure);

  void evaluate(const std::vector<parallel::distributed::Vector<double> > &velocity,const parallel::distributed::Vector<double> &pressure);

  void evaluate(const parallel::distributed::BlockVector<double> &velocity,const parallel::distributed::Vector<double> &pressure);

  void evaluate_xwall(const parallel::distributed::Vector<double> &velocity,
                      const parallel::distributed::Vector<double> &pressure,
                      const DoFHandler<dim>                       &dof_handler_wdist,
                      const FEParameters<dim>                     &fe_param);

  void write_output(const std::string output_prefix,
                    const double      viscosity,
                    unsigned int      statistics_number = 0);

  void reset();

private:
  const DoFHandler<dim> &dof_handler;
  const DoFHandler<dim> &dof_handler_p;
  MPI_Comm communicator;
//  static const int n_points_y = 21;
  const double h;// = 0.028;
  const double y_max;// = 3.036*h;
  const double y_min;// = h;
  const double x_max;// = 9.0*h;
  const Mapping<dim> &mapping_;
  int numchsamp;
  void do_evaluate(const std::vector<const parallel::distributed::Vector<double> *> &velocity,const parallel::distributed::Vector<double> &pressure);

  void do_evaluate_xwall(const std::vector<const parallel::distributed::Vector<double> *> &velocity,
                         const parallel::distributed::Vector<double> &pressure,
                         const DoFHandler<dim>                       &dof_handler_wdist,
                         const FEParameters<dim>                     &fe_param);
  inline bool exists_test0 (const std::string& name)
  {
      std::ifstream f(name.c_str());
      return f.good();
  }

  // variables for evaluation of velocity at certain points x_over_h
  unsigned int n_points_y;
  unsigned int n_points_y_glob;

  std::vector<double> x_over_h;
  std::vector<std::vector<double> > y_vec_glob;
  std::vector<std::vector<double> > y_h_glob;
  std::vector<std::vector<std::vector<double> > > vel_glob;
  std::vector<std::vector<std::vector<double> > > velsq_glob;
  std::vector<std::vector<double> > veluv_glob;
  std::vector<std::vector<std::vector<double> > > epsii_glob;

  // variables for evaluation of velocity at the lower boundary (tau_w(x), ...)
  static const unsigned int n_points_x = 10;
  unsigned int n_points_x_glob;
  const bool DNS;

  std::vector<double> x_glob;
  std::vector<double> y1_bottom_glob;  // cell high of the fist cell at the domain bottom over fe-degree+1
  std::vector<double> y1_top_glob;
  std::vector<double> dudy_bottom_glob;
  std::vector<double> p_bottom_glob;
  std::vector<double> dudy_top_glob;
  std::vector<double> p_top_glob;

};

#endif
