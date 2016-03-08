
// Navier-Stokes splitting program
// authors: Niklas Fehn, Benjamin Krank, Martin Kronbichler, LNM
// years: 2015-2016

#include <deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/thread_local_storage.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/integrators/laplace.h>

#include <../include/statistics_manager.h>

#include <fstream>
#include <sstream>

#include "poisson_solver.h"

//#define XWALL
//#define COMPDIV
#define LOWMEMORY 1 //compute grad-div matrices directly instead of saving them
#define PRESPARTIAL
#define DIVUPARTIAL

//#define WEAK_PROJECTION

#define CONSCONVPBC
//#define SKEWSYMMVISC

//#define VORTEX
//#define STOKES
//#define POISEUILLE
//#define KOVASZNAY
//#define BELTRAMI
//#define FLOW_PAST_CYLINDER
#define CHANNEL

namespace DG_NavierStokes
{
  using namespace dealii;

#ifdef VORTEX
  const unsigned int fe_degree = 2; //2
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 1;
  const unsigned int dimension = 2; // dimension >= 2
  const unsigned int refine_steps_min = 0; //1
  const unsigned int refine_steps_max = 0;

  const double START_TIME = 0.0;
  const double END_TIME = 2.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool DIVU_TIMESERIES = true; //true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.01; //0.001

  const double VISCOSITY = 0.005;

  const double MAX_VELOCITY = 1.4;
  const double stab_factor_pressure = 1.0;
  const double stab_factor_viscous = stab_factor_pressure;
  const double K=1.5e-3; //1.0e2; //stabilization parameter null-space projection
  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool variabletauw = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool pure_dirichlet_bc = false;

  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const std::string output_prefix = "v001_4_p2_k100_sf1_cfl005";

  const unsigned int output_solver_info_every_timesteps = 1e4;
  const unsigned int output_solver_info_details = 1e4;

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef POISEUILLE
  const unsigned int fe_degree = 2;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 1;
  const unsigned int dimension = 2; // dimension >= 2
  const unsigned int refine_steps_min = 3;
  const unsigned int refine_steps_max = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool DIVU_TIMESERIES = false; //true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.005;

  const double VISCOSITY = 0.1;

  const double MAX_VELOCITY = 1.0;
  const double stab_factor_pressure = 1.0;
  const double stab_factor_viscous = stab_factor_pressure;
  const double K=1.0e2; //stabilization parameter null-space projection
  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool variabletauw = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool pure_dirichlet_bc = false;

  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const std::string output_prefix = "vortex";

  const unsigned int output_solver_info_every_timesteps = 1e4;
  const unsigned int output_solver_info_details = 1e4;

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef KOVASZNAY
  const unsigned int fe_degree = 2;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 1;
  const unsigned int dimension = 2; // dimension >= 2
  const unsigned int refine_steps_min = 3;
  const unsigned int refine_steps_max = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool DIVU_TIMESERIES = false; //true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.01;

  const double VISCOSITY = 0.025;

  const double MAX_VELOCITY = 3.6;
  const double stab_factor_pressure = 1.0;
  const double stab_factor_viscous = stab_factor_pressure;
  const double K=1.0e2; //stabilization parameter null-space projection
  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool variabletauw = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool pure_dirichlet_bc = false;

  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const std::string output_prefix = "kovasznay";

   const unsigned int output_solver_info_every_timesteps = 1e4;
   const unsigned int output_solver_info_details = 1e4;

   const unsigned int ORDER_TIME_INTEGRATOR = 3;
   const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef BELTRAMI
  const unsigned int fe_degree = 3;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 1;
  const unsigned int dimension = 3; // dimension >= 2
  const unsigned int refine_steps_min = 2;
  const unsigned int refine_steps_max = 2;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool DIVU_TIMESERIES = false; //true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.01;

  const double VISCOSITY = 0.1;

  const double MAX_VELOCITY = 3.5;
  const double stab_factor_pressure = 1.0;
  const double stab_factor_viscous = stab_factor_pressure;
  const double K=0.0e2; //stabilization parameter null-space projection
  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool variabletauw = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool pure_dirichlet_bc = true;

  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const std::string output_prefix = "beltrami";

   const unsigned int output_solver_info_every_timesteps = 1e4;
   const unsigned int output_solver_info_details = 1e4;

   const unsigned int ORDER_TIME_INTEGRATOR = 3;
   const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef STOKES
  const unsigned int fe_degree = 3;//3
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 1;
  const unsigned int dimension = 2; // dimension >= 2
  const unsigned int refine_steps_min = 3;//2
  const unsigned int refine_steps_max = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool DIVU_TIMESERIES = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.2; // CFL number irrelevant for Stokes flow problem
  const double TIME_STEP_SIZE = 5.0e-3; //5.0e-4

  const double VISCOSITY = 1.0;

  const double MAX_VELOCITY = 2.65; // MAX_VELOCITY also irrelevant
  const double stab_factor_pressure = 1.0;
  const double stab_factor_viscous = stab_factor_pressure;
  const double K=1.0e2; //stabilization parameter null-space projection
  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool variabletauw = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool pure_dirichlet_bc = true;

  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const std::string output_prefix = "stokes";

  const unsigned int output_solver_info_every_timesteps = 1e4;
  const unsigned int output_solver_info_details = 1e4;

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef FLOW_PAST_CYLINDER
  const unsigned int fe_degree = 1;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 1;
  const unsigned int dimension = 2; // dimension >= 2
  const unsigned int refine_steps_min = 1;
  const unsigned int refine_steps_max = 1;

  const double START_TIME = 0.0;
  const double END_TIME = 10.0;
  const double OUTPUT_INTERVAL_TIME = 0.5;
  const double OUTPUT_START_TIME = 0.0;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool DIVU_TIMESERIES = true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.1;

  const double VISCOSITY = 0.001;
  const double Um = 0.3;//(dimension == 2 ? 1.5 : 2.25); //2D-1: 0.3; 3D-1: 0.45;
  const double D = 0.1;

  const double MAX_VELOCITY = Um;
  const double stab_factor_pressure = 1.0;
  const double stab_factor_viscous = stab_factor_pressure;
  const double K=1.0e1; //stabilization parameter null-space projection
  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool variabletauw = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool pure_dirichlet_bc = false;

  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-16;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const std::string output_prefix = "fpc_r0_p2_k1";

  const unsigned int output_solver_info_every_timesteps = 1e4;
  const unsigned int output_solver_info_details = 1e4;

  const unsigned int ORDER_TIME_INTEGRATOR = 2;
  const bool START_WITH_LOW_ORDER = true;
#endif

#ifdef CHANNEL
  const unsigned int fe_degree = 4;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int fe_degree_xwall = 1;
  const unsigned int n_q_points_1d_xwall = 1;
  const unsigned int dimension = 3; // dimension >= 2
  const unsigned int refine_steps_min = 2;
  const unsigned int refine_steps_max = 2;

  const double START_TIME = 0.0;
  const double END_TIME = 70.0; // Poisseuille 5.0;  Kovasznay 1.0
  const double OUTPUT_INTERVAL_TIME = 1.0;
  const double OUTPUT_START_TIME = 50.0;
  const double STATISTICS_START_TIME = 50.0;
  const bool DIVU_TIMESERIES = false; //true;
  const int STATISTICS_EVERY = 1;
  const int MAX_NUM_STEPS = 1e7;
  const double CFL = 1.0;

  const double VISCOSITY = 1./180.0;//0.005; // Taylor vortex: 0.01; vortex problem (Hesthaven): 0.025; Poisseuille 0.005; Kovasznay 0.025; Stokes 1.0

  const double MAX_VELOCITY = 15.0; // Taylor vortex: 1; vortex problem (Hesthaven): 1.5; Poisseuille 1.0; Kovasznay 4.0
  const double stab_factor_pressure = 1.0;
  const double stab_factor_viscous = 1.0 * stab_factor_pressure;
  const double K=1.0e-3; //stabilization parameter null-space projection
  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool variabletauw = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool pure_dirichlet_bc = true;

  const double REL_TOL_PRESSURE = 1.0e-3;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-3;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const std::string output_prefix = "ch180_4_p4_weak_projection_method";

  const unsigned int output_solver_info_every_timesteps = 10;
  const unsigned int output_solver_info_details = 10;

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = true;
#endif


  template<int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
  AnalyticalSolution (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

  virtual ~AnalyticalSolution(){};

  virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();
    double result = 0.0;
    (void)t;
    /*********************** cavitiy flow *******************************/
  /*  const double T = 0.1;
    if(component == 0 && (std::abs(p[1]-1.0)<1.0e-15))
      result = t<T? (t/T) : 1.0; */
    /********************************************************************/

    /*********************** Cuette flow problem ************************/
    // stationary
  /*  if(component == 0)
          result = ((p[1]+1.0)*0.5); */

    // instationary
   /* const double T = 1.0;
    if(component == 0)
      result = ((p[1]+1.0)*0.5)*(t<T? (t/T) : 1.0); */
    /********************************************************************/

    /****************** Poisseuille flow problem ************************/
#ifdef POISEUILLE
    // constant velocity profile at inflow
   /* const double pressure_gradient = -0.01;
    double T = 0.5;
    if(component == 0 && (std::abs(p[0]+1.0)<1.0e-12))
    result = (t<T? (t/T) : 1.0); */

    // parabolic velocity profile at inflow - stationary
    const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
    if(component == 0)
    result = 1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0;
    if(component == dim)
    result = (p[0]-4.0)*pressure_gradient;

    // parabolic velocity profile at inflow - instationary
//    const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
//    double T = 0.5;
#endif
    /********************************************************************/

    /****************** turbulent channel flow ************************/
#ifdef CHANNEL
    if(component == 0)
    {
      if(p[1]<0.9999&&p[1]>-0.9999)
        result = -22.0*(pow(p[1],2.0)-1.0)*(1.0+((double)rand()/RAND_MAX-1.0)*1.0);//*1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0*(t<T? (t/T) : 1.0);
      else
        result = 0.0;
    }
    if(component == 1|| component == 2)
    {
      result = 0.;
    }
      if(component == dim)
    result = 0.0;//(p[0]-1.0)*pressure_gradient*(t<T? (t/T) : 1.0);
    if(component >dim)
      result = 0.0;
#endif

    /********************************************************************/

    /************************* vortex problem ***************************/
    //Taylor vortex problem (Shahbazi et al.,2007)
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = (-std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 1)
//      result = (+std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 2)
//      result = -0.25*(std::cos(2*pi*p[0])+std::cos(2*pi*p[1]))*std::exp(-4.0*pi*pi*t*VISCOSITY);

    // vortex problem (Hesthaven)
#ifdef VORTEX
    const double pi = numbers::PI;
    if(component == 0)
      result = -std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if(component == 1)
      result = std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if(component == dim)
      result = -std::cos(2*pi*p[0])*std::cos(2*pi*p[1])*std::exp(-8.0*pi*pi*VISCOSITY*t);
#endif
    /********************************************************************/

    /************************* Kovasznay flow ***************************/
#ifdef KOVASZNAY
    const double pi = numbers::PI;
    const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);
    if (component == 0)
      result = 1.0 - std::exp(lambda*p[0])*std::cos(2*pi*p[1]);
    else if (component == 1)
      result = lambda/2.0/pi*std::exp(lambda*p[0])*std::sin(2*pi*p[1]);
    else if (component == dim)
      result = 0.5*(1.0-std::exp(2.0*lambda*p[0]));
#endif
    /********************************************************************/

    /************************* Beltrami flow ****************************/
#ifdef BELTRAMI
    const double pi = numbers::PI;
    const double a = 0.25*pi;
    const double d = 2*a;
    if (component == 0)
      result = -a*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == 1)
      result = -a*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == 2)
      result = -a*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == dim)
        result = -a*a*0.5*(std::exp(2*a*p[0]) + std::exp(2*a*p[1]) + std::exp(2*a*p[2]) +
                           2*std::sin(a*p[0]+d*p[1])*std::cos(a*p[2]+d*p[0])*std::exp(a*(p[1]+p[2])) +
                           2*std::sin(a*p[1]+d*p[2])*std::cos(a*p[0]+d*p[1])*std::exp(a*(p[2]+p[0])) +
                           2*std::sin(a*p[2]+d*p[0])*std::cos(a*p[1]+d*p[2])*std::exp(a*(p[0]+p[1]))) * std::exp(-2*VISCOSITY*d*d*t);
#endif
    /********************************************************************/

    /************* Stokes problem (Guermond,2003 & 2006) ****************/
#ifdef STOKES
    const double pi = numbers::PI;
    double sint = std::sin(t);
    double sinx = std::sin(pi*p[0]);
    double siny = std::sin(pi*p[1]);
    double cosx = std::cos(pi*p[0]);
    double sin2x = std::sin(2.*pi*p[0]);
    double sin2y = std::sin(2.*pi*p[1]);
    if (component == 0)
      result = pi*sint*sin2y*pow(sinx,2.);
    else if (component == 1)
      result = -pi*sint*sin2x*pow(siny,2.);
    else if (component == dim)
      result = cosx*siny*sint;
#endif
    /********************************************************************/

    /********************** flow past cylinder **************************/
#ifdef FLOW_PAST_CYLINDER
    if(component == 0 && std::abs(p[0]-(dim==2 ? 0.3 : 0.0))<1.e-12)
    {
      const double pi = numbers::PI;
      const double T = 0.2;
      const double H = 0.41;
      double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
      result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0);//( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0); //( (t/T)<1.0 ? t/T : 1.0); //std::sin(pi*t/END_TIME);
      if (dim == 3)
        result *= p[2] * (H-p[2]);
    }
#endif
    /********************************************************************/

  return result;
  }

  template<int dim>
  class NeumannBoundaryVelocity : public Function<dim>
  {
  public:
    NeumannBoundaryVelocity (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual ~NeumannBoundaryVelocity(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double NeumannBoundaryVelocity<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();
    double result = 0.0;
    (void)t;

    // Kovasznay flow
#ifdef KOVASZNAY
    // Laplace formulation of viscous term -> prescribe velocity gradient (grad U)*n on Gamma_N
//    const double pi = numbers::PI;
//    const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);
//    if (component == 0)
//      result = -lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
//    else if (component == 1)
//      result = std::pow(lambda,2.0)/2/pi*std::exp(lambda)*std::sin(2*pi*p[1]);

    // Divergence formulation of viscous term -> prescribe (grad U + (grad U) ^T)*n on Gamma_N
    const double pi = numbers::PI;
    const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);
    if (component == 0)
      result = -2.0*lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
    else if (component == 1)
      result = (std::pow(lambda,2.0)/2/pi+2.0*pi)*std::exp(lambda)*std::sin(2*pi*p[1]);
#endif

    //Taylor vortex (Shahbazi et al.,2007)
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = (pi*std::sin(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 1)
//      result = (+pi*std::cos(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);

    // vortex problem (Hesthaven)
#ifdef VORTEX
    // Laplace formulation of viscous term -> prescribe velocity gradient (grad U)*n on Gamma_N
//    const double pi = numbers::PI;
//    if(component==0)
//    {
//      if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
//        result = 2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//      else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
//        result = -2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    }
//    else if(component==1)
//    {
//      if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
//        result = -2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//      else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
//        result = 2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    }
    // Divergence formulation of viscous term -> prescribe (grad U + (grad U) ^T)*n on Gamma_N
    const double pi = numbers::PI;
    if(component==0)
    {
      if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
        result = -2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
      else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
        result = 2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
    }
    else if(component==1)
    {
      if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
        result = -2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
      else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
        result = 2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
    }
#endif

#ifdef POISEUILLE
//    if(component==1)
//      result = - MAX_VELOCITY * 2.0 * p[1];
#endif
    return result;
  }

  template<int dim>
  class RHS : public Function<dim>
  {
  public:
    RHS (const unsigned int   component,
      const double     time = 0.) : Function<dim>(1, time),time(time),component(component) {}

    virtual ~RHS(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const double time;
    const unsigned int component;
  };

  template<int dim>
  double RHS<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
#ifdef CHANNEL
    //channel flow with periodic bc
    if(component==0)
      if(time<0.01)
        return 1.0*(1.0+((double)rand()/RAND_MAX)*0.0);
      else
        return 1.0;
    else
      return 0.0;
#endif

  double t = this->get_time();
  double result = 0.0;
  (void)t;

#ifdef STOKES
  // Stokes problem (Guermond,2003 & 2006)
  const double pi = numbers::PI;
  double sint = std::sin(t);
  double cost = std::cos(t);
  double sinx = std::sin(pi*p[0]);
  double siny = std::sin(pi*p[1]);
  double cosx = std::cos(pi*p[0]);
  double cosy = std::cos(pi*p[1]);
  double sin2x = std::sin(2.*pi*p[0]);
  double sin2y = std::sin(2.*pi*p[1]);
  if (component == 0)
    result = pi*cost*sin2y*pow(sinx,2.)
        - 2.*pow(pi,3.)*sint*sin2y*(1.-4.*pow(sinx,2.))
        - pi*sint*sinx*siny;
  else if (component == 1)
    result = -pi*cost*sin2x*pow(siny,2.)
        + 2.*pow(pi,3.)*sint*sin2x*(1.-4.*pow(siny,2.))
        + pi*sint*cosx*cosy;
#endif

  return result;
  }

  template<int dim>
  class PressureBC_dudt : public Function<dim>
  {
  public:
    PressureBC_dudt (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual ~PressureBC_dudt(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double PressureBC_dudt<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
  double t = this->get_time();
  double result = 0.0;

  //Taylor vortex (Shahbazi et al.,2007)
//  const double pi = numbers::PI;
//  if(component == 0)
//    result = (2.0*pi*pi*VISCOSITY*std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//  else if(component == 1)
//    result = (-2.0*pi*pi*VISCOSITY*std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
#ifdef VORTEX
//   vortex problem (Hesthaven)
  const double pi = numbers::PI;
  if(component == 0)
    result = 4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
  else if(component == 1)
    result = -4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
#endif

  // Beltrami flow
#ifdef BELTRAMI
  const double pi = numbers::PI;
  const double a = 0.25*pi;
  const double d = 2*a;
  if (component == 0)
    result = a*VISCOSITY*d*d*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-VISCOSITY*d*d*t);
  else if (component == 1)
    result = a*VISCOSITY*d*d*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-VISCOSITY*d*d*t);
  else if (component == 2)
    result = a*VISCOSITY*d*d*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-VISCOSITY*d*d*t);
#endif

  // Stokes problem (Guermond,2003 & 2006)
#ifdef STOKES
  const double pi = numbers::PI;
  double cost = std::cos(t);
  double sinx = std::sin(pi*p[0]);
  double siny = std::sin(pi*p[1]);
  double sin2x = std::sin(2.*pi*p[0]);
  double sin2y = std::sin(2.*pi*p[1]);
  if (component == 0)
    result = pi*cost*sin2y*pow(sinx,2.);
  else if (component == 1)
    result = -pi*cost*sin2x*pow(siny,2.);
#endif

  // flow past cylinder
#ifdef FLOW_PAST_CYLINDER
//  if(component == 0 && std::abs(p[0]-(dim==2 ? 0.3 : 0.0))<1.e-12)
//  {
//    const double pi = numbers::PI;
//    const double H = 0.41;
//    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
//    result = coefficient * p[1] * (H-p[1]) * std::cos(pi*t/END_TIME)*pi/END_TIME;
//      if (dim == 3)
//        result *= p[2] * (H-p[2]);
//  }
#endif

  return result;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall> struct NavierStokesViscousMatrix;
  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall> struct WeakProjectionMatrix;

  struct SimpleSpaldingsLaw
  {
    static
  double SpaldingsLaw(double dist, double utau)
  {
    //watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some multiplications
    const double yplus=dist*utau/VISCOSITY;
    double psi=0.0;


    if(yplus>11.0)//this is approximately where the intersection of log law and linear region lies
      psi=log(yplus)+5.17*0.41;
    else
      psi=yplus*0.41;

    double inc=10.0;
    double fn=10.0;
    int count=0;
    bool converged = false;
    while(not converged)
    {
      const double psiquad=psi*psi;
      const double exppsi=std::exp(psi);
      const double expmkmb=std::exp(-0.41*5.17);
             fn=-yplus + psi*(1./0.41)+(expmkmb)*(exppsi-(1.0)-psi-psiquad*(0.5) - psiquad*psi/(6.0) - psiquad*psiquad/(24.0));
             double dfn= 1/0.41+expmkmb*(exppsi-(1.0)-psi-psiquad*(0.5) - psiquad*psi/(6.0));

      inc=fn/dfn;

      psi-=inc;

      bool test=false;
      //do loop for all if one of the values is not converged
        if((std::abs(inc)>1.0E-14 && abs(fn)>1.0E-14&&1000>count++))
            test=true;

      converged = not test;
    }

    return psi;

    //Reichardt's law 1951
    // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
  }
  };

  template <int dim, int n_q_points_1d, typename Number>
    class EvaluationXWall
    {

    public:
    EvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                        const parallel::distributed::Vector<double>& wdist,
                        const parallel::distributed::Vector<double>& tauw):
                          mydata(matrix_free),
                          wdist(wdist),
                          tauw(tauw),
                          evaluate_value(true),
                          evaluate_gradient(true),
                          evaluate_hessian(false),
                          k(0.41),
                          km1(1.0/k),
                          B(5.17),
                          expmkmb(exp(-k*B))
      {};

    void reinit(AlignedVector<VectorizedArray<Number> > qp_wdist,
        AlignedVector<VectorizedArray<Number> > qp_tauw,
        AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_gradwdist,
        AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_gradtauw,
        unsigned int n_q_points,
        std::vector<bool> enriched_components)
    {

      qp_enrichment.resize(n_q_points);
      qp_grad_enrichment.resize(n_q_points);
      for(unsigned int q=0;q<n_q_points;++q)
      {
        qp_enrichment[q] =  EnrichmentShapeDer(qp_wdist[q], qp_tauw[q],
            qp_gradwdist[q], qp_gradtauw[q],&(qp_grad_enrichment[q]), enriched_components);

        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        {
          if(not enriched_components.at(v))
          {
            qp_enrichment[q][v] = 0.0;
            for (unsigned int d = 0; d<dim; d++)
              qp_grad_enrichment[q][d][v] = 0.0;
          }

        }
      }

    };

    void evaluate(const bool evaluate_val,
               const bool evaluate_grad,
               const bool evaluate_hess = false)
    {
      evaluate_value = evaluate_val;
      evaluate_gradient = evaluate_grad;
      //second derivative not implemented yet
      evaluate_hessian = evaluate_hess;
      Assert(not evaluate_hessian,ExcInternalError());
    }
    VectorizedArray<Number> enrichment(unsigned int q){return qp_enrichment[q];}
    Tensor<1,dim,VectorizedArray<Number> > enrichment_gradient(unsigned int q){return qp_grad_enrichment[q];}
    protected:
    VectorizedArray<Number> EnrichmentShapeDer(VectorizedArray<Number> wdist, VectorizedArray<Number> tauw,
        Tensor<1,dim,VectorizedArray<Number> > gradwdist, Tensor<1,dim,VectorizedArray<Number> > gradtauw,
        Tensor<1,dim,VectorizedArray<Number> >* gradpsi, std::vector<bool> enriched_components)
      {
           VectorizedArray<Number> density = make_vectorized_array(1.0);
//        //calculate transformation ---------------------------------------

         Tensor<1,dim,VectorizedArray<Number> > gradtrans;

         const VectorizedArray<Number> utau=std::sqrt(tauw*make_vectorized_array(1.0)/density);
         const VectorizedArray<Number> fac=make_vectorized_array(0.5)/std::sqrt(density*tauw);
         const VectorizedArray<Number> wdistfac=wdist*fac;
//
         for(unsigned int sdm=0;sdm < dim;++sdm)
           gradtrans[sdm]=(utau*gradwdist[sdm]+wdistfac*gradtauw[sdm])*make_vectorized_array(1.0/VISCOSITY);

         //get enrichment function and scalar derivatives
           VectorizedArray<Number> psigp = SpaldingsLaw(wdist, utau, enriched_components)*make_vectorized_array(1.0);
           VectorizedArray<Number> derpsigpsc=DerSpaldingsLaw(psigp)*make_vectorized_array(1.0);
//         //calculate final derivatives
         Tensor<1,dim,VectorizedArray<Number> > gradpsiq;
         for(int sdm=0;sdm < dim;++sdm)
         {
           gradpsiq[sdm]=derpsigpsc*gradtrans[sdm];
         }

         (*gradpsi)=gradpsiq;

        return psigp;
      }

      const MatrixFree<dim,Number> &mydata;

    const parallel::distributed::Vector<double>& wdist;
    const parallel::distributed::Vector<double>& tauw;

    private:

    bool evaluate_value;
    bool evaluate_gradient;
    bool evaluate_hessian;

    const Number k;
    const Number km1;
    const Number B;
    const Number expmkmb;

    AlignedVector<VectorizedArray<Number> > qp_enrichment;
    AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_grad_enrichment;


      VectorizedArray<Number> SpaldingsLaw(VectorizedArray<Number> dist, VectorizedArray<Number> utau, std::vector<bool> enriched_components)
      {
        //watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some multiplications
        const VectorizedArray<Number> yplus=dist*utau*make_vectorized_array(1.0/VISCOSITY);
        VectorizedArray<Number> psi=make_vectorized_array(0.0);

        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        {
          if(enriched_components.at(v))
          {
            if(yplus[v]>11.0)//this is approximately where the intersection of log law and linear region lies
              psi[v]=log(yplus[v])+B*k;
            else
              psi[v]=yplus[v]*k;
          }
          else
            psi[v] = 0.0;
        }

        VectorizedArray<Number> inc=make_vectorized_array(10.0);
        VectorizedArray<Number> fn=make_vectorized_array(10.0);
        int count=0;
        bool converged = false;
        while(not converged)
        {
          VectorizedArray<Number> psiquad=psi*psi;
          VectorizedArray<Number> exppsi=std::exp(psi);
                 fn=-yplus + psi*make_vectorized_array(km1)+make_vectorized_array(expmkmb)*(exppsi-make_vectorized_array(1.0)-psi-psiquad*make_vectorized_array(0.5) - psiquad*psi/make_vectorized_array(6.0) - psiquad*psiquad/make_vectorized_array(24.0));
                 VectorizedArray<Number> dfn= km1+expmkmb*(exppsi-make_vectorized_array(1.0)-psi-psiquad*make_vectorized_array(0.5) - psiquad*psi/make_vectorized_array(6.0));

          inc=fn/dfn;

          psi-=inc;

          bool test=false;
          //do loop for all if one of the values is not converged
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(enriched_components.at(v))
              if((std::abs(inc[v])>1.0E-14 && abs(fn[v])>1.0E-14&&1000>count++))
                test=true;
          }
          converged = not test;
        }

        return psi;

        //Reichardt's law 1951
        // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
      }

      VectorizedArray<Number> DerSpaldingsLaw(VectorizedArray<Number> psi)
      {
        //derivative with respect to y+!
        //spaldings law according to paper (derivative)
        return make_vectorized_array(1.0)/(make_vectorized_array(1.0/k)+make_vectorized_array(expmkmb)*(std::exp(psi)-make_vectorized_array(1.0)-psi-psi*psi*make_vectorized_array(0.5)-psi*psi*psi/make_vectorized_array(6.0)));

      // Reichardt's law
      //  double yplus=dist*utau*viscinv_;
      //  return (0.4/(k_*(1.0+0.4*yplus))+7.8*(1.0/11.0*exp(-yplus/11.0)-1.0/11.0*exp(-yplus/3.0)+yplus/33.0*exp(-yplus/3.0)))*k_;
      }

      Number Der2SpaldingsLaw(Number psi,Number derpsi)
      {
        //derivative with respect to y+!
        //spaldings law according to paper (2nd derivative)
        return -make_vectorized_array(expmkmb)*(exp(psi)-make_vectorized_array(1.)-psi-psi*psi*make_vectorized_array(0.5))*derpsi*derpsi*derpsi;

        // Reichardt's law
      //  double yplus=dist*utau*viscinv_;
      //  return (-0.4*0.4/(k_*(1.0+0.4*yplus)*(1.0+0.4*yplus))+7.8*(-1.0/121.0*exp(-yplus/11.0)+(2.0/33.0-yplus/99.0)*exp(-yplus/3.0)))*k_;
      }
    };

  template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
              int n_components_ = 1, typename Number = double >
    class FEEvaluationXWall : public EvaluationXWall<dim,n_q_points_1d, Number>
    {
      typedef FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
      typedef Number                            number_type;
      typedef typename BaseClass::value_type    value_type;
      typedef typename BaseClass::gradient_type gradient_type;

public:
    FEEvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                        const parallel::distributed::Vector<double>& wdist,
                        const parallel::distributed::Vector<double>& tauw,
                        const unsigned int            fe_no = 0,
                        const unsigned int            quad_no = 0):
                          EvaluationXWall<dim,n_q_points_1d, Number>::EvaluationXWall(matrix_free, wdist, tauw),
                          fe_eval(),
                          fe_eval_xwall(),
                          fe_eval_tauw(),
                          values(),
                          gradients(),
                          std_dofs_per_cell(0),
                          dofs_per_cell(0),
                          tensor_dofs_per_cell(0),
                          n_q_points(0),
                          enriched(false)
      {
        {
          FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval_tmp(matrix_free,fe_no,quad_no);
          fe_eval.resize(1,fe_eval_tmp);
        }
#ifdef XWALL
        {
          FEEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall_tmp(matrix_free,3,quad_no);
          fe_eval_xwall.resize(1,fe_eval_xwall_tmp);
        }
#endif
        {
          FEEvaluation<dim,1,n_q_points_1d,1,double> fe_eval_tauw_tmp(matrix_free,2,quad_no);
          fe_eval_tauw.resize(1,fe_eval_tauw_tmp);
        }
        values.resize(fe_eval[0].n_q_points,value_type());
        gradients.resize(fe_eval[0].n_q_points,gradient_type());
        n_q_points = fe_eval[0].n_q_points;
      };

      void reinit(const unsigned int cell)
      {
#ifdef XWALL
        {
          enriched = false;
          values.resize(fe_eval[0].n_q_points,value_type());
          gradients.resize(fe_eval[0].n_q_points,gradient_type());
  //        decide if we have an enriched element via the y component of the cell center
          for (unsigned int v=0; v<EvaluationXWall<dim,n_q_points_1d, Number>::mydata.n_components_filled(cell); ++v)
          {
            typename DoFHandler<dim>::cell_iterator dcell = EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(cell, v);
//            std::cout << ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL))) << std::endl;
            if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
              enriched = true;
          }
          enriched_components.resize(VectorizedArray<Number>::n_array_elements);
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            enriched_components.at(v) = false;
          if(enriched)
          {
            //store, exactly which component of the vectorized array is enriched
            for (unsigned int v=0; v<EvaluationXWall<dim,n_q_points_1d, Number>::mydata.n_components_filled(cell); ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell = EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(cell, v);
              if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                  enriched_components.at(v) = true;
            }

            //initialize the enrichment function
            {
              fe_eval_tauw[0].reinit(cell);
              //get wall distance and wss at quadrature points
              fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
              fe_eval_tauw[0].evaluate(true, true);

              AlignedVector<VectorizedArray<Number> > cell_wdist;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradwdist;
              cell_wdist.resize(fe_eval_tauw[0].n_q_points);
              cell_gradwdist.resize(fe_eval_tauw[0].n_q_points);
              for(unsigned int q=0;q<fe_eval_tauw[0].n_q_points;++q)
              {
                cell_wdist[q] = fe_eval_tauw[0].get_value(q);
                cell_gradwdist[q] = fe_eval_tauw[0].get_gradient(q);
              }

              fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);

              fe_eval_tauw[0].evaluate(true, true);

              AlignedVector<VectorizedArray<Number> > cell_tauw;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradtauw;

              cell_tauw.resize(fe_eval_tauw[0].n_q_points);
              cell_gradtauw.resize(fe_eval_tauw[0].n_q_points);

              for(unsigned int q=0;q<fe_eval_tauw[0].n_q_points;++q)
              {
                cell_tauw[q] = fe_eval_tauw[0].get_value(q);
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                    Assert( fe_eval_tauw[0].get_value(q)[v] > 1.0e-9 ,ExcInternalError());
                }

                cell_gradtauw[q] = fe_eval_tauw[0].get_gradient(q);
              }
              EvaluationXWall<dim,n_q_points_1d, Number>::reinit(cell_wdist, cell_tauw, cell_gradwdist, cell_gradtauw, fe_eval_tauw[0].n_q_points,enriched_components);
            }
          }
          fe_eval_xwall[0].reinit(cell);
        }
#endif
        fe_eval[0].reinit(cell);
        std_dofs_per_cell = fe_eval[0].dofs_per_cell;
#ifdef XWALL
        if(enriched)
        {
          dofs_per_cell = fe_eval[0].dofs_per_cell + fe_eval_xwall[0].dofs_per_cell;
          tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell + fe_eval_xwall[0].tensor_dofs_per_cell;
        }
        else
        {
          dofs_per_cell = fe_eval[0].dofs_per_cell;
          tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell;
        }
#else
        dofs_per_cell = fe_eval[0].dofs_per_cell;
        tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell;
#endif
      }

      VectorizedArray<double> * begin_dof_values()
    {
        return fe_eval[0].begin_dof_values();
    }

      void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &src_xwall)
      {

        fe_eval[0].read_dof_values(src);
#ifdef XWALL
        fe_eval_xwall[0].read_dof_values(src_xwall);
#endif
      }

      void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i,const std::vector<parallel::distributed::Vector<double> > &src_xwall, unsigned int j)
      {
        fe_eval[0].read_dof_values(src,i);
#ifdef XWALL
        fe_eval_xwall[0].read_dof_values(src_xwall,j);
#endif
      }
      void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i,const parallel::distributed::BlockVector<double> &src_xwall, unsigned int j)
      {
        fe_eval[0].read_dof_values(src,i);
#ifdef XWALL
        fe_eval_xwall[0].read_dof_values(src_xwall,j);
#endif
      }

      void evaluate(const bool evaluate_val,
                 const bool evaluate_grad,
                 const bool evaluate_hess = false)
      {
        fe_eval[0].evaluate(evaluate_val,evaluate_grad,evaluate_hess);
#ifdef XWALL
          if(enriched)
          {
            gradients.resize(fe_eval[0].n_q_points,gradient_type());
            values.resize(fe_eval[0].n_q_points,value_type());
            fe_eval_xwall[0].evaluate(true,evaluate_grad);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            EvaluationXWall<dim,n_q_points_1d,Number>::evaluate(evaluate_val,evaluate_grad,evaluate_hess);
            //evaluate gradient
            if(evaluate_grad)
            {
              gradient_type submitgradient = gradient_type();
              gradient_type gradient = gradient_type();
              //there are 2 parts due to chain rule
              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              {
                submitgradient = gradient_type();
                gradient = fe_eval_xwall[0].get_gradient(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
                val_enrgrad_to_grad(gradient, q);
                //delete enrichment part where not needed
                //this is essential, code won't work otherwise
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                  if(enriched_components.at(v))
                    add_array_component_to_gradient(submitgradient,gradient,v);
                gradients[q] = submitgradient;
              }
            }
            if(evaluate_val)
            {
              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              {
                value_type finalvalue = fe_eval_xwall[0].get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
                value_type submitvalue = value_type();
                //delete enrichment part where not needed
                //this is essential, code won't work otherwise
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                  if(enriched_components.at(v))
                    add_array_component_to_value(submitvalue,finalvalue,v);
                values[q]=submitvalue;
              }
            }
          }
#endif
      }

      void val_enrgrad_to_grad(Tensor<2,dim,VectorizedArray<Number> >& grad, unsigned int q)
      {
        for(unsigned int j=0;j<dim;++j)
        {
          for(unsigned int i=0;i<dim;++i)
          {
            grad[j][i] += fe_eval_xwall[0].get_value(q)[j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }
      void val_enrgrad_to_grad(Tensor<1,dim,VectorizedArray<Number> >& grad, unsigned int q)
      {
        for(unsigned int i=0;i<dim;++i)
        {
          grad[i] += fe_eval_xwall[0].get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
        }
      }


      void submit_value(const value_type val_in,
          const unsigned int q_point)
      {
        fe_eval[0].submit_value(val_in,q_point);
#ifdef XWALL
        values[q_point] = value_type();
          if(enriched)
            values[q_point] = val_in;
#endif
      }
      void submit_value(const Tensor<1,1,VectorizedArray<Number> > val_in,
          const unsigned int q_point)
      {
        fe_eval[0].submit_value(val_in[0],q_point);
#ifdef XWALL
        values[q_point] = value_type();
          if(enriched)
            values[q_point] = val_in[0];
#endif
      }

      void submit_gradient(const gradient_type grad_in,
          const unsigned int q_point)
      {
        fe_eval[0].submit_gradient(grad_in,q_point);
#ifdef XWALL
        gradients[q_point] = gradient_type();
        if(enriched)
          gradients[q_point] = grad_in;
#endif
      }

      void value_type_unit(VectorizedArray<Number>* test)
        {
          *test = make_vectorized_array(1.);
        }

      void value_type_unit(Tensor<1,n_components_,VectorizedArray<Number> >* test)
        {
          for(unsigned int i = 0; i< n_components_; i++)
            (*test)[i] = make_vectorized_array(1.);
        }

      void print_value_type_unit(VectorizedArray<Number> test)
        {
          std::cout << test[0] << std::endl;
        }

      void print_value_type_unit(Tensor<1,n_components_,VectorizedArray<Number> > test)
        {
          for(unsigned int i = 0; i< n_components_; i++)
            std::cout << test[i][0] << "  ";
          std::cout << std::endl;
        }

      value_type get_value(const unsigned int q_point)
      {
#ifdef XWALL
        if(enriched)
          return values[q_point] + fe_eval[0].get_value(q_point);
#endif
          return fe_eval[0].get_value(q_point);
      }
      void add_array_component_to_value(VectorizedArray<Number>& val,const VectorizedArray<Number>& toadd, unsigned int v)
      {
        val[v] += toadd[v];
      }
      void add_array_component_to_value(Tensor<1,n_components_,VectorizedArray<Number> >& val,const Tensor<1,n_components_,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          val[d][v] += toadd[d][v];
      }


      gradient_type get_gradient (const unsigned int q_point)
      {
#ifdef XWALL
          if(enriched)
            return fe_eval[0].get_gradient(q_point) + gradients[q_point];
#endif
        return fe_eval[0].get_gradient(q_point);
      }

      gradient_type get_symmetric_gradient (const unsigned int q_point)
      {
        return make_symmetric(get_gradient(q_point));
      }

      void add_array_component_to_gradient(Tensor<2,dim,VectorizedArray<Number> >& grad,const Tensor<2,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int comp = 0; comp<dim; comp++)
          for (unsigned int d = 0; d<dim; d++)
            grad[comp][d][v] += toadd[comp][d][v];
      }
      void add_array_component_to_gradient(Tensor<1,dim,VectorizedArray<Number> >& grad,const Tensor<1,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          grad[d][v] += toadd[d][v];
      }

      Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
    {
        Tensor<2,dim,VectorizedArray<Number> > symgrad;
        for (unsigned int i = 0; i<dim; i++)
          for (unsigned int j = 0; j<dim; j++)
            symgrad[i][j] =  grad[i][j] + grad[j][i];
        return symgrad;
    }

    Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
      {
          Tensor<1,dim,VectorizedArray<Number> > symgrad;
          Assert(false, ExcInternalError());
          return symgrad;
      }

      void integrate (const bool integrate_val,
                      const bool integrate_grad)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            AlignedVector<value_type> tmp_values(fe_eval[0].n_q_points,value_type());
            if(integrate_val)
              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
                tmp_values[q]=values[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            //the scalar product of the second part of the gradient is computed directly and added to the value
            if(integrate_grad)
            {
              //first, zero out all non-enriched vectorized array components
              grad_enr_to_val(tmp_values, gradients);

              for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
                fe_eval_xwall[0].submit_gradient(gradients[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q),q);
            }

            for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              fe_eval_xwall[0].submit_value(tmp_values[q],q);
            //integrate
            fe_eval_xwall[0].integrate(true,integrate_grad);
          }
        }
#endif
        fe_eval[0].integrate(integrate_val, integrate_grad);
      }

      void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
        {
          for(int j=0; j<dim;++j)//comp
          {
            for(int i=0; i<dim;++i)//dim
            {
              tmp_values[q][j] += gradient[q][j][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
            }
          }
        }
      }
      void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
        {
          for(int i=0; i<dim;++i)//dim
          {
            tmp_values[q] += gradient[q][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }

      void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
      {
        fe_eval[0].distribute_local_to_global(dst);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall[0].distribute_local_to_global(dst_xwall);
#endif
      }

      void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
      {
        fe_eval[0].distribute_local_to_global(dst,i);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall[0].distribute_local_to_global(dst_xwall,j);
#endif
      }

      void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i,parallel::distributed::BlockVector<double> &dst_xwall, unsigned int j)
      {
        fe_eval[0].distribute_local_to_global(dst,i);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall[0].distribute_local_to_global(dst_xwall,j);
#endif
      }

      void set_dof_values (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
      {
        fe_eval[0].set_dof_values(dst);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall[0].set_dof_values(dst_xwall);
#endif
      }

      void set_dof_values (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
      {
        fe_eval[0].set_dof_values(dst,i);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall[0].set_dof_values(dst_xwall,j);
#endif
      }

      void fill_JxW_values(AlignedVector<VectorizedArray<Number> > &JxW_values) const
      {
        fe_eval[0].fill_JxW_values(JxW_values);
      }

      Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
      {
        return fe_eval[0].quadrature_point(q);
      }

      VectorizedArray<Number> get_divergence(unsigned int q)
    {
#ifdef XWALL
        if(enriched)
        {
          VectorizedArray<Number> div_enr= make_vectorized_array(0.0);
          for (unsigned int i=0;i<dim;i++)
            div_enr += gradients[q][i][i];
          return fe_eval[0].get_divergence(q) + div_enr;
        }
#endif
        return fe_eval[0].get_divergence(q);
    }

    Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
    get_curl (const unsigned int q_point) const
     {
#ifdef XWALL
      if(enriched)
      {
        // copy from generic function into dim-specialization function
        const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
        Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
        switch (dim)
          {
          case 1:
            Assert (false,
                    ExcMessage("Computing the curl in 1d is not a useful operation"));
            break;
          case 2:
            curl[0] = grad[1][0] - grad[0][1];
            break;
          case 3:
            curl[0] = grad[2][1] - grad[1][2];
            curl[1] = grad[0][2] - grad[2][0];
            curl[2] = grad[1][0] - grad[0][1];
            break;
          default:
            Assert (false, ExcNotImplemented());
            break;
          }
        return fe_eval[0].get_curl(q_point) + curl;
      }
#endif
      return fe_eval[0].get_curl(q_point);
     }
    VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
    {
#ifdef XWALL
      if(enriched)
      {
        VectorizedArray<Number> returnvalue = make_vectorized_array(0.0);
        if(j<fe_eval[0].dofs_per_cell*n_components_)
          returnvalue =  fe_eval[0].begin_dof_values()[j];
        else
        {
          returnvalue = fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_];
        }
        return returnvalue;
      }
      else
        return fe_eval[0].begin_dof_values()[j];
#else

      return fe_eval[0].begin_dof_values()[j];
#endif
    }
    void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
    {
#ifdef XWALL
      if(enriched)
      {
        if(j<fe_eval[0].dofs_per_cell*n_components_)
          fe_eval[0].begin_dof_values()[j][v] = value;
        else
          fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_][v] = value;
      }
      else
        fe_eval[0].begin_dof_values()[j][v]=value;
      return;
#else
      fe_eval[0].begin_dof_values()[j][v]=value;
      return;
#endif
    }
    void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
    {
#ifdef XWALL
      if(enriched)
      {
        if(j<fe_eval[0].dofs_per_cell*n_components_)
          fe_eval[0].begin_dof_values()[j] = value;
        else
          fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_] = value;
      }
      else
        fe_eval[0].begin_dof_values()[j]=value;
      return;
#else
      fe_eval[0].begin_dof_values()[j]=value;
      return;
#endif
    }
    bool component_enriched(unsigned int v)
    {
      if(not enriched)
        return false;
      else
        return enriched_components.at(v);
    }

    void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int cell)
    {
      eddyvisc.resize(n_q_points);
      if(CS > 1e-10)
      {
        const VectorizedArray<Number> Cs = make_vectorized_array(CS);
        VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
        fe_eval_tauw[0].reinit(cell);
        {
          VectorizedArray<Number> volume = make_vectorized_array(0.);
          {
            AlignedVector<VectorizedArray<Number> > JxW_values;
            JxW_values.resize(fe_eval_tauw[0].n_q_points);
            fe_eval_tauw[0].fill_JxW_values(JxW_values);
            for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
              volume += JxW_values[q];
          }
          reinit(cell);
          read_dof_values(solution_n,0,solution_n,dim+1);
          evaluate (false,true,false);
          AlignedVector<VectorizedArray<Number> > wdist;
          wdist.resize(fe_eval_tauw[0].n_q_points);
          fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
          fe_eval_tauw[0].evaluate(true,false,false);
          for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
            wdist[q] = fe_eval_tauw[0].get_value(q);
          fe_eval_tauw[0].reinit(cell);
          fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
          fe_eval_tauw[0].evaluate(true,false,false);

          const VectorizedArray<Number> hvol = std::pow(volume, 1./(double)dim) * hfac;

          for (unsigned int q=0; q<n_q_points; ++q)
          {
            Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

            VectorizedArray<Number> snorm = make_vectorized_array(0.);
            for (unsigned int i = 0; i<dim ; i++)
              for (unsigned int j = 0; j<dim ; j++)
                snorm += (s[i][j])*(s[i][j]);
            snorm *= make_vectorized_array<Number>(0.5);
            //simple wall correction
            VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw[0].get_value(q))*0.04));
            VectorizedArray<Number> lm = Cs*hvol*fmu;
            eddyvisc[q]= make_vectorized_array(VISCOSITY) + lm*lm*std::sqrt(snorm);
          }
        }
        //initialize again to get a clean version
        reinit(cell);
      }
#ifdef XWALL
      else if (ML>0.1 && enriched)
      {
        fe_eval_tauw[0].reinit(cell);
        {
          read_dof_values(solution_n,0,solution_n,dim+1);
          evaluate (false,true,false);
          AlignedVector<VectorizedArray<Number> > wdist;
          wdist.resize(fe_eval_tauw[0].n_q_points);
          fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
          fe_eval_tauw[0].evaluate(true,false,false);
          for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
            wdist[q] = fe_eval_tauw[0].get_value(q);
          fe_eval_tauw[0].reinit(cell);
          fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
          fe_eval_tauw[0].evaluate(true,false,false);

          for (unsigned int q=0; q<n_q_points; ++q)
          {
            Tensor<2,dim,VectorizedArray<Number> > s = get_gradient(q);
            Tensor<2,dim,VectorizedArray<Number> > om;
            for (unsigned int i=0; i<dim;i++)
              for (unsigned int j=0;j<dim;j++)
                om[i][j]=0.5*(s[i][j]-s[j][i]);

            VectorizedArray<Number> osum = make_vectorized_array(0.);
            for (unsigned int i=0; i<dim;i++)
              for (unsigned int j=0;j<dim;j++)
                osum += om[i][j]*om[i][j];
            VectorizedArray<Number> onorm = std::sqrt(2.*osum);

            //simple wall correction
            VectorizedArray<Number> l = 0.41*wdist[q]*(1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw[0].get_value(q))*0.04));
            VectorizedArray<Number> vt = l*l*onorm;
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                eddyvisc[q][v]= VISCOSITY + vt[v];
              }
              else
                eddyvisc[q][v]= VISCOSITY;
            }
          }
        }
        //initialize again to get a clean version
        reinit(cell);
    }
#endif
      else
        for (unsigned int q=0; q<n_q_points; ++q)
          eddyvisc[q]= make_vectorized_array(VISCOSITY);

      return;
    }
    private:
      AlignedVector<FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> > fe_eval;
      AlignedVector<FEEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> > fe_eval_xwall;
      AlignedVector<FEEvaluation<dim,1,n_q_points_1d,1,double> > fe_eval_tauw;
      AlignedVector<value_type> values;
      AlignedVector<gradient_type> gradients;

    public:
      unsigned int std_dofs_per_cell;
      unsigned int dofs_per_cell;
      unsigned int tensor_dofs_per_cell;
      unsigned int n_q_points;
      bool enriched;
      std::vector<bool> enriched_components;
      AlignedVector<VectorizedArray<Number> > eddyvisc;

    };


  template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
              int n_components_ = 1, typename Number = double >
    class FEFaceEvaluationXWall : public EvaluationXWall<dim,n_q_points_1d, Number>
    {
    public:
      typedef FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
      typedef Number                            number_type;
      typedef typename BaseClass::value_type    value_type;
      typedef typename BaseClass::gradient_type gradient_type;

      FEFaceEvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                        const parallel::distributed::Vector<double>& wdist,
                        const parallel::distributed::Vector<double>& tauw,
                        const bool                    is_left_face = true,
                        const unsigned int            fe_no = 0,
                        const unsigned int            quad_no = 0,
                        const bool                    no_gradients_on_faces = false):
                          EvaluationXWall<dim,n_q_points_1d, Number>::EvaluationXWall(matrix_free, wdist, tauw),
                          fe_eval(matrix_free,is_left_face,fe_no,quad_no,no_gradients_on_faces),
                          fe_eval_xwall(matrix_free,is_left_face,3,quad_no,no_gradients_on_faces),
                          fe_eval_tauw(matrix_free,is_left_face,2,quad_no,no_gradients_on_faces),
                          is_left_face(is_left_face),
                          values(fe_eval.n_q_points),
                          gradients(fe_eval.n_q_points),
                          dofs_per_cell(0),
                          tensor_dofs_per_cell(0),
                          n_q_points(fe_eval.n_q_points),
                          enriched(false)
      {
      };

      void reinit(const unsigned int f)
      {
#ifdef XWALL
        {
          enriched = false;
          values.resize(fe_eval.n_q_points,value_type());
          gradients.resize(fe_eval.n_q_points,gradient_type());
          if(is_left_face)
          {
  //        decide if we have an enriched element via the y component of the cell center
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
                  if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                    enriched = true;
            }
          }
          else
          {
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
                  if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                    enriched = true;
            }
          }
          enriched_components.resize(VectorizedArray<Number>::n_array_elements);
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            enriched_components.at(v) = false;
          if(enriched)
          {
            //store, exactly which component of the vectorized array is enriched
            if(is_left_face)
            {
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
              {
                typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
                    if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                      enriched_components.at(v)=(true);
              }
            }
            else
            {
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
              EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
              {
                typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
                    EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
                    if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                      enriched_components.at(v)=(true);
              }
            }

            Assert(enriched_components.size()==VectorizedArray<Number>::n_array_elements,ExcInternalError());

            //initialize the enrichment function
            {
              fe_eval_tauw.reinit(f);
              //get wall distance and wss at quadrature points
              fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
              fe_eval_tauw.evaluate(true, true);

              AlignedVector<VectorizedArray<Number> > face_wdist;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradwdist;
              face_wdist.resize(fe_eval_tauw.n_q_points);
              face_gradwdist.resize(fe_eval_tauw.n_q_points);
              for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
              {
                face_wdist[q] = fe_eval_tauw.get_value(q);
                face_gradwdist[q] = fe_eval_tauw.get_gradient(q);
              }

              fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
              fe_eval_tauw.evaluate(true, true);
              AlignedVector<VectorizedArray<Number> > face_tauw;
              AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradtauw;
              face_tauw.resize(fe_eval_tauw.n_q_points);
              face_gradtauw.resize(fe_eval_tauw.n_q_points);
              for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
              {
                face_tauw[q] = fe_eval_tauw.get_value(q);
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                {
                  if(enriched_components.at(v))
                    Assert( fe_eval_tauw.get_value(q)[v] > 1.0e-9 ,ExcInternalError());
                }

                face_gradtauw[q] = fe_eval_tauw.get_gradient(q);
              }
              EvaluationXWall<dim,n_q_points_1d, Number>::reinit(face_wdist, face_tauw, face_gradwdist, face_gradtauw, fe_eval_tauw.n_q_points,enriched_components);
            }
          }
          fe_eval_xwall.reinit(f);
        }
#endif
        fe_eval.reinit(f);
#ifdef XWALL
        if(enriched)
        {
          dofs_per_cell = fe_eval.dofs_per_cell + fe_eval_xwall.dofs_per_cell;
          tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell + fe_eval_xwall.tensor_dofs_per_cell;
        }
        else
        {
          dofs_per_cell = fe_eval.dofs_per_cell;
          tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell;
        }
#else
        dofs_per_cell = fe_eval.dofs_per_cell;
        tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell;
#endif
      }

      void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &src_xwall)
      {
        fe_eval.read_dof_values(src);
#ifdef XWALL
        fe_eval_xwall.read_dof_values(src_xwall);
#endif
      }

      void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i,const std::vector<parallel::distributed::Vector<double> > &src_xwall, unsigned int j)
      {
        fe_eval.read_dof_values(src,i);
#ifdef XWALL
        fe_eval_xwall.read_dof_values(src_xwall,j);
#endif
      }

      void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i,const parallel::distributed::BlockVector<double> &src_xwall, unsigned int j)
      {
        fe_eval.read_dof_values(src,i);
#ifdef XWALL
        fe_eval_xwall.read_dof_values(src_xwall,j);
#endif
      }

      void evaluate(const bool evaluate_val,
                 const bool evaluate_grad,
                 const bool evaluate_hess = false)
      {
        AssertThrow(evaluate_hess == false, ExcNotImplemented());
        fe_eval.evaluate(evaluate_val,evaluate_grad);
#ifdef XWALL
          if(enriched)
          {
            gradients.resize(fe_eval.n_q_points,gradient_type());
            values.resize(fe_eval.n_q_points,value_type());
            fe_eval_xwall.evaluate(true,evaluate_grad);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            EvaluationXWall<dim,n_q_points_1d,Number>::evaluate(evaluate_val,evaluate_grad,evaluate_hess);
            //evaluate gradient
            if(evaluate_grad)
            {
              //there are 2 parts due to chain rule
              gradient_type gradient = gradient_type();
              gradient_type submitgradient = gradient_type();
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              {
                submitgradient = gradient_type();
                gradient = fe_eval_xwall.get_gradient(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
                val_enrgrad_to_grad(gradient,q);
                //delete enrichment part where not needed
                //this is essential, code won't work otherwise
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                  if(enriched_components.at(v))
                    add_array_component_to_gradient(submitgradient,gradient,v);

                gradients[q] = submitgradient;
              }
            }
            if(evaluate_val)
            {
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              {
                value_type finalvalue = fe_eval_xwall.get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
                value_type submitvalue = value_type();
                //delete enrichment part where not needed
                //this is essential, code won't work otherwise
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                  if(enriched_components.at(v))
                    add_array_component_to_value(submitvalue,finalvalue,v);
                values[q]=submitvalue;
              }
            }
          }
#endif
      }
      void val_enrgrad_to_grad(Tensor<2,dim,VectorizedArray<Number> >& grad, unsigned int q)
      {
        for(unsigned int j=0;j<dim;++j)
        {
          for(unsigned int i=0;i<dim;++i)
          {
            grad[j][i] += fe_eval_xwall.get_value(q)[j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }
      void val_enrgrad_to_grad(Tensor<1,dim,VectorizedArray<Number> >& grad, unsigned int q)
      {
        for(unsigned int i=0;i<dim;++i)
        {
          grad[i] += fe_eval_xwall.get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
        }
      }

      void submit_value(const value_type val_in,
          const unsigned int q_point)
      {
        fe_eval.submit_value(val_in,q_point);
#ifdef XWALL
        values[q_point] = value_type();
        if(enriched)
          values[q_point] = val_in;
#endif
      }

      void submit_gradient(const gradient_type grad_in,
          const unsigned int q_point)
      {
        fe_eval.submit_gradient(grad_in,q_point);
#ifdef XWALL
        gradients[q_point] = gradient_type();
        if(enriched)
          gradients[q_point] = grad_in;
#endif
      }

      value_type get_value(const unsigned int q_point)
      {
#ifdef XWALL
        {
          if(enriched)
            return fe_eval.get_value(q_point) + values[q_point];//fe_eval.get_value(q_point) + values[q_point];
        }
#endif
          return fe_eval.get_value(q_point);
      }
      void add_array_component_to_value(VectorizedArray<Number>& val,const VectorizedArray<Number>& toadd, unsigned int v)
      {
        val[v] += toadd[v];
      }
      void add_array_component_to_value(Tensor<1,n_components_, VectorizedArray<Number> >& val,const Tensor<1,n_components_,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          val[d][v] += toadd[d][v];
      }

      gradient_type get_gradient (const unsigned int q_point)
      {
#ifdef XWALL
        if(enriched)
          return fe_eval.get_gradient(q_point) + gradients[q_point];
#endif
        return fe_eval.get_gradient(q_point);
      }

      gradient_type get_symmetric_gradient (const unsigned int q_point)
      {
        return make_symmetric(get_gradient(q_point));
      }

      Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
    {
        Tensor<2,dim,VectorizedArray<Number> > symgrad;
        for (unsigned int i = 0; i<dim; i++)
          for (unsigned int j = 0; j<dim; j++)
            symgrad[i][j] = grad[i][j] + grad[j][i];
        return symgrad;
    }

      Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
    {
        Tensor<1,dim,VectorizedArray<Number> > symgrad;
        // symmetric gradient is not defined in that case
        Assert(false, ExcInternalError());
        return symgrad;
    }

      void add_array_component_to_gradient(Tensor<2,dim,VectorizedArray<Number> >& grad,const Tensor<2,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int comp = 0; comp<dim; comp++)
          for (unsigned int d = 0; d<dim; d++)
            grad[comp][d][v] += toadd[comp][d][v];
      }
      void add_array_component_to_gradient(Tensor<1,dim,VectorizedArray<Number> >& grad,const Tensor<1,dim,VectorizedArray<Number> >& toadd, unsigned int v)
      {
        for (unsigned int d = 0; d<n_components_; d++)
          grad[d][v] += toadd[d][v];
      }

      VectorizedArray<Number> get_divergence(unsigned int q)
    {
#ifdef XWALL
        if(enriched)
        {
          VectorizedArray<Number> div_enr= make_vectorized_array(0.0);
          for (unsigned int i=0;i<dim;i++)
            div_enr += gradients[q][i][i];
          return fe_eval.get_divergence(q) + div_enr;
        }
#endif
        return fe_eval.get_divergence(q);
    }

      Tensor<1,dim,VectorizedArray<Number> > get_normal_vector(const unsigned int q_point) const
      {
        return fe_eval.get_normal_vector(q_point);
      }

      void integrate (const bool integrate_val,
                      const bool integrate_grad)
      {
#ifdef XWALL
        {
          if(enriched)
          {
            AlignedVector<value_type> tmp_values(fe_eval.n_q_points,value_type());
            if(integrate_val)
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
                tmp_values[q]=values[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
            //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
            //the scalar product of the second part of the gradient is computed directly and added to the value
            if(integrate_grad)
            {
              grad_enr_to_val(tmp_values,gradients);
              for(unsigned int q=0;q<fe_eval.n_q_points;++q)
                fe_eval_xwall.submit_gradient(gradients[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q),q);
            }

            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              fe_eval_xwall.submit_value(tmp_values[q],q);
            //integrate
            fe_eval_xwall.integrate(true,integrate_grad);
          }
        }
#endif
        fe_eval.integrate(integrate_val, integrate_grad);
      }

      void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {

          for(int j=0; j<dim;++j)//comp
          {
            for(int i=0; i<dim;++i)//dim
            {
              tmp_values[q][j] += gradient[q][j][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
            }
          }
        }
      }
      void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
      {
        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          for(int i=0; i<dim;++i)//dim
          {
            tmp_values[q] += gradient[q][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }

      void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
      {
        fe_eval.distribute_local_to_global(dst);
#ifdef XWALL
          if(enriched)
            fe_eval_xwall.distribute_local_to_global(dst_xwall);
#endif
      }

      void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
      {
        fe_eval.distribute_local_to_global(dst,i);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall.distribute_local_to_global(dst_xwall,j);
#endif
      }


      void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i,parallel::distributed::BlockVector<double> &dst_xwall, unsigned int j)
      {
        fe_eval.distribute_local_to_global(dst,i);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall.distribute_local_to_global(dst_xwall,j);
#endif
      }

      Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
      {
        return fe_eval.quadrature_point(q);
      }

      VectorizedArray<Number> get_normal_volume_fraction()
      {
        return fe_eval.get_normal_volume_fraction();
      }

      VectorizedArray<Number> read_cell_data(const AlignedVector<VectorizedArray<Number> > &cell_data)
      {
        return fe_eval.read_cell_data(cell_data);
      }

      Tensor<1,n_components_,VectorizedArray<Number> > get_normal_gradient(const unsigned int q_point) const
      {
#ifdef XWALL
      {
        if(enriched)
        {
          Tensor<1,n_components_,VectorizedArray<Number> > grad_out;
          for (unsigned int comp=0; comp<n_components_; comp++)
          {
            grad_out[comp] = gradients[q_point][comp][0] *
                             fe_eval.get_normal_vector(q_point)[0];
            for (unsigned int d=1; d<dim; ++d)
              grad_out[comp] += gradients[q_point][comp][d] *
                               fe_eval.get_normal_vector(q_point)[d];
          }
          return fe_eval.get_normal_gradient(q_point) + grad_out;
        }
      }
#endif
        return fe_eval.get_normal_gradient(q_point);
      }
      VectorizedArray<Number> get_normal_gradient(const unsigned int q_point,bool test) const
      {
#ifdef XWALL
      if(enriched)
      {
        VectorizedArray<Number> grad_out;
          grad_out = gradients[q_point][0] *
                           fe_eval.get_normal_vector(q_point)[0];
          for (unsigned int d=1; d<dim; ++d)
            grad_out += gradients[q_point][d] *
                             fe_eval.get_normal_vector(q_point)[d];

          grad_out +=  fe_eval.get_normal_gradient(q_point);
        return grad_out;
      }
#endif
        return fe_eval.get_normal_gradient(q_point);
      }

      void submit_normal_gradient (const Tensor<1,n_components_,VectorizedArray<Number> > grad_in,
                                const unsigned int q)
      {
        fe_eval.submit_normal_gradient(grad_in,q);
#ifdef XWALL
        gradients[q]=gradient_type();
      if(enriched)
      {
        for (unsigned int comp=0; comp<n_components_; comp++)
          {
            for (unsigned int d=0; d<dim; ++d)
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
              {
                if(enriched_components.at(v))
                {
                  gradients[q][comp][d][v] = grad_in[comp][v] *
                  fe_eval.get_normal_vector(q)[d][v];
                }
                else
                  gradients[q][comp][d][v] = 0.0;
              }
          }
      }
#endif
      }
      void submit_normal_gradient (const VectorizedArray<Number> grad_in,
                                const unsigned int q)
      {
        fe_eval.submit_normal_gradient(grad_in,q);
#ifdef XWALL
        gradients[q]=gradient_type();
        if(enriched)
        {
          for (unsigned int d=0; d<dim; ++d)
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                gradients[q][d][v] = grad_in[v] *
                fe_eval.get_normal_vector(q)[d][v];
              }
              else
                gradients[q][d][v] = 0.0;
            }
        }
#endif
      }
      Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
      get_curl (const unsigned int q_point) const
       {
  #ifdef XWALL
        if(enriched)
        {
          // copy from generic function into dim-specialization function
          const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
          Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
          switch (dim)
            {
            case 1:
              Assert (false,
                      ExcMessage("Computing the curl in 1d is not a useful operation"));
              break;
            case 2:
              curl[0] = grad[1][0] - grad[0][1];
              break;
            case 3:
              curl[0] = grad[2][1] - grad[1][2];
              curl[1] = grad[0][2] - grad[2][0];
              curl[2] = grad[1][0] - grad[0][1];
              break;
            default:
              Assert (false, ExcNotImplemented());
              break;
            }
          return fe_eval.get_curl(q_point) + curl;
        }
  #endif
        return fe_eval.get_curl(q_point);
       }

      VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
      {
  #ifdef XWALL
        if(enriched)
        {
          VectorizedArray<Number> returnvalue = make_vectorized_array(0.0);
          if(j<fe_eval.dofs_per_cell*n_components_)
            returnvalue = fe_eval.begin_dof_values()[j];
          else
            returnvalue = fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_];
          return returnvalue;
        }
        else
          return fe_eval.begin_dof_values()[j];
  #else

        return fe_eval.begin_dof_values()[j];
  #endif
      }
      void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
      {
  #ifdef XWALL
        if(enriched)
        {
          if(j<fe_eval.dofs_per_cell*n_components_)
            fe_eval.begin_dof_values()[j][v] = value;
          else
            fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_][v] = value;
        }
        else
          fe_eval.begin_dof_values()[j][v]=value;
        return;
  #else
        fe_eval.begin_dof_values()[j][v]=value;
        return;
  #endif
      }
      void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
      {
  #ifdef XWALL
        if(enriched)
        {
          if(j<fe_eval.dofs_per_cell*n_components_)
            fe_eval.begin_dof_values()[j] = value;
          else
            fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_] = value;
        }
        else
          fe_eval.begin_dof_values()[j]=value;
        return;
  #else
        fe_eval.begin_dof_values()[j]=value;
        return;
  #endif
      }
      void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int face, const VectorizedArray<Number> volume)
      {
        eddyvisc.resize(n_q_points);
        if(CS > 1e-10)
        {
          const VectorizedArray<Number> Cs = make_vectorized_array(CS);
          VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
          fe_eval_tauw.reinit(face);
          {
            reinit(face);
            read_dof_values(solution_n,0,solution_n,dim+1);
            evaluate (false,true,false);
            AlignedVector<VectorizedArray<Number> > wdist;
            wdist.resize(fe_eval_tauw.n_q_points);
            fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
            fe_eval_tauw.evaluate(true,false);
            for (unsigned int q=0; q<fe_eval_tauw.n_q_points; ++q)
              wdist[q] = fe_eval_tauw.get_value(q);
            fe_eval_tauw.reinit(face);
            fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
            fe_eval_tauw.evaluate(true,false);

            const VectorizedArray<Number> hvol = hfac * std::pow(volume, 1./(double)dim);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
              Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

              VectorizedArray<Number> snorm = make_vectorized_array(0.);
              for (unsigned int i = 0; i<dim ; i++)
                for (unsigned int j = 0; j<dim ; j++)
                  snorm += (s[i][j])*(s[i][j]);
              snorm *= make_vectorized_array<Number>(0.5);
              //simple wall correction
              VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw.get_value(q))*0.04));
              VectorizedArray<Number> lm = Cs*hvol*fmu;
              eddyvisc[q]= make_vectorized_array(VISCOSITY) + lm*lm*std::sqrt(snorm);
            }
          }
          //initialize again to get a clean version
          reinit(face);
        }
#ifdef XWALL
      else if (ML>0.1 && enriched)
      {
        VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
        fe_eval_tauw.reinit(face);
        {
          read_dof_values(solution_n,0,solution_n,dim+1);
          evaluate (false,true,false);
          AlignedVector<VectorizedArray<Number> > wdist;
          wdist.resize(fe_eval_tauw.n_q_points);
          fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
          fe_eval_tauw.evaluate(true,false);
          for (unsigned int q=0; q<fe_eval_tauw.n_q_points; ++q)
            wdist[q] = fe_eval_tauw.get_value(q);
          fe_eval_tauw.reinit(face);
          fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
          fe_eval_tauw.evaluate(true,false);

          for (unsigned int q=0; q<n_q_points; ++q)
          {
            Tensor<2,dim,VectorizedArray<Number> > s = get_gradient(q);
            Tensor<2,dim,VectorizedArray<Number> > om;
            for (unsigned int i=0; i<dim;i++)
              for (unsigned int j=0;j<dim;j++)
                om[i][j]=0.5*(s[i][j]-s[j][i]);

            VectorizedArray<Number> osum = make_vectorized_array(0.);
            for (unsigned int i=0; i<dim;i++)
              for (unsigned int j=0;j<dim;j++)
                osum += om[i][j]*om[i][j];
            VectorizedArray<Number> onorm = std::sqrt(2.*osum);

            //simple wall correction
            VectorizedArray<Number> l = 0.41*wdist[q]*(1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw.get_value(q))*0.04));
            VectorizedArray<Number> vt = l*l*onorm;
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
              {
                eddyvisc[q][v]= VISCOSITY + vt[v];
              }
              else
                eddyvisc[q][v]= VISCOSITY;
            }
          }
        }
        //initialize again to get a clean version
        reinit(face);
    }
#endif
        else
          for (unsigned int q=0; q<n_q_points; ++q)
            eddyvisc[q]= make_vectorized_array(VISCOSITY);

        return;
      }

      void fill_JxW_values(AlignedVector<VectorizedArray<Number> > &JxW_values) const
      {
        fe_eval.fill_JxW_values(JxW_values);
      }

    private:
      FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval;
      FEFaceEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall;
      FEFaceEvaluation<dim,1,n_q_points_1d,1,Number> fe_eval_tauw;
      bool is_left_face;
      AlignedVector<value_type> values;
      AlignedVector<gradient_type> gradients;


    public:
      unsigned int dofs_per_cell;
      unsigned int tensor_dofs_per_cell;
      const unsigned int n_q_points;
      bool enriched;
      std::vector<bool> enriched_components;
      AlignedVector<VectorizedArray<Number> > eddyvisc;
    };



  template<int dim, int fe_degree, int fe_degree_xwall>
  class XWall
  {
  //time-integration-level routines for xwall
  public:
    XWall(const DoFHandler<dim> &dof_handler,
        MatrixFree<dim,double>* data,
        double visc,
        AlignedVector<VectorizedArray<double> > &element_volume);

    //initialize everything, e.g.
    //setup of wall distance
    //setup of communication of tauw to off-wall nodes
    //setup quadrature rules
    //possibly setup new matrixfree data object only including the xwall elements
    void initialize()
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "\nXWall Initialization:" << std::endl;

      //initialize wall distance and closest wall-node connectivity
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Initialize wall distance:...";
      InitWDist();
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << " done!" << std::endl;

      //initialize some vectors
      (*mydata).initialize_dof_vector(tauw, 2);
      tauw = 1.0;
      tauw_n=tauw;
    }

    //Update wall shear stress at the beginning of every time step
    void UpdateTauW(std::vector<parallel::distributed::Vector<double> > &solution_np);

    DoFHandler<dim>* ReturnDofHandlerWallDistance(){return &dof_handler_wall_distance;}
    const parallel::distributed::Vector<double>* ReturnWDist() const
        {return &wall_distance;}
    const parallel::distributed::Vector<double>* ReturnTauW() const
        {return &tauw;}
    const parallel::distributed::Vector<double>* ReturnTauWN() const
        {return &tauw_n;}

    ConstraintMatrix* ReturnConstraintMatrix()
        {return &constraint_periodic;}

    const FE_Q<dim>* ReturnFE() const
        {return &fe_wall_distance;}

    // fill the periodicity constraints given a level 0 periodicity structure
    void initialize_constraints(const std::vector< GridTools::PeriodicFacePair< typename Triangulation<dim>::cell_iterator > > &periodic_face_pair);
  private:

    void InitWDist();

    //calculate wall shear stress based on current solution
    void CalculateWallShearStress(const std::vector<parallel::distributed::Vector<double> >   &src,
        parallel::distributed::Vector<double>      &dst);

    //element-level routines
    void local_rhs_dummy (const MatrixFree<dim,double>                &,
                          parallel::distributed::Vector<double>      &,
                          const std::vector<parallel::distributed::Vector<double> >    &,
                          const std::pair<unsigned int,unsigned int>          &) const;

    void local_rhs_wss_boundary_face(const MatrixFree<dim,double>              &data,
                      parallel::distributed::Vector<double>      &dst,
                      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>          &face_range) const;

    void local_rhs_dummy_face (const MatrixFree<dim,double>              &,
                  parallel::distributed::Vector<double>      &,
                  const std::vector<parallel::distributed::Vector<double> >  &,
                  const std::pair<unsigned int,unsigned int>          &) const;

    void local_rhs_normalization_boundary_face(const MatrixFree<dim,double>              &data,
                      parallel::distributed::Vector<double>      &dst,
                      const std::vector<parallel::distributed::Vector<double> >  &,
                      const std::pair<unsigned int,unsigned int>          &face_range) const;

    //continuous vectors with linear interpolation
    FE_Q<dim> fe_wall_distance;
    DoFHandler<dim> dof_handler_wall_distance;
    parallel::distributed::Vector<double> wall_distance;
    parallel::distributed::Vector<double> tauw_boundary;
    std::vector<unsigned int> vector_to_tauw_boundary;
    parallel::distributed::Vector<double> tauw;
    parallel::distributed::Vector<double> tauw_n;
    MatrixFree<dim,double>* mydata;
    double viscosity;
//    parallel::distributed::Vector<double> &eddy_viscosity;
    AlignedVector<VectorizedArray<double> >& element_volume;
    ConstraintMatrix constraint_periodic;

  public:

  };

  template<int dim, int fe_degree, int fe_degree_xwall>
  XWall<dim,fe_degree,fe_degree_xwall>::XWall(const DoFHandler<dim> &dof_handler,
      MatrixFree<dim,double>* data,
      double visc,
      AlignedVector<VectorizedArray<double> > &element_volume)
  :fe_wall_distance(QGaussLobatto<1>(1+1)),
   dof_handler_wall_distance(dof_handler.get_triangulation()),
   mydata(data),
   viscosity(visc),
   element_volume(element_volume)
  {
    dof_handler_wall_distance.distribute_dofs(fe_wall_distance);
    dof_handler_wall_distance.distribute_mg_dofs(fe_wall_distance);
  }

  template<int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::InitWDist()
  {
    // layout of aux_vector: 0-dim: normal, dim: distance, dim+1: nearest dof
    // index, dim+2: touch count (for computing weighted normals); normals not
    // currently used
    std::vector<parallel::distributed::Vector<double> > aux_vectors(dim+3);

    // store integer indices in a double. In order not to get overflow, we
    // need to make sure the global index fits into a double -> this limits
    // the maximum size in the dof indices to 2^53 (approx 10^15)
#ifdef DEAL_II_WITH_64BIT_INTEGERS
    AssertThrow(dof_handler_wall_distance.n_dofs() <
                (types::global_dof_index(1ull) << 53),
                ExcMessage("Sizes larger than 2^53 currently not supported"));
#endif

    IndexSet locally_relevant_set;
    DoFTools::extract_locally_relevant_dofs(dof_handler_wall_distance,
                                            locally_relevant_set);
    aux_vectors[0].reinit(dof_handler_wall_distance.locally_owned_dofs(),
                          locally_relevant_set, MPI_COMM_WORLD);
    for (unsigned int d=1; d<aux_vectors.size(); ++d)
      aux_vectors[d].reinit(aux_vectors[0]);

    // assign distance to close to infinity (we would like to use inf here but
    // there are checks in deal.II whether numbers are finite so we must use a
    // finite number here)
    const double unreached = 1e305;
    aux_vectors[dim] = unreached;

    // TODO: get the actual set of wall (Dirichlet) boundaries as input
    // arguments. Currently, this is matched with what is set in the outer
    // problem type.
    std::set<types::boundary_id> wall_boundaries;
    wall_boundaries.insert(0);

    // set the initial distance for the wall to zero and initialize the normal
    // directions
    {
      QGauss<dim-1> face_quadrature(1);
      FEFaceValues<dim> fe_face_values(fe_wall_distance, face_quadrature,
                                       update_normal_vectors);
      std::vector<types::global_dof_index> dof_indices(fe_wall_distance.dofs_per_face);
      int found = 0;
      for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_wall_distance.begin_active(); cell != dof_handler_wall_distance.end(); ++cell)
        if (cell->is_locally_owned())
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) &&
                wall_boundaries.find(cell->face(f)->boundary_id()) !=
                wall_boundaries.end())
              {
                found = 1;
                cell->face(f)->get_dof_indices(dof_indices);
                // get normal vector on face
                fe_face_values.reinit(cell, f);
                const Tensor<1,dim> normal = fe_face_values.normal_vector(0);
                for (unsigned int i=0; i<dof_indices.size(); ++i)
                  {
                    for (unsigned int d=0; d<dim; ++d)
                      aux_vectors[d](dof_indices[i]) += normal[d];
                    aux_vectors[dim](dof_indices[i]) = 0.;
                    if(constraint_periodic.is_constrained(dof_indices[i]))
                      aux_vectors[dim+1](dof_indices[i]) = (*constraint_periodic.get_constraint_entries(dof_indices[i]))[0].first;
                    else
                      aux_vectors[dim+1](dof_indices[i]) = dof_indices[i];
                    aux_vectors[dim+2](dof_indices[i]) += 1.;
                  }
              }
      int found_global = Utilities::MPI::sum(found,MPI_COMM_WORLD);
      //at least one processor has to have walls
      AssertThrow(found_global>0, ExcMessage("Could not find any wall. Aborting."));
      for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
        if (aux_vectors[dim+2].local_element(i) != 0)
          for (unsigned int d=0; d<dim; ++d)
            aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
    }

    // this algorithm finds the closest point on the interface by simply
    // searching locally on each element. This algorithm is only correct for
    // simple meshes (as it searches purely locally and can result in zig-zag
    // paths that are nowhere near optimal on general meshes) but it works in
    // parallel when the mesh can be arbitrarily decomposed among
    // processors. A generic class of algorithms to find the closest point on
    // the wall (not necessarily on a node of the mesh) is by some interface
    // evolution method similar to finding signed distance functions to a
    // given interface (see e.g. Sethian, Level Set Methods and Fast Marching
    // Methods, 2000, Chapter 6). But I do not know how to keep track of the
    // point of origin in those algorithms which is essential here, so skip
    // that for the moment. -- MK, Dec 2015

    // loop as long as we have untracked degrees of freedom. this loop should
    // terminate after a number of steps that is approximately half the width
    // of the mesh in elements
    while (aux_vectors[dim].linfty_norm() == unreached)
      {
        aux_vectors[dim+2] = 0.;
        for (unsigned int d=0; d<dim+2; ++d)
          aux_vectors[d].update_ghost_values();

        // get a pristine vector with the content of the distances at the
        // beginning of the step to distinguish which degrees of freedom were
        // already touched before the current loop and which are in the
        // process of being updated
        parallel::distributed::Vector<double> distances_step(aux_vectors[dim]);
        distances_step.update_ghost_values();

        AssertThrow(fe_wall_distance.dofs_per_cell ==
                    GeometryInfo<dim>::vertices_per_cell, ExcNotImplemented());
        Quadrature<dim> quadrature(fe_wall_distance.get_unit_support_points());
        FEValues<dim> fe_values(fe_wall_distance, quadrature, update_quadrature_points);
        std::vector<types::global_dof_index> dof_indices(fe_wall_distance.dofs_per_cell);

        // go through all locally owned and ghosted cells and compute the
        // nearest point from within the element. Since we have both ghosted
        // and owned cells, we can be sure that the locally owned vector
        // elements get the closest point from the neighborhood
        for (typename DoFHandler<dim>::active_cell_iterator cell =
               dof_handler_wall_distance.begin_active();
             cell != dof_handler_wall_distance.end(); ++cell)
          if (!cell->is_artificial())
            {
              bool cell_is_initialized = false;
              cell->get_dof_indices(dof_indices);

              for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                // point is unreached -> find the closest point within cell
                // that is already reached
                if (distances_step(dof_indices[v]) == unreached)
                  {
                    for (unsigned int w=0; w<GeometryInfo<dim>::vertices_per_cell; ++w)
                      if (distances_step(dof_indices[w]) < unreached)
                        {
                          if (! cell_is_initialized)
                            {
                              fe_values.reinit(cell);
                              cell_is_initialized = true;
                            }

                          // here are the normal vectors in case they should
                          // be necessary in a refined version of the
                          // algorithm
                          /*
                          Tensor<1,dim> normal;
                          for (unsigned int d=0; d<dim; ++d)
                            normal[d] = aux_vectors[d](dof_indices[w]);
                          */
                          const Tensor<1,dim> distance_vec =
                            fe_values.quadrature_point(v) - fe_values.quadrature_point(w);
                          if (distances_step(dof_indices[w]) + distance_vec.norm() <
                              aux_vectors[dim](dof_indices[v]))
                            {
                              aux_vectors[dim](dof_indices[v]) =
                                distances_step(dof_indices[w]) + distance_vec.norm();
                              aux_vectors[dim+1](dof_indices[v]) =
                                aux_vectors[dim+1](dof_indices[w]);
                              for (unsigned int d=0; d<dim; ++d)
                                aux_vectors[d](dof_indices[v]) +=
                                  aux_vectors[d](dof_indices[w]);
                              aux_vectors[dim+2](dof_indices[v]) += 1;
                            }
                        }
                  }
            }
        for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
          if (aux_vectors[dim+2].local_element(i) != 0)
            for (unsigned int d=0; d<dim; ++d)
              aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
      }
    aux_vectors[dim+1].update_ghost_values();

    // at this point we could do a search for closer points in the
    // neighborhood of the points identified before (but it is probably quite
    // difficult to do and one needs to search in layers around a given point
    // to have all data available locally; I currently do not have a good idea
    // to sort out this mess and I am not sure whether we really need
    // something better than the local search above). -- MK, Dec 2015

    // copy the aux vector with extended ghosting into a vector that fits the
    // matrix-free partitioner
    (*mydata).initialize_dof_vector(wall_distance, 2);
    AssertThrow(wall_distance.local_size() == aux_vectors[dim].local_size(),
                ExcMessage("Vector sizes do not match, cannot import wall distances"));
    wall_distance = aux_vectors[dim];
    wall_distance.update_ghost_values();

    IndexSet accessed_indices(aux_vectors[dim+1].size());
    {
      // copy the accumulated indices into an index vector
      std::vector<types::global_dof_index> my_indices;
      my_indices.reserve(aux_vectors[dim+1].local_size());
      for (unsigned int i=0; i<aux_vectors[dim+1].local_size(); ++i)
        my_indices.push_back(static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));
      // sort and compress out duplicates
      std::sort(my_indices.begin(), my_indices.end());
      my_indices.erase(std::unique(my_indices.begin(), my_indices.end()),
                       my_indices.end());
      accessed_indices.add_indices(my_indices.begin(),
                                   my_indices.end());
    }

    // create partitioner for exchange of ghost data (after having computed
    // the vector of wall shear stresses)
    std_cxx11::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner
      (new Utilities::MPI::Partitioner(dof_handler_wall_distance.locally_owned_dofs(),
                                       accessed_indices, MPI_COMM_WORLD));
    tauw_boundary.reinit(vector_partitioner);

    vector_to_tauw_boundary.resize(wall_distance.local_size());
    for (unsigned int i=0; i<wall_distance.local_size(); ++i)
      vector_to_tauw_boundary[i] = vector_partitioner->global_to_local
        (static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));

  }

  template<int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::UpdateTauW(std::vector<parallel::distributed::Vector<double> > &solution_np)
  {
    //store old wall shear stress
    tauw_n.swap(tauw);

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "\nCompute new tauw: ";
    CalculateWallShearStress(solution_np,tauw);
    //mean does not work currently because of all off-wall nodes in the vector
//    double tauwmean = tauw.mean_value();
//    std::cout << "mean = " << tauwmean << " ";

    double tauwmax = tauw.linfty_norm();
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "max = " << tauwmax << " ";

    double minloc = 1e9;
    for(unsigned int i = 0; i < tauw.local_size(); ++i)
    {
      if(tauw.local_element(i)>0.0)
      {
        if(minloc > tauw.local_element(i))
          minloc = tauw.local_element(i);
      }
    }
    const double minglob = Utilities::MPI::min(minloc, MPI_COMM_WORLD);

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "min = " << minglob << " ";
    if(not variabletauw)
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "(manually set to 1.0) ";
      tauw = 1.0;
    }
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << std::endl;
    tauw.update_ghost_values();
  }

  template<int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim, fe_degree,fe_degree_xwall>::
  CalculateWallShearStress (const std::vector<parallel::distributed::Vector<double> >   &src,
            parallel::distributed::Vector<double>      &dst)
  {
    parallel::distributed::Vector<double> normalization;
    (*mydata).initialize_dof_vector(normalization, 2);
    parallel::distributed::Vector<double> force;
    (*mydata).initialize_dof_vector(force, 2);

    // initialize
    force = 0.0;
    normalization = 0.0;

    // run loop to compute the local integrals
    (*mydata).loop (&XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy_face,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_wss_boundary_face,
              this, force, src);

    (*mydata).loop (&XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy_face,
        &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_normalization_boundary_face,
              this, normalization, src);

    // run normalization
    double mean = 0.0;
    unsigned int count = 0;
    for(unsigned int i = 0; i < force.local_size(); ++i)
    {
      if(normalization.local_element(i)>0.0)
      {
        tauw_boundary.local_element(i) = force.local_element(i) / normalization.local_element(i);
        mean += tauw_boundary.local_element(i);
        count++;
      }
    }
    mean = Utilities::MPI::sum(mean,MPI_COMM_WORLD);
    count = Utilities::MPI::sum(count,MPI_COMM_WORLD);
    mean /= (double)count;
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "mean = " << mean << " ";

    // communicate the boundary values for the shear stress to the calling
    // processor and access the data according to the vector_to_tauw_boundary
    // field
    tauw_boundary.update_ghost_values();

    for (unsigned int i=0; i<tauw.local_size(); ++i)
      dst.local_element(i) = (1.-DTAUW)*tauw_n.local_element(i)+DTAUW*tauw_boundary.local_element(vector_to_tauw_boundary[i]);
    dst.update_ghost_values();
  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_dummy (const MatrixFree<dim,double>                &,
              parallel::distributed::Vector<double>      &,
              const std::vector<parallel::distributed::Vector<double> >  &,
              const std::pair<unsigned int,unsigned int>           &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_wss_boundary_face (const MatrixFree<dim,double>             &data,
                         parallel::distributed::Vector<double>    &dst,
                         const std::vector<parallel::distributed::Vector<double> >  &src,
                         const std::pair<unsigned int,unsigned int>          &face_range) const
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,double> fe_eval_xwall(data,wall_distance,tauw,true,0,3);
    FEFaceEvaluation<dim,1,n_q_points_1d_xwall,1,double> fe_eval_tauw(data,true,2,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,double> fe_eval_xwall(data,wall_distance,tauw,true,0,0);
    FEFaceEvaluation<dim,1,fe_degree+1,1,double> fe_eval_tauw(data,true,2,0);
#endif
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        fe_eval_xwall.reinit (face);
        fe_eval_xwall.evaluate_eddy_viscosity(src,face,fe_eval_xwall.read_cell_data(element_volume));
        fe_eval_tauw.reinit (face);

        fe_eval_xwall.read_dof_values(src,0,src,dim+1);
        fe_eval_xwall.evaluate(false,true);
        if(fe_eval_xwall.n_q_points != fe_eval_tauw.n_q_points)
          std::cerr << "\nwrong number of quadrature points" << std::endl;

        for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
        {
          Tensor<1, dim, VectorizedArray<double> > average_gradient = fe_eval_xwall.get_normal_gradient(q);

          VectorizedArray<double> tauwsc = make_vectorized_array<double>(0.0);
          tauwsc = average_gradient.norm();
          tauwsc *= fe_eval_xwall.eddyvisc[q];
          fe_eval_tauw.submit_value(tauwsc,q);
        }
        fe_eval_tauw.integrate(true,false);
        fe_eval_tauw.distribute_local_to_global(dst);
      }
    }
  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_normalization_boundary_face (const MatrixFree<dim,double>             &data,
                         parallel::distributed::Vector<double>    &dst,
                         const std::vector<parallel::distributed::Vector<double> >  &,
                         const std::pair<unsigned int,unsigned int>          &face_range) const
  {
    FEFaceEvaluation<dim,1,fe_degree+1,1,double> fe_eval_tauw(data,true,2,0);
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        fe_eval_tauw.reinit (face);

        for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
          fe_eval_tauw.submit_value(make_vectorized_array<double>(1.0),q);

        fe_eval_tauw.integrate(true,false);
        fe_eval_tauw.distribute_local_to_global(dst);
      }
    }
  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  local_rhs_dummy_face (const MatrixFree<dim,double>                 &,
                parallel::distributed::Vector<double>      &,
                const std::vector<parallel::distributed::Vector<double> >  &,
                const std::pair<unsigned int,unsigned int>          &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_xwall>
  void XWall<dim,fe_degree,fe_degree_xwall>::
  initialize_constraints(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs)
  {
    IndexSet xwall_relevant_set;
    DoFTools::extract_locally_relevant_dofs(dof_handler_wall_distance,
                                            xwall_relevant_set);
    constraint_periodic.clear();
    constraint_periodic.reinit(xwall_relevant_set);
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> >
      periodic_face_pairs_dh(periodic_face_pairs.size());
    for (unsigned int i=0; i<periodic_face_pairs.size(); ++i)
      {
        GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> pair;
        pair.cell[0] = typename DoFHandler<dim>::cell_iterator
          (&periodic_face_pairs[i].cell[0]->get_triangulation(),
           periodic_face_pairs[i].cell[0]->level(),
           periodic_face_pairs[i].cell[0]->index(),
           &dof_handler_wall_distance);
        pair.cell[1] = typename DoFHandler<dim>::cell_iterator
          (&periodic_face_pairs[i].cell[1]->get_triangulation(),
           periodic_face_pairs[i].cell[1]->level(),
           periodic_face_pairs[i].cell[1]->index(),
           &dof_handler_wall_distance);
        pair.face_idx[0] = periodic_face_pairs[i].face_idx[0];
        pair.face_idx[1] = periodic_face_pairs[i].face_idx[1];
        pair.orientation = periodic_face_pairs[i].orientation;
        pair.matrix = periodic_face_pairs[i].matrix;
        periodic_face_pairs_dh[i] = pair;
      }
    DoFTools::make_periodicity_constraints<DoFHandler<dim> >(periodic_face_pairs_dh, constraint_periodic);
    DoFTools::make_hanging_node_constraints(dof_handler_wall_distance,
                                            constraint_periodic);

    constraint_periodic.close();
  }



  /// Collect all data for the inverse mass matrix operation in a struct in
  /// order to avoid allocating the memory repeatedly.
  template <int dim, int fe_degree, typename Number>
  struct InverseMassMatrixData
  {
    InverseMassMatrixData(const MatrixFree<dim,Number> &data,
                          const unsigned int fe_index = 0,
                          const unsigned int quad_index = 0)
      :
      phi(1, FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number>(data,fe_index,
                                                                quad_index)),
      coefficients(FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number>::n_q_points),
      inverse(phi[0])
    {}

    // Manually implement the copy operator because CellwiseInverseMassMatrix
    // must point to the object 'phi'
    InverseMassMatrixData(const InverseMassMatrixData &other)
      :
      phi(other.phi),
      coefficients(other.coefficients),
      inverse(phi[0])
    {}

    // For memory alignment reasons, need to place the FEEvaluation object
    // into an aligned vector
    AlignedVector<FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> > phi;
    AlignedVector<VectorizedArray<Number> > coefficients;
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,Number> inverse;
  };



  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  class NavierStokesOperation
  {
public:
  typedef double value_type;
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;

  NavierStokesOperation(const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler,const DoFHandler<dim> &dof_handler_p, const DoFHandler<dim> &dof_handler_xwall, const double time_step_size,
      const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
      std::set<types::boundary_id> dirichlet_bc_indicator, std::set<types::boundary_id> neumann_bc_indicator);

  ~NavierStokesOperation()
  {
    data.clear();
  }

  void do_timestep (const double  &cur_time,const double  &delta_t, const unsigned int &time_step_number);

  void  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> > &src,
                std::vector<parallel::distributed::Vector<value_type> >    &dst);

  void  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst);

  void  apply_viscous (const parallel::distributed::BlockVector<value_type>     &src,
                   parallel::distributed::BlockVector<value_type>      &dst) const;

  void  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >   &src,
                   parallel::distributed::BlockVector<value_type>  &dst);

  void  shift_pressure (parallel::distributed::Vector<value_type>  &pressure);

  void apply_inverse_mass_matrix(const parallel::distributed::BlockVector<value_type>  &src,
                parallel::distributed::BlockVector<value_type>    &dst) const;

  void precompute_inverse_mass_matrix();

  void xwall_projection();

  void  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                parallel::distributed::Vector<value_type>      &dst);

  void rhs_projection (const std::vector<parallel::distributed::Vector<value_type> > &src,
			std::vector<parallel::distributed::Vector<value_type> > &dst);

  void  apply_weak_projection (const parallel::distributed::BlockVector<value_type>      &src,
                   parallel::distributed::BlockVector<value_type>       &dst);

  void  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                   std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                      std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void analyse_computing_times();

  std::vector<parallel::distributed::Vector<value_type> > solution_nm2, solution_nm, solution_n, velocity_temp, solution_np;
  std::vector<parallel::distributed::Vector<value_type> > vorticity_nm2, vorticity_nm, vorticity_n;
  std::vector<parallel::distributed::Vector<value_type> > rhs_convection_nm2, rhs_convection_nm, rhs_convection_n, rhs_proj;
  std::vector<parallel::distributed::Vector<value_type> > f;
  std::vector<parallel::distributed::Vector<value_type> > xwallstatevec;
  parallel::distributed::BlockVector<value_type> rhs_visc;
  parallel::distributed::BlockVector<value_type> solution_temp_visc;

  parallel::distributed::BlockVector<value_type> rhs_proj_block;
  parallel::distributed::BlockVector<value_type> solution_proj_block;

  parallel::distributed::Vector<value_type> rhs_p;
#ifdef COMPDIV
  parallel::distributed::Vector<value_type> divergence_old, divergence_new;
#endif

  const MatrixFree<dim,value_type> & get_data() const
  {
    return data;
  }

  void calculate_diagonal_viscous(std::vector<parallel::distributed::Vector<value_type> > &diagonal, unsigned int level) const;

  XWall<dim,fe_degree,fe_degree_xwall>* ReturnXWall(){return &xwall;}

private:
  MatrixFree<dim,value_type> data;

  MappingQ<dim> mapping;

  double time, time_step;
  const double viscosity;
  double gamma0;
  double alpha[3], beta[3];

  Timer total_time;

  std::vector<double> computing_times;
  std::vector<double> times_cg_velo;
  std::vector<unsigned int> iterations_cg_velo;
  std::vector<double> times_cg_pressure;
  std::vector<unsigned int> iterations_cg_pressure;
  PoissonSolver<dim> pressure_poisson_solver;

  AlignedVector<VectorizedArray<value_type> > element_volume;
  AlignedVector<VectorizedArray<value_type> > array_stab_fac_weak_projection;

  Point<dim> first_point;
  types::global_dof_index dof_index_first_point;

  XWall<dim,fe_degree,fe_degree_xwall> xwall;
  std::vector<Table<2,VectorizedArray<value_type> > > matrices;
  std::vector<std::vector<std::vector<LAPACKFullMatrix<value_type> > > > div_matrices;

  mutable std_cxx11::shared_ptr<Threads::ThreadLocalStorage<InverseMassMatrixData<dim,fe_degree,value_type> > > mass_matrix_data;

  std::set<types::boundary_id> dirichlet_boundary;
  std::set<types::boundary_id> neumann_boundary;

  void update_time_integrator(unsigned int time_step_number);
  void check_time_integrator(unsigned int time_step_number);

  // impulse equation
  void local_rhs_convection (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_convection_face (const MatrixFree<dim,value_type>              &data,
                  std::vector<parallel::distributed::Vector<double> >      &dst,
                  const std::vector<parallel::distributed::Vector<double> >  &src,
                  const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_convection_boundary_face(const MatrixFree<dim,value_type>              &data,
                      std::vector<parallel::distributed::Vector<double> >      &dst,
                      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_compute_rhs (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >     &dst,
                        const std::vector<parallel::distributed::Vector<double> >   &,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_apply_weak_projection (const MatrixFree<dim,value_type>        &data,
                        parallel::distributed::BlockVector<double>       &dst,
                        const parallel::distributed::BlockVector<double>  &src,
                        const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_apply_weak_projection_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::BlockVector<double>       &dst,
                  const parallel::distributed::BlockVector<double>   &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_weak_projection_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::BlockVector<double>       &dst,
                      const parallel::distributed::BlockVector<double>   &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_viscous (const MatrixFree<dim,value_type>        &data,
                        parallel::distributed::BlockVector<double>      &dst,
                        const parallel::distributed::BlockVector<double>  &src,
                        const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_apply_viscous_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::BlockVector<double>      &dst,
                  const parallel::distributed::BlockVector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_viscous_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::BlockVector<double>      &dst,
                      const parallel::distributed::BlockVector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_rhs_viscous (const MatrixFree<dim,value_type>                &data,
                        parallel::distributed::BlockVector<double>      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_viscous_face (const MatrixFree<dim,value_type>              &data,
                parallel::distributed::BlockVector<double>      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_viscous_boundary_face(const MatrixFree<dim,value_type>              &data,
                    parallel::distributed::BlockVector<double>      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_diagonal_viscous(const MatrixFree<dim,value_type>        &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_diagonal_viscous_face (const MatrixFree<dim,value_type>      &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_diagonal_viscous_boundary_face(const MatrixFree<dim,value_type>      &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_rhs_pressure (const MatrixFree<dim,value_type>                &data,
                        parallel::distributed::Vector<double>      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_pressure_face (const MatrixFree<dim,value_type>              &data,
                parallel::distributed::Vector<double>      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_pressure_boundary_face(const MatrixFree<dim,value_type>              &data,
                    parallel::distributed::Vector<double>      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  // inverse mass matrix velocity
  void local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      std::vector<parallel::distributed::Vector<value_type> >    &dst,
                      const std::vector<parallel::distributed::Vector<value_type> >  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range) const;
  // inverse mass matrix velocity
  void local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      parallel::distributed::BlockVector<value_type>    &dst,
                      const parallel::distributed::BlockVector<value_type>  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void   local_compute_divergence (const MatrixFree<dim,value_type>        &data,
      parallel::distributed::Vector<value_type>    &dst,
      const std::vector<parallel::distributed::Vector<value_type> >  &src,
      const std::pair<unsigned int,unsigned int>   &cell_range) const;
  // inverse mass matrix velocity
  void local_precompute_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      std::vector<parallel::distributed::Vector<value_type> >    &,
                      const std::vector<parallel::distributed::Vector<value_type> >  &,
                      const std::pair<unsigned int,unsigned int>          &cell_range);

  // inverse mass matrix velocity
  void local_project_xwall(const MatrixFree<dim,value_type>                &data,
                      std::vector<parallel::distributed::Vector<value_type> >    &dst,
                      const std::vector<parallel::distributed::Vector<value_type> >  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range);

  // projection step
  void local_rhs_projection (const MatrixFree<dim,value_type>                &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >    &src,
                    const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_projection_face (const MatrixFree<dim,value_type>              &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_projection_boundary_face (const MatrixFree<dim,value_type>              &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_compute_vorticity (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >      &dst,
                            const std::vector<parallel::distributed::Vector<double> >    &src,
                            const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_grad_div_projection (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >      &dst,
                            const std::vector<parallel::distributed::Vector<double> >    &src,
                            const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_precompute_grad_div_projection (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >      &,
                            const std::vector<parallel::distributed::Vector<double> >    &,
                            const std::pair<unsigned int,unsigned int>          &cell_range);

  void local_fast_grad_div_projection (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >      &dst,
                            const std::vector<parallel::distributed::Vector<double> >    &src,
                            const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_solve_projection (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >      &dst,
                            const std::vector<parallel::distributed::Vector<double> >    &src,
                            const std::pair<unsigned int,unsigned int>          &cell_range);

  //penalty parameter
  void calculate_penalty_parameter(double &factor) const;

  void compute_lift_and_drag(const bool clear_files);

  void compute_pressure_difference(const bool clear_files);

  void calculate_stab_fac_weak_projection();

  void local_calculate_stab_fac_weak_projection(const MatrixFree<dim,value_type>        &data,
                                                std::vector<parallel::distributed::Vector<value_type> > &,
                                                const std::vector<parallel::distributed::Vector<value_type> > &,
                                                const std::pair<unsigned int,unsigned int>   &cell_range);

  void my_point_value(const Mapping<dim> &mapping,
                 const DoFHandler<dim> &dof_handler,
                 const parallel::distributed::Vector<double> &solution,
                 const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > &cell_point,
                 Vector<double> &value);
  public:
  void local_compute_divu_for_channel_stats (const MatrixFree<dim,value_type>                &data,
                            std::vector<double >      &test,
                            const std::vector<parallel::distributed::Vector<double> >    &source,
                            const std::pair<unsigned int,unsigned int>          &cell_range);
  void local_compute_divu_for_channel_stats_face (const MatrixFree<dim,double>              &data,
                  std::vector<double >      &test,
                  const std::vector<parallel::distributed::Vector<double> >  &source,
                  const std::pair<unsigned int,unsigned int>          &face_range) ;
  void local_compute_divu_for_channel_stats_boundary_face (const MatrixFree<dim,double>              &data,
                  std::vector<double >      &test,
                  const std::vector<parallel::distributed::Vector<double> >  &source,
                  const std::pair<unsigned int,unsigned int>          &face_range) ;
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::NavierStokesOperation(
                                                                       const Mapping<dim> &mapping,
                                                                       const DoFHandler<dim> &dof_handler,
                                                                       const DoFHandler<dim> &dof_handler_p,
                                                                       const DoFHandler<dim> &dof_handler_xwall,
                                                                       const double time_step_size,
                                                                       const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
                                                                       std::set<types::boundary_id> dirichlet_bc_indicator,
                                                                       std::set<types::boundary_id> neumann_bc_indicator):
#ifdef XWALL
  rhs_visc(dim*2),
  solution_temp_visc(dim*2),
#else
  rhs_visc(dim),
  solution_temp_visc(dim),
#endif
  rhs_proj_block(dim),
  solution_proj_block(dim),
  mapping(fe_degree),
  time(0.0),
  time_step(time_step_size),
  viscosity(VISCOSITY),
  gamma0(1.0),
  computing_times(5),
  times_cg_velo(3),
  iterations_cg_velo(3),
  times_cg_pressure(2),
  iterations_cg_pressure(2),
  element_volume(0),
  array_stab_fac_weak_projection(0),
  xwall(dof_handler,&data,viscosity,element_volume),
  dirichlet_boundary(dirichlet_bc_indicator),
  neumann_boundary(neumann_bc_indicator)
  {
    alpha[0] = beta[0] = 1.;
    alpha[1] = alpha[2] = beta[1] = beta[2] = 0.;
    total_time.restart();

  // use this gamma0 when initializing the multigrid solver (in function calculate_diagonal_viscous())
  if(ORDER_TIME_INTEGRATOR == 2)
    gamma0 = 3.0/2.0;
  else if(ORDER_TIME_INTEGRATOR == 3)
    gamma0 = 11.0/6.0;

  xwall.initialize_constraints(periodic_face_pairs);

  // initialize matrix_free_data
  typename MatrixFree<dim,value_type>::AdditionalData additional_data;
  additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim,value_type>::AdditionalData::partition_partition;
  additional_data.build_face_info = true;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points | update_normal_vectors |
                                          update_values);
  additional_data.periodic_face_pairs_level_0 = periodic_face_pairs;

  std::vector<const DoFHandler<dim> * >  dof_handler_vec;
  dof_handler_vec.push_back(&dof_handler);
  dof_handler_vec.push_back(&dof_handler_p);
  dof_handler_vec.push_back((xwall.ReturnDofHandlerWallDistance()));
  dof_handler_vec.push_back(&dof_handler_xwall);

  ConstraintMatrix constraint, constraint_p;
  constraint.close();
  constraint_p.close();
  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  constraint_matrix_vec.push_back(&constraint);
  constraint_matrix_vec.push_back(&constraint_p);
  constraint_matrix_vec.push_back(xwall.ReturnConstraintMatrix());
  constraint_matrix_vec.push_back(&constraint);

  std::vector<Quadrature<1> > quadratures;
  quadratures.push_back(QGauss<1>(fe_degree+1));
  quadratures.push_back(QGauss<1>(fe_degree_p+1));
  // quadrature formula 2: exact integration of convective term
  quadratures.push_back(QGauss<1>(fe_degree + (fe_degree+2)/2));
  quadratures.push_back(QGauss<1>(n_q_points_1d_xwall));

//  const MappingQ<dim> mapping(fe_degree);

  data.reinit (mapping, dof_handler_vec, constraint_matrix_vec,
               quadratures, additional_data);

  // generate initial mass matrix data to avoid allocating it over and over
  // again
  mass_matrix_data.reset(new Threads::ThreadLocalStorage<InverseMassMatrixData<dim,fe_degree,value_type> >(InverseMassMatrixData<dim,fe_degree,value_type>(data,0,0)));

  // PCG - solver for pressure
  PoissonSolverData<dim> solver_data;
  solver_data.poisson_dof_index = 1;
  solver_data.poisson_quad_index = 1;
  solver_data.periodic_face_pairs_level0 = periodic_face_pairs;
  solver_data.penalty_factor = stab_factor_pressure; //stab_factor/nu*delta_t;
  solver_data.solver_tolerance = REL_TOL_PRESSURE;
  solver_data.dirichlet_boundaries = neumann_boundary;
  solver_data.neumann_boundaries = dirichlet_boundary;
  solver_data.coarse_solver = PoissonSolverData<dim>::coarse_chebyshev_smoother;//coarse_chebyshev_smoother;//coarse_iterative_jacobi;
  pressure_poisson_solver.initialize(mapping, data, solver_data);

//  smoother_data_viscous.smoothing_range = 30;
//  smoother_data_viscous.degree = 5; //empirically: use degree = 3 - 6
//  smoother_data_viscous.eig_cg_n_iterations = 30;
//  mg_smoother_viscous.initialize(mg_matrices_viscous, smoother_data_viscous);
  gamma0 = 1.0;

  // initialize solution vectors
  solution_n.resize(dim+1+dim);
  data.initialize_dof_vector(solution_n[0], 0);
  for (unsigned int d=1;d<dim;++d)
  {
    solution_n[d] = solution_n[0];
  }
  data.initialize_dof_vector(solution_n[dim], 1);
  data.initialize_dof_vector(solution_n[dim+1], 3);
  for (unsigned int d=1;d<dim;++d)
  {
    solution_n[dim+d+1] = solution_n[dim+1];
  }
  solution_nm2 = solution_n;
  solution_nm = solution_n;
  solution_np = solution_n;
  rhs_p = solution_n[dim];

  rhs_proj.resize(dim);
  data.initialize_dof_vector(rhs_proj[0], 0);
  for (unsigned int d=1;d<dim;++d)
  {
    rhs_proj[d] = rhs_proj[0];
  }
  for (unsigned int d=0;d<dim;++d)
  {
    rhs_proj_block.block(d) = rhs_proj[d];
    solution_proj_block.block(d) = rhs_proj[d];
  } 

  velocity_temp.resize(2*dim);
  data.initialize_dof_vector(velocity_temp[0],0);
  data.initialize_dof_vector(velocity_temp[dim],3);
  for (unsigned int d=1;d<dim;++d)
  {
    velocity_temp[d] = velocity_temp[0];
    velocity_temp[d+dim] = velocity_temp[dim];
  }
  for (unsigned int d=0;d<dim;++d)
  {
    rhs_visc.block(d)=velocity_temp[d];
    solution_temp_visc.block(d)=velocity_temp[d];
#ifdef XWALL
    rhs_visc.block(d+dim)=velocity_temp[d+dim];
    solution_temp_visc.block(d+dim)=velocity_temp[d+dim];
#endif
  }

  vorticity_n.resize(2*number_vorticity_components);
  data.initialize_dof_vector(vorticity_n[0]);
  for (unsigned int d=1;d<number_vorticity_components;++d)
  {
    vorticity_n[d] = vorticity_n[0];
  }
  data.initialize_dof_vector(vorticity_n[number_vorticity_components],3);
  for (unsigned int d=1;d<number_vorticity_components;++d)
  {
    vorticity_n[d+number_vorticity_components] = vorticity_n[number_vorticity_components];
  }
  vorticity_nm2 = vorticity_n;
  vorticity_nm = vorticity_n;
#ifdef COMPDIV
  data.initialize_dof_vector(divergence_old,0);
  divergence_new = divergence_old;
#endif
//  data.initialize_dof_vector(eddy_viscosity,0);

  rhs_convection_n.resize(2*dim);
#ifdef XWALL
  data.initialize_dof_vector(rhs_convection_n[0],0);
  data.initialize_dof_vector(rhs_convection_n[dim],3);
  for (unsigned int d=1;d<dim;++d)
  {
    rhs_convection_n[d] = rhs_convection_n[0];
    rhs_convection_n[d+dim] = rhs_convection_n[dim];
  }
  rhs_convection_nm2 = rhs_convection_n;
  rhs_convection_nm = rhs_convection_n;
  f = rhs_convection_n;
#else
  data.initialize_dof_vector(rhs_convection_n[0],0);
  for (unsigned int d=1;d<dim;++d)
  {
    rhs_convection_n[d] = rhs_convection_n[0];
  }
  rhs_convection_nm2 = rhs_convection_n;
  rhs_convection_nm = rhs_convection_n;
  f = rhs_convection_n;
#endif

  dof_index_first_point = 0;
  for(unsigned int d=0;d<dim;++d)
    first_point[d] = 0.0;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    typename DoFHandler<dim>::active_cell_iterator first_cell;
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(), endc = dof_handler_p.end();
    for(;cell!=endc;++cell)
      if (cell->is_locally_owned())
      {
        first_cell = cell;
        break;
      }
  FEValues<dim> fe_values(dof_handler_p.get_fe(),
              Quadrature<dim>(dof_handler_p.get_fe().get_unit_support_points()),
              update_quadrature_points);
  fe_values.reinit(first_cell);
  first_point = fe_values.quadrature_point(0);
  std::vector<types::global_dof_index>
  dof_indices(dof_handler_p.get_fe().dofs_per_cell);
  first_cell->get_dof_indices(dof_indices);
  dof_index_first_point = dof_indices[0];
  }
  dof_index_first_point = Utilities::MPI::sum(dof_index_first_point,MPI_COMM_WORLD);
  for(unsigned int d=0;d<dim;++d)
    first_point[d] = Utilities::MPI::sum(first_point[d],MPI_COMM_WORLD);
  xwall.initialize();
  xwallstatevec.push_back(*xwall.ReturnWDist());
  xwallstatevec.push_back(*xwall.ReturnTauW());
  //make sure that these vectors are available on the ghosted elements
  xwallstatevec[0].update_ghost_values();
  xwallstatevec[1].update_ghost_values();
#ifdef XWALL
  matrices.resize(data.n_macro_cells());
#endif
#ifndef LOWMEMORY
#ifndef XWALL
  {
    std::vector<parallel::distributed::Vector<value_type> > dummy;
    div_matrices.resize(data.n_macro_cells());
    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_precompute_grad_div_projection,this, dummy, dummy);
  }
  double sum = Utilities::MPI::sum(MemoryConsumption::memory_consumption(div_matrices), MPI_COMM_WORLD);
  if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "memory consumption total: " << sum << std::endl;
#endif
#endif
  QGauss<dim> quadrature(fe_degree+1);
  FEValues<dim> fe_values(mapping, dof_handler.get_fe(), quadrature, update_JxW_values);
  element_volume.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  for (unsigned int i=0; i<data.n_macro_cells()+data.n_macro_ghost_cells(); ++i)
    for (unsigned int v=0; v<data.n_components_filled(i); ++v)
      {
        typename DoFHandler<dim>::cell_iterator cell = data.get_cell_iterator(i,v);
        fe_values.reinit(cell);
        double volume = 0.;
        for (unsigned int q=0; q<quadrature.size(); ++q)
          volume += fe_values.JxW(q);
        element_volume[i][v] = volume;
        //pcout << "surface to volume ratio: " << pressure_poisson_solver.get_matrix().get_array_penalty_parameter()[i][v] << std::endl;
      }
  array_stab_fac_weak_projection.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct WeakProjectionMatrix : public Subscriptor
  {
    void initialize(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    {
      ns_operation = &ns_op;
    }
    void vmult (parallel::distributed::BlockVector<double> &dst,
        const parallel::distributed::BlockVector<double> &src) const
    {
      ns_operation->apply_weak_projection(src,dst);
    }
    NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
  };

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct NavierStokesViscousMatrix : public Subscriptor
  {
    void initialize(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    {
      ns_operation = &ns_op;
    }
    void vmult (parallel::distributed::BlockVector<double> &dst,
        const parallel::distributed::BlockVector<double> &src) const
    {
      ns_operation->apply_viscous(src,dst);
    }

    NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
  };

//  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  struct NavierStokesViscousMatrix : public Subscriptor
//  {
//      void initialize(const NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op, unsigned int lvl)
//      {
//        ns_operation = &ns_op;
//        level = lvl;
//        ns_operation->get_data(level).initialize_dof_vector(diagonal.block(0),0);
//        ns_operation->get_data(level).initialize_dof_vector(diagonal.block(1),3);
//        std::vector<parallel::distributed::Vector<double> >  dst_tmp;
//        dst_tmp.resize(2);
//        ns_operation->calculate_diagonal_viscous(dst_tmp,level);
//        diagonal.block(0)=dst_tmp.at(0);
//        diagonal.block(1)=dst_tmp.at(1);
//      }
//
//      unsigned int m() const
//      {
//        return ns_operation->get_data(level).get_vector_partitioner(0)->size()+ns_operation->get_data(level).get_vector_partitioner(3)->size();
//      }
//
//      double el(const unsigned int row,const unsigned int /*col*/) const
//      {
//        return diagonal(row);
//      }
//
////      void vmult (parallel::distributed::Vector<double> &dst,
////          const parallel::distributed::Vector<double> &src) const
////      {
////        Assert(false,ExcInternalError());
//////        dst = 0;
//////        vmult_add(dst,src);
////      }
//      void vmult (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        dst.block(0) = 0;
//        dst.block(1) = 0;
//        vmult_add(dst,src);
//      }
//
//      void Tvmult (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        dst.block(0) = 0;
//        dst.block(1) = 0;
//        vmult_add(dst,src);
//      }
//
//      void Tvmult_add (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        vmult_add(dst,src);
//      }
//
//      void vmult_add (parallel::distributed::BlockVector<double> &dst,
//          const parallel::distributed::BlockVector<double> &src) const
//      {
//        ns_operation->apply_viscous(src,dst,level);
//      }
//
//      const NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
//      unsigned int level;
//      parallel::distributed::BlockVector<double> diagonal;
//  };

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct PreconditionerInverseMassMatrix
  {
    PreconditionerInverseMassMatrix(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    :
      ns_op(ns_op)
    {}

    void vmult (parallel::distributed::BlockVector<double> &dst,
        const parallel::distributed::BlockVector<double> &src) const
    {
      ns_op.apply_inverse_mass_matrix(src,dst);
    }

    NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op;
  };



  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  do_timestep (const double &cur_time, const double  &delta_t, const unsigned int &time_step_number)
  {
    time = cur_time;
    time_step = delta_t;

    if(START_WITH_LOW_ORDER == false && time_step_number == 1)
    {
      update_time_integrator(ORDER_TIME_INTEGRATOR-1);

      // calculate rhs_convection using the analytical solution at time t_nm and t_nm2
      time-=time_step;
      rhs_convection(solution_nm,rhs_convection_nm);
      time-=time_step;
      rhs_convection(solution_nm2,rhs_convection_nm2);

      // reset time
      time = cur_time;
    }
    /*
    if(time_step_number < ORDER_TIME_INTEGRATOR+1)
      check_time_integrator(time_step_number);
    */

    //important because I am using this on element level without giving it as argument to element loop
    for (unsigned int d=0; d<dim; ++d)
    {
      solution_n[d].update_ghost_values();
#ifdef XWALL
      solution_n[d+dim+1].update_ghost_values();
#endif
    }
#ifdef CHANNEL
  /***************** STEP 0: xwall update ***********************************/
    {
      xwall.UpdateTauW(solution_n);
      xwallstatevec[1]=*xwall.ReturnTauW();
      xwallstatevec[0].update_ghost_values();
      xwallstatevec[1].update_ghost_values();
#ifdef XWALL
      if(variabletauw)
      {
        precompute_inverse_mass_matrix();
        //project vectors
        //solution_n, solution_nm, solution_nm2
        xwall_projection();
        //it should be cheaper an the quality is probably better if we recompute them instead of projecting them
        rhs_convection(solution_nm,rhs_convection_nm);
        rhs_convection(solution_nm2,rhs_convection_nm2);
        compute_vorticity(solution_n,vorticity_n);
        compute_vorticity(solution_nm,vorticity_nm);
        compute_vorticity(solution_nm2,vorticity_nm2);
      }
      else if(time_step_number == 1) //the shape functions don't change
        precompute_inverse_mass_matrix();
#endif
    }
#endif
  /**************************************************************************/

  /***************** STEP 1: convective (nonlinear) term ********************/
    Timer timer;
    timer.restart();
    rhs_convection(solution_n,rhs_convection_n);
    compute_rhs(f);
    for (unsigned int d=0; d<dim; ++d)
    {
      solution_np[d].equ(beta[0],rhs_convection_n[d]);
      solution_np[d].add(beta[1],rhs_convection_nm[d],beta[2],rhs_convection_nm2[d]); // Stokes problem: velocity_temp[d] = f[d];
      solution_np[d] += f[d];
#ifdef STOKES
      solution_np[d] = f[d];
#endif
      solution_np[d].sadd(time_step,alpha[0],solution_n[d]);
      solution_np[d].add(alpha[1],solution_nm[d],alpha[2],solution_nm2[d]);
      //xwall
#ifdef XWALL
      solution_np[d+dim+1].equ(beta[0],rhs_convection_n[d+dim]);
      solution_np[d+dim+1].add(beta[1],rhs_convection_nm[d+dim],beta[2],rhs_convection_nm2[d+dim]); // Stokes problem: velocity_temp[d] = f[d];
      solution_np[d+dim+1] += f[d+dim];
      solution_np[d+dim+1].sadd(time_step, alpha[0], solution_n[d+1+dim]);
      solution_np[d+dim+1].add(alpha[1],solution_nm[d+1+dim],
                               alpha[2],solution_nm2[d+1+dim]);
#endif
    }
    rhs_convection_nm2.swap(rhs_convection_nm);
    rhs_convection_nm.swap(rhs_convection_n);
    computing_times[0] += timer.wall_time();
  /*************************************************************************/

  /************ STEP 2: solve poisson equation for pressure ****************/
  timer.restart();

  rhs_pressure(solution_np,rhs_p);

  // extrapolate from the oldest solution to get a good initial estimate for the solver
  solution_np[dim].equ(beta[0], solution_n[dim]);
  solution_np[dim].add(beta[1], solution_nm[dim], beta[2], solution_nm2[dim]);

  unsigned int pres_niter = pressure_poisson_solver.solve(solution_np[dim], rhs_p);

  if(pure_dirichlet_bc)
  {
    shift_pressure(solution_np[dim]);
  }

  if(time_step_number%output_solver_info_every_timesteps == 0)
  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory =
      Utilities::MPI::min_max_avg (stats.VmRSS/1024., MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && time_step_number%output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Number of timesteps: " << time_step_number << std::endl;
      std::cout << "Solve Poisson equation for p: PCG iterations: " << std::setw(3) << pres_niter << "  Wall time: " << timer.wall_time() << std::endl;
      std::cout << "Memory stats [MB]: " << memory.min << " [p" << memory.min_index << "] "
                << memory.avg << " " << memory.max << " [p" << memory.max_index << "]"
                << std::endl;
    }
  }

  computing_times[1] += timer.wall_time();
  /*************************************************************************/

  /********************** STEP 3: projection *******************************/
  timer.restart();

#ifdef WEAK_PROJECTION
  // so-called weak projection: solve global system of equations enforcing
  // both divergence-free constraint and continuity of velocity
  rhs_projection(solution_np,rhs_proj);
  for (unsigned int d=0; d<dim;++d)
  {
    rhs_proj_block.block(d) = rhs_proj[d];
    solution_proj_block.block(d) = velocity_temp[d];
  }
  solution_proj_block.collect_sizes();
  rhs_proj_block.collect_sizes();
  calculate_stab_fac_weak_projection();

  ReductionControl solver_control (1e5, ABS_TOL_PROJECTION, REL_TOL_PROJECTION);
  SolverCG<parallel::distributed::BlockVector<value_type> > solver (solver_control);
  WeakProjectionMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> weak_projection_matrix;
  weak_projection_matrix.initialize(*this);
  PreconditionerInverseMassMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> preconditioner_proj(*this);
  try
  {
    solver.solve (weak_projection_matrix, solution_proj_block, rhs_proj_block, preconditioner_proj);//PreconditionIdentity());
  }
  catch (SolverControl::NoConvergence)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout<<"Weak projection solver failed to solve to given tolerance." << std::endl;
  }
  for(unsigned int d=0;d<dim;++d)
  {
    velocity_temp[d] = solution_proj_block.block(d);
  }
#else
  // local element-by-element projection
  apply_projection(solution_np,velocity_temp);
#endif
  
  computing_times[2] += timer.wall_time();
  /*************************************************************************/

  /************************ STEP 4: viscous term ***************************/
  timer.restart();

  rhs_viscous(velocity_temp,rhs_visc);

  // set maximum number of iterations, tolerance
  ReductionControl solver_control_velocity (1e5, ABS_TOL_VISCOUS, REL_TOL_VISCOUS);//1.e-5
//  SolverCG<parallel::distributed::BlockVector<double> > solver_velocity (solver_control_velocity);
#ifdef SKEWSYMMVISC
  SolverGMRES<parallel::distributed::BlockVector<double> > solver_velocity (solver_control_velocity);
#else
  SolverCG<parallel::distributed::BlockVector<double> > solver_velocity (solver_control_velocity);
#endif
  NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> ns_viscous_matrix;
  ns_viscous_matrix.initialize(*this);

  // extrapolate from the oldest solution to get a good initial estimate for the solver
  for (unsigned int i=0; i<dim;i++)
  {
    solution_temp_visc.block(i) = solution_n[i];
    solution_temp_visc.block(i).sadd(beta[0], beta[1], solution_nm[i]);
    solution_temp_visc.block(i).add(beta[2], solution_nm2[i]);
#ifdef XWALL
    solution_temp_visc.block(i+dim) = solution_n[i+dim+1];
    solution_temp_visc.block(i+dim).sadd(beta[0], beta[1], solution_nm[i+dim+1]);
    solution_temp_visc.block(i+dim).add(beta[2], solution_nm2[i+dim+1]);
#endif
  }
  solution_temp_visc.collect_sizes();
  rhs_visc.collect_sizes();

  PreconditionerInverseMassMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> preconditioner(*this);

  try
  {
    solver_velocity.solve (ns_viscous_matrix, solution_temp_visc, rhs_visc, preconditioner);//PreconditionIdentity());
  }
  catch (SolverControl::NoConvergence)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout<<"Viscous solver failed to solve to given tolerance." << std::endl;
  }

  for (unsigned int i=0; i<dim;i++)
  {
    solution_np[i] = solution_temp_visc.block(i);
#ifdef XWALL
    solution_np[i+dim+1] = solution_temp_visc.block(i+dim);
#endif
  }

  computing_times[3] += timer.wall_time();
  /*************************************************************************/

  timer.restart();

  // solution at t_n-1 -> solution at t_n-2, solution at t_n -> solution at t_n-1  and   solution at t_n+1 -> solution at t_n
  solution_nm2.swap(solution_nm);
  solution_nm.swap(solution_n);
  solution_n.swap(solution_np);

  vorticity_nm2.swap(vorticity_nm);
  vorticity_nm.swap(vorticity_n);

  compute_vorticity(solution_n,vorticity_n);
  computing_times[4] += timer.wall_time();

#ifdef FLOW_PAST_CYLINDER
  compute_lift_and_drag(time_step_number == 1);
  compute_pressure_difference(time_step_number == 1);
#endif

  if(START_WITH_LOW_ORDER  == true)
  {
    if(time_step_number == 1 && ORDER_TIME_INTEGRATOR >1)
      update_time_integrator(time_step_number);
    if(time_step_number == 2 && ORDER_TIME_INTEGRATOR >2)
      update_time_integrator(time_step_number);
  }

  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  update_time_integrator (unsigned int time_step_number)
  {
    if(time_step_number == 1)
    {
      //BDF2
      gamma0 = 3.0/2.0;
      alpha[0] = 2.0;
      alpha[1] = -0.5;
      alpha[2] = 0.0;
      beta[0] = 2.0;
      beta[1] = -1.0;
      beta[2] = 0.0;
    }
    if(time_step_number == 2)
    {
      //BDF3
      gamma0 = 11./6.;
      alpha[0] = 3.;
      alpha[1] = -1.5;
      alpha[2] = 1./3.;
      beta[0] = 3.0;
      beta[1] = -3.0;
      beta[2] = 1.0;
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  check_time_integrator(unsigned int time_step_number)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
    std::cout << "Time integrator constants: time step "<< time_step_number << std::endl
              <<"Gamma0: " << gamma0   << std::endl
              <<"Alpha0: " << alpha[0] << std::endl
              <<"Alpha1: " << alpha[1] << std::endl
              <<"Alpha2: " << alpha[2] << std::endl
              <<"Beta0: "  << beta[0]  << std::endl
              <<"Beta1: "  << beta[1]  << std::endl
              <<"Beta2: "  << beta[2]  << std::endl << std::endl;
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  analyse_computing_times()
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::string names [5] = {"Convection","Pressure","Projection","Viscous  ","Other   "};
  pcout << std::endl << "Computing times:    \t       min       avg       max    p_min  p_max" << std::endl;
  double total_avg_time = 0;
  for (unsigned int i=0; i<computing_times.size(); ++i)
    {
      Utilities::MPI::MinMaxAvg data =
        Utilities::MPI::min_max_avg (computing_times[i], MPI_COMM_WORLD);
      pcout << "Step " << i+1 <<  ": " << names[i] << "\t "
            << std::setprecision(4) << std::setw(9) << data.min << " "
            << std::setprecision(4) << std::setw(9) << data.avg << " "
            << std::setprecision(4) << std::setw(9) << data.max << "   "
            << std::setw(6) << data.min_index << " "
            << std::setw(6) << data.max_index << std::endl;
      total_avg_time += data.avg;
    }
  pcout  <<"Time in steps 1-" << computing_times.size() << ":\t           "
         << std::setprecision(4) << std::setw(9) << total_avg_time << std::endl;
  Utilities::MPI::MinMaxAvg data =
    Utilities::MPI::min_max_avg (total_time.wall_time(), MPI_COMM_WORLD);
  pcout  <<"Global time since setup: "
         << std::setprecision(4) << std::setw(9) << data.min << " "
         << std::setprecision(4) << std::setw(9) << data.avg << " "
         << std::setprecision(4) << std::setw(9) << data.max << "   "
         << std::setw(6) << data.min_index << " "
         << std::setw(6) << data.max_index << std::endl;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_penalty_parameter(double &factor) const
  {
    // triangular/tetrahedral elements: penalty parameter = stab_factor*(p+1)(p+d)/dim * surface/volume
//  factor = stab_factor_viscous * (fe_degree +1.0) * (fe_degree + dim) / dim;

    // quadrilateral/hexahedral elements: penalty parameter = stab_factor*(p+1)(p+1) * surface/volume
    factor = stab_factor_viscous * (fe_degree +1.0) * (fe_degree + 1.0);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_lift_and_drag(const bool clear_files)
  {
  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval_velocity(data,true,0,0);
  FEFaceEvaluation<dim,fe_degree_p,fe_degree+1,1,value_type> fe_eval_pressure(data,true,1,0);

  Tensor<1,dim,value_type> Force;
  for(unsigned int d=0;d<dim;++d)
    Force[d] = 0.0;

  for(unsigned int face=data.n_macro_inner_faces(); face<(data.n_macro_inner_faces()+data.n_macro_boundary_faces()); face++)
  {
    fe_eval_velocity.reinit (face);
    fe_eval_velocity.read_dof_values(solution_n,0);
    fe_eval_velocity.evaluate(false,true);

    fe_eval_pressure.reinit (face);
    fe_eval_pressure.read_dof_values(solution_n,dim);
    fe_eval_pressure.evaluate(true,false);

    if (data.get_boundary_indicator(face) == 2)
    {
      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<value_type> pressure = fe_eval_pressure.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
        Tensor<2,dim,VectorizedArray<value_type> > velocity_gradient = fe_eval_velocity.get_gradient(q);
        fe_eval_velocity.submit_value(pressure*normal -  make_vectorized_array<value_type>(viscosity)*
            (velocity_gradient+transpose(velocity_gradient))*normal,q);
      }
      Tensor<1,dim,VectorizedArray<value_type> > Force_local = fe_eval_velocity.integrate_value();

      // sum over all entries of VectorizedArray
      for (unsigned int d=0; d<dim;++d)
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          Force[d] += Force_local[d][n];
    }
  }
  Force = Utilities::MPI::sum(Force,MPI_COMM_WORLD);

  // compute lift and drag coefficients (c = (F/rho)/(1/2 U² D)
#ifdef FLOW_PAST_CYLINDER
  const double U = Um * (dim==2 ? 2./3. : 4./9.);
  const double H = 0.41;
  if(dim == 2)
    Force *= 2.0/pow(U,2.0)/D;
  else if(dim == 3)
    Force *= 2.0/pow(U,2.0)/D/H;
#endif

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {
    std::string filename_drag, filename_lift;
    filename_drag = "output/drag_refine" + Utilities::int_to_string(data.get_dof_handler(1).get_triangulation().n_levels()-1) + "_fedegree" + Utilities::int_to_string(fe_degree) + ".txt"; //filename_drag = "drag.txt";
    filename_lift = "output/lift_refine" + Utilities::int_to_string(data.get_dof_handler(1).get_triangulation().n_levels()-1) + "_fedegree" + Utilities::int_to_string(fe_degree) + ".txt"; //filename_lift = "lift.txt";

    std::ofstream f_drag,f_lift;
    if(clear_files)
    {
      f_drag.open(filename_drag.c_str(),std::ios::trunc);
      f_lift.open(filename_lift.c_str(),std::ios::trunc);
    }
    else
    {
      f_drag.open(filename_drag.c_str(),std::ios::app);
      f_lift.open(filename_lift.c_str(),std::ios::app);
    }
    f_drag<<std::scientific<<std::setprecision(6)<<time+time_step<<"\t"<<Force[0]<<std::endl;
    f_drag.close();
    f_lift<<std::scientific<<std::setprecision(6)<<time+time_step<<"\t"<<Force[1]<<std::endl;
    f_lift.close();
  }
  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  compute_eddy_viscosity(const std::vector<parallel::distributed::Vector<value_type> >     &src)
//  {
//
//    eddy_viscosity = 0;
//    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_eddy_viscosity,this, eddy_viscosity, src);
//
//    const double mean = eddy_viscosity.mean_value();
//    eddy_viscosity.update_ghost_values();
//    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
//      std::cout << "new viscosity:   " << mean << "/" << viscosity << std::endl;
//  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_compute_eddy_viscosity(const MatrixFree<dim,value_type>                  &data,
//                parallel::distributed::Vector<value_type>      &dst,
//                const std::vector<parallel::distributed::Vector<value_type> >  &src,
//                const std::pair<unsigned int,unsigned int>            &cell_range) const
//  {
//    const VectorizedArray<value_type> Cs = make_vectorized_array(CS);
//    VectorizedArray<value_type> hfac = make_vectorized_array(1.0/(double)fe_degree);
//
//  //Warning: eddy viscosity is only interpolated using the polynomial space
//
//  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> velocity_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
//  FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi(data,0,0);
//  FEEvaluation<dim,1,fe_degree+1,1,double> fe_wdist(data,2,0);
//  FEEvaluation<dim,1,fe_degree+1,1,double> fe_tauw(data,2,0);
//  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);
//  const unsigned int dofs_per_cell = phi.dofs_per_cell;
//  AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
//
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//    phi.reinit(cell);
//    {
//      VectorizedArray<value_type> volume = make_vectorized_array(0.);
//      {
//        AlignedVector<VectorizedArray<value_type> > JxW_values;
//        JxW_values.resize(phi.n_q_points);
//        phi.fill_JxW_values(JxW_values);
//        for (unsigned int q=0; q<phi.n_q_points; ++q)
//          volume += JxW_values[q];
//      }
//      velocity_xwall.reinit(cell);
//      velocity_xwall.read_dof_values(src,0,src,dim+1);
//      velocity_xwall.evaluate (false,true,false);
//      fe_wdist.reinit(cell);
//      fe_wdist.read_dof_values(xwallstatevec[0]);
//      fe_wdist.evaluate(true,false,false);
//      fe_tauw.reinit(cell);
//      fe_tauw.read_dof_values(xwallstatevec[1]);
//      fe_tauw.evaluate(true,false,false);
//      for (unsigned int q=0; q<phi.n_q_points; ++q)
//      {
//        Tensor<2,dim,VectorizedArray<value_type> > s = velocity_xwall.get_gradient(q);
//
//        VectorizedArray<value_type> snorm = make_vectorized_array(0.);
//        for (unsigned int i = 0; i<dim ; i++)
//          for (unsigned int j = 0; j<dim ; j++)
//            snorm += make_vectorized_array(0.5)*(s[i][j]+s[j][i])*(s[i][j]+s[j][i]);
//        //simple wall correction
//        VectorizedArray<value_type> fmu = (1.-std::exp(-fe_wdist.get_value(q)/viscosity*std::sqrt(fe_tauw.get_value(q))/25.));
//        VectorizedArray<value_type> lm = Cs*std::pow(volume,1./3.)*hfac*fmu;
//        phi.submit_value (make_vectorized_array(viscosity) + std::pow(lm,2.)*std::sqrt(make_vectorized_array(2.)*snorm), q);
//      }
//      phi.integrate (true,false);
//
//      inverse.fill_inverse_JxW_values(coefficients);
//      inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());
//
//      phi.set_dof_values(dst);
//    }
//  }
//
//  }
  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_pressure_difference(const bool clear_files)
  {
  double pressure_1 = 0.0, pressure_2 = 0.0;
  unsigned int counter_1 = 0, counter_2 = 0;

  Point<dim> point_1, point_2;
  if(dim == 2)
  {
    Point<dim> point_1_2D(0.45,0.2), point_2_2D(0.55,0.2);
    point_1 = point_1_2D;
    point_2 = point_2_2D;
  }
  else if(dim == 3)
  {
    Point<dim> point_1_3D(0.45,0.2,0.205), point_2_3D(0.55,0.2,0.205);
    point_1 = point_1_3D;
    point_2 = point_2_3D;
  }

  // serial computation
//  Vector<double> value_1(1), value_2(1);
//  VectorTools::point_value(mapping,data.get_dof_handler(1),solution_n[dim],point_1,value_1);
//  pressure_1 = value_1(0);
//  VectorTools::point_value(mapping,data.get_dof_handler(1),solution_n[dim],point_2,value_2);
//  pressure_2 = value_2(0);

  // parallel computation
  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
  cell_point_1 = GridTools::find_active_cell_around_point (mapping,data.get_dof_handler(1), point_1);
  if(cell_point_1.first->is_locally_owned())
  {
    counter_1 = 1;
    //std::cout<< "Point 1 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;

    Vector<double> value(1);
    my_point_value(mapping,data.get_dof_handler(1),solution_n[dim],cell_point_1,value);
    pressure_1 = value(0);
  }
  counter_1 = Utilities::MPI::sum(counter_1,MPI_COMM_WORLD);
  pressure_1 = Utilities::MPI::sum(pressure_1,MPI_COMM_WORLD);
  pressure_1 = pressure_1/counter_1;

  const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
  cell_point_2 = GridTools::find_active_cell_around_point (mapping,data.get_dof_handler(1), point_2);
  if(cell_point_2.first->is_locally_owned())
  {
    counter_2 = 1;
    //std::cout<< "Point 2 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;

    Vector<double> value(1);
    my_point_value(mapping,data.get_dof_handler(1),solution_n[dim],cell_point_2,value);
    pressure_2 = value(0);
  }
  counter_2 = Utilities::MPI::sum(counter_2,MPI_COMM_WORLD);
  pressure_2 = Utilities::MPI::sum(pressure_2,MPI_COMM_WORLD);
  pressure_2 = pressure_2/counter_2;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {
    std::string filename = "output/pressure_difference_refine" + Utilities::int_to_string(data.get_dof_handler(1).get_triangulation().n_levels()-1) + "_fedegree" + Utilities::int_to_string(fe_degree) + ".txt"; // filename = "pressure_difference.txt";

    std::ofstream f;
    if(clear_files)
    {
      f.open(filename.c_str(),std::ios::trunc);
    }
    else
    {
      f.open(filename.c_str(),std::ios::app);
    }
    f << std::scientific << std::setprecision(6) << time+time_step << "\t" << pressure_1-pressure_2 << std::endl;
    f.close();
  }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  my_point_value(const Mapping<dim> &mapping,
              const DoFHandler<dim> &dof_handler,
              const parallel::distributed::Vector<double> &solution,
              const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > &cell_point,
              Vector<double> &value)
  {
  const FiniteElement<dim> &fe = dof_handler.get_fe();
  Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,ExcInternalError());

  const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

  FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
  fe_values.reinit(cell_point.first);

  // then use this to get at the values of
  // the given fe_function at this point
  std::vector<Vector<double> > u_value(1, Vector<double> (fe.n_components()));
  fe_values.get_function_values(solution, u_value);
  value = u_value[0];
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_diagonal_viscous(std::vector<parallel::distributed::Vector<value_type> > &diagonal,
  unsigned int level) const
  {

    std::vector<parallel::distributed::Vector<double> >  src_tmp;
    //not implemented with symmetric formulation
    Assert(false,ExcInternalError());
    data[level].loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_diagonal_viscous,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_diagonal_viscous_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_diagonal_viscous_boundary_face,
              this, diagonal, src_tmp);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_diagonal_viscous (const MatrixFree<dim,value_type>        &data,
               std::vector<parallel::distributed::Vector<double> >    &dst,
               const std::vector<parallel::distributed::Vector<double> >  &src,
               const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
#ifdef XWALL
    Assert(false,ExcInternalError());
#endif
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

   for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
   {
     fe_eval_xwall.reinit (cell);
     fe_eval_xwall.evaluate_eddy_viscosity(solution_n,cell);

    VectorizedArray<value_type> local_diagonal_vector[fe_eval_xwall.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
      fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));
      fe_eval_xwall.evaluate (true,true,false);
      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        fe_eval_xwall.submit_value (gamma0/time_step*fe_eval_xwall.get_value(q), q);
        fe_eval_xwall.submit_gradient (make_vectorized_array<value_type>(fe_eval_xwall.eddyvisc[q])*fe_eval_xwall.get_gradient(q), q);
      }
      fe_eval_xwall.integrate (true,true);
      local_diagonal_vector[j] = fe_eval_xwall.read_cellwise_dof_value(j);
    }
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
      fe_eval_xwall.write_cellwise_dof_value(j,local_diagonal_vector[j]);
    fe_eval_xwall.distribute_local_to_global (dst.at(0),dst.at(1));
   }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_diagonal_viscous_face (const MatrixFree<dim,value_type>       &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                   const std::pair<unsigned int,unsigned int>  &face_range) const
  {
#ifdef XWALL
    Assert(false,ExcInternalError());
#endif
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,1,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,2);
#endif

    const unsigned int level = data.get_cell_iterator(0,0)->level();

     for(unsigned int face=face_range.first; face<face_range.second; face++)
     {
       fe_eval_xwall.reinit (face);
       fe_eval_xwall_neighbor.reinit (face);
       fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));
       fe_eval_xwall_neighbor.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall_neighbor.read_cell_data(element_volume));
       double factor = 1.;
       calculate_penalty_parameter(factor);
       //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = std::max(fe_eval_xwall.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()),fe_eval_xwall_neighbor.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter())) * (value_type)factor;

       // element-
       VectorizedArray<value_type> local_diagonal_vector[fe_eval_xwall.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
      for (unsigned int i=0; i<fe_eval_xwall_neighbor.dofs_per_cell; ++i)
        fe_eval_xwall_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));

      fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));

      fe_eval_xwall.evaluate(true,true);
      fe_eval_xwall_neighbor.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        VectorizedArray<value_type> uM = fe_eval_xwall.get_value(q);
        VectorizedArray<value_type> uP = fe_eval_xwall_neighbor.get_value(q);

        VectorizedArray<value_type> jump_value = uM - uP;
        VectorizedArray<value_type> average_gradient =
            ( fe_eval_xwall.get_normal_gradient(q,true) + fe_eval_xwall_neighbor.get_normal_gradient(q,true) ) * make_vectorized_array<value_type>(0.5);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval_xwall.submit_normal_gradient(-0.5*fe_eval_xwall.eddyvisc[q]*jump_value,q);
        fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);
      }
      fe_eval_xwall.integrate(true,true);
      local_diagonal_vector[j] = fe_eval_xwall.read_cellwise_dof_value(j);
       }
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
      fe_eval_xwall.write_cellwise_dof_value(j, local_diagonal_vector[j]);
    fe_eval_xwall.distribute_local_to_global(dst.at(0),dst.at(1));

       // neighbor (element+)
    VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_xwall_neighbor.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_xwall_neighbor.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
      for (unsigned int i=0; i<fe_eval_xwall_neighbor.dofs_per_cell; ++i)
        fe_eval_xwall_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));

      fe_eval_xwall_neighbor.write_cellwise_dof_value(j,make_vectorized_array(1.));

      fe_eval_xwall.evaluate(true,true);
      fe_eval_xwall_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
        {
          VectorizedArray<value_type> uM = fe_eval_xwall.get_value(q);
          VectorizedArray<value_type> uP = fe_eval_xwall_neighbor.get_value(q);

          VectorizedArray<value_type> jump_value = uM - uP;
          VectorizedArray<value_type> average_gradient =
              ( fe_eval_xwall.get_normal_gradient(q,true) + fe_eval_xwall_neighbor.get_normal_gradient(q,true) ) * make_vectorized_array<value_type>(0.5);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval_xwall_neighbor.submit_normal_gradient(-0.5*fe_eval_xwall_neighbor.eddyvisc[q]*jump_value,q);
          fe_eval_xwall_neighbor.submit_value(fe_eval_xwall_neighbor.eddyvisc[q]*average_gradient,q);
        }
      fe_eval_xwall_neighbor.integrate(true,true);
      local_diagonal_vector_neighbor[j] = fe_eval_xwall_neighbor.read_cellwise_dof_value(j);
    }
    for (unsigned int j=0; j<fe_eval_xwall_neighbor.dofs_per_cell; ++j)
      fe_eval_xwall_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);
    fe_eval_xwall_neighbor.distribute_local_to_global(dst.at(0),dst.at(1));
     }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_diagonal_viscous_boundary_face (const MatrixFree<dim,value_type>       &data,
      std::vector<parallel::distributed::Vector<double> >    &dst,
      const std::vector<parallel::distributed::Vector<double> >  &src,
                       const std::pair<unsigned int,unsigned int>  &face_range) const
  {
#ifdef XWALL
    Assert(false,ExcInternalError());
#endif
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,1,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
#endif

    const unsigned int level = data.get_cell_iterator(0,0)->level();

     for(unsigned int face=face_range.first; face<face_range.second; face++)
     {
       fe_eval_xwall.reinit (face);
       fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));
       double factor = 1.;
       calculate_penalty_parameter(factor);
       //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = fe_eval_xwall.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()) * (value_type)factor;

       VectorizedArray<value_type> local_diagonal_vector[fe_eval_xwall.tensor_dofs_per_cell];
       for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
       {
         for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
      {
           fe_eval_xwall.write_cellwise_dof_value(i, make_vectorized_array(0.));
      }
         fe_eval_xwall.write_cellwise_dof_value(j, make_vectorized_array(1.));
      fe_eval_xwall.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          VectorizedArray<value_type> uM = fe_eval_xwall.get_value(q);
          VectorizedArray<value_type> uP = -uM;

          VectorizedArray<value_type> jump_value = uM - uP;
          VectorizedArray<value_type> average_gradient = fe_eval_xwall.get_normal_gradient(q,true);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval_xwall.submit_normal_gradient(-0.5*fe_eval_xwall.eddyvisc[q]*jump_value,q);
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);
        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
          fe_eval_xwall.submit_normal_gradient(-0.5*fe_eval_xwall.eddyvisc[q]*jump_value,q);
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);
        }
      }
      fe_eval_xwall.integrate(true,true);
      local_diagonal_vector[j] = fe_eval_xwall.read_cellwise_dof_value(j);
       }
    for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
      fe_eval_xwall.write_cellwise_dof_value(j, local_diagonal_vector[j]);
    fe_eval_xwall.distribute_local_to_global(dst.at(0),dst.at(1));
     }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> >   &src,
            std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<dim;++d)
  {
    dst[d] = 0;
#ifdef XWALL
    dst[d+dim] = 0;
#endif
  }

  // data.loop
  data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection_boundary_face,
            this, dst, src);

  data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
                             this, dst, dst);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst)
  {
  for(unsigned int d=0;d<dim;++d)
    dst[d] = 0;

  // data.loop
  data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_rhs,this, dst, dst);
  // data.cell_loop
  data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,this, dst, dst);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_viscous (const parallel::distributed::BlockVector<value_type>   &src,
              parallel::distributed::BlockVector<value_type>      &dst) const
  {
    for(unsigned int d=0;d<dim;++d)
    {
      dst.block(d)=0;
#ifdef XWALL
      dst.block(d+dim)=0;
#endif
    }
  data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_viscous_boundary_face,
            this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >   &src,
            parallel::distributed::BlockVector<value_type>      &dst)
  {
  for(unsigned int d=0;d<dim;++d)
    dst.block(d) = 0;
#ifdef XWALL
  for(unsigned int d=0;d<dim;++d)
    dst.block(d+dim) = 0;
#endif

  data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous_boundary_face,
            this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection (const MatrixFree<dim,value_type>              &data,
            std::vector<parallel::distributed::Vector<double> >      &dst,
            const std::vector<parallel::distributed::Vector<double> >  &src,
            const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
  // inexact integration  (data,0,0) : second argument: which dof-handler, third argument: which quadrature
//  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,2);
#endif
  // exact integration of convective term
//  FEEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> velocity (data,0,2);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_xwall.reinit(cell);
  //    velocity.reinit (cell);
      fe_eval_xwall.read_dof_values(src,0, src, dim+1);
      fe_eval_xwall.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_xwall.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > F
          = outer_product(u,u);
        fe_eval_xwall.submit_gradient (F, q);

      }
      fe_eval_xwall.integrate (false,true);
      fe_eval_xwall.distribute_local_to_global (dst,0, dst, dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection_face (const MatrixFree<dim,value_type>               &data,
              std::vector<parallel::distributed::Vector<double> >      &dst,
              const std::vector<parallel::distributed::Vector<double> >  &src,
              const std::pair<unsigned int,unsigned int>          &face_range) const
  {
  // inexact integration
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval_neighbor(data,false,0,0);

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,2);
#endif
  // exact integration
//  FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval(data,true,0,2);
//  FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_neighbor(data,false,0,2);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {

    fe_eval_xwall.reinit(face);
    fe_eval_xwall_neighbor.reinit (face);

    fe_eval_xwall.read_dof_values(src, 0, src, dim+1);
//    fe_eval.read_dof_values(src,0);
    fe_eval_xwall.evaluate(true, false);
//    fe_eval.evaluate(true,false);
    fe_eval_xwall_neighbor.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall_neighbor.evaluate(true,false);

    for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_xwall_neighbor.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);

      const VectorizedArray<value_type> uM_n = uM*normal;
      const VectorizedArray<value_type> uP_n = uP*normal;

      // calculation of lambda according to Shahbazi et al., i.e.
      // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
      // where the maximum eigenvalue of the flux Jacobian is the
      // maximum eigenvalue of (u^T * normal) * I + u * normal^T, which is
      // abs(2*u^T*normal) (this can be verified by rank-1 matrix algebra)
      const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

      Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
      Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
      Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

      fe_eval_xwall.submit_value(-lf_flux,q);
      fe_eval_xwall_neighbor.submit_value(lf_flux,q);
    }
    fe_eval_xwall.integrate(true,false);
    fe_eval_xwall.distribute_local_to_global(dst,0, dst, dim);
    fe_eval_xwall_neighbor.integrate(true,false);
    fe_eval_xwall_neighbor.distribute_local_to_global(dst,0,dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection_boundary_face (const MatrixFree<dim,value_type>             &data,
                       std::vector<parallel::distributed::Vector<double> >    &dst,
                       const std::vector<parallel::distributed::Vector<double> >  &src,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
  {
  // inexact integration
//    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
#endif

    for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_xwall.reinit (face);
    fe_eval_xwall.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall.evaluate(true,false);

    for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
    {
      if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
      {
        // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);

        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
        Tensor<1,dim,VectorizedArray<value_type> > g_n;
        for(unsigned int d=0;d<dim;++d)
        {
          AnalyticalSolution<dim> dirichlet_boundary(d,time);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
            array[n] = dirichlet_boundary.value(q_point);
          }
          g_n[d].load(&array[0]);
        }

        Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_n;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);
        const VectorizedArray<value_type> uM_n = uM*normal;
        const VectorizedArray<value_type> uP_n = uP*normal;

        // calculation of lambda according to Shahbazi et al., i.e.
        // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
        // where the maximum eigenvalue of the flux Jacobian is the
        // maximum eigenvalue of (u^T * normal) * I + u * normal^T, which is
        // abs(2*u^T*normal) (this can be verified by rank-1 matrix algebra)
        const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval_xwall.submit_value(-lf_flux,q);
      }
      else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
      {
        // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);
        const VectorizedArray<value_type> uM_n = uM*normal;
        const VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);

        Tensor<1,dim,VectorizedArray<value_type> > jump_value;
        for(unsigned d=0;d<dim;++d)
          jump_value[d] = 0.0;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = uM*uM_n;
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval_xwall.submit_value(-lf_flux,q);
      }
    }

    fe_eval_xwall.integrate(true,false);
    fe_eval_xwall.distribute_local_to_global(dst,0, dst, dim);
  }
  }

//  template <int dim, int model>
//  class Evaluator
//  {
//    void evaluate()
//    if (model == 0)
//      ...
//    else
//      ...
//
//  };
//
//  template <int dim>
//  class Evaluator<dim,0>
//  {
//    void evaluate()
//    {
//      ...;
//    }
//  }
//
//  template <int dim>
//  class Evaluator<dim,0>
//  {
//    void evaluate()
//    {
//      ...;
//    }
//  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_rhs (const MatrixFree<dim,value_type>             &data,
          std::vector<parallel::distributed::Vector<double> >     &dst,
          const std::vector<parallel::distributed::Vector<double> > &,
          const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_xwall.reinit (cell);

      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
        Tensor<1,dim,VectorizedArray<value_type> > rhs;
        for(unsigned int d=0;d<dim;++d)
        {
          RHS<dim> f(d,time+time_step);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
            array[n] = f.value(q_point);
          }
          rhs[d].load(&array[0]);
        }
        fe_eval_xwall.submit_value (rhs, q);
      }
      fe_eval_xwall.integrate (true,false);
      fe_eval_xwall.distribute_local_to_global(dst,0, dst, dim);
    }
  }


  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_viscous (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::BlockVector<double>      &dst,
            const parallel::distributed::BlockVector<double>  &src,
            const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_xwall.reinit(cell);
    fe_eval_xwall.evaluate_eddy_viscosity(solution_n,cell);
    fe_eval_xwall.read_dof_values(src,0,src,dim);
    fe_eval_xwall.evaluate (true,true);


    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      fe_eval_xwall.submit_value (gamma0/time_step * fe_eval_xwall.get_value(q), q);
      fe_eval_xwall.submit_gradient (fe_eval_xwall.eddyvisc[q]*fe_eval_xwall.get_symmetric_gradient(q), q);
    }
    fe_eval_xwall.integrate (true,true);
    fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_viscous_face (const MatrixFree<dim,value_type>       &data,
                parallel::distributed::BlockVector<double>      &dst,
                const parallel::distributed::BlockVector<double>  &src,
                const std::pair<unsigned int,unsigned int>  &face_range) const
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,0);
#endif

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall_neighbor.reinit (face);
      fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));
      fe_eval_xwall_neighbor.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall_neighbor.read_cell_data(element_volume));

      fe_eval_xwall.read_dof_values(src,0,src,dim);
      fe_eval_xwall.evaluate(true,true);
      fe_eval_xwall_neighbor.read_dof_values(src,0,src,dim);
      fe_eval_xwall_neighbor.evaluate(true,true);

      double factor = 1.;
      calculate_penalty_parameter(factor);
      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = std::max(fe_eval_xwall.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()),fe_eval_xwall_neighbor.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter())) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {

        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_xwall_neighbor.get_value(q);
        VectorizedArray<value_type> average_viscosity = 0.5*(fe_eval_xwall.eddyvisc[q] + fe_eval_xwall_neighbor.eddyvisc[q]);
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
        Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor =
            ( fe_eval_xwall.get_symmetric_gradient(q) + fe_eval_xwall_neighbor.get_symmetric_gradient(q)) * make_vectorized_array<value_type>(0.5);
        Tensor<2,dim,VectorizedArray<value_type> > jump_tensor =
            outer_product(jump_value,fe_eval_xwall.get_normal_vector(q));


        //we do not want to symmetrize the penalty part
        average_gradient_tensor = average_gradient_tensor*average_viscosity - std::max(fe_eval_xwall.eddyvisc[q], fe_eval_xwall_neighbor.eddyvisc[q])*jump_tensor * sigmaF;

        Tensor<1,dim,VectorizedArray<value_type> > average_gradient;
        for (unsigned int comp=0; comp<dim; comp++)
          {
          average_gradient[comp] = average_gradient_tensor[comp][0] *
              fe_eval_xwall.get_normal_vector(q)[0];
            for (unsigned int d=1; d<dim; ++d)
              average_gradient[comp] += average_gradient_tensor[comp][d] *
                fe_eval_xwall.get_normal_vector(q)[d];
          }
#ifdef SKEWSYMMVISC
      fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.make_symmetric(average_viscosity*jump_tensor),q);
      fe_eval_xwall_neighbor.submit_gradient(0.5*fe_eval_xwall.make_symmetric(average_viscosity*jump_tensor),q);
#else
      fe_eval_xwall.submit_gradient(-0.5*fe_eval_xwall.make_symmetric(average_viscosity*jump_tensor),q);
      fe_eval_xwall_neighbor.submit_gradient(-0.5*fe_eval_xwall.make_symmetric(average_viscosity*jump_tensor),q);
#endif
        fe_eval_xwall.submit_value(-average_gradient,q);
        fe_eval_xwall_neighbor.submit_value(average_gradient,q);

      }
      fe_eval_xwall.integrate(true,true);
      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
      fe_eval_xwall_neighbor.integrate(true,true);
      fe_eval_xwall_neighbor.distribute_local_to_global(dst,0,dst,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_viscous_boundary_face (const MatrixFree<dim,value_type>       &data,
                    parallel::distributed::BlockVector<double>      &dst,
                    const parallel::distributed::BlockVector<double>  &src,
                    const std::pair<unsigned int,unsigned int>  &face_range) const
  {
//    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
#endif
    const unsigned int level = data.get_cell_iterator(0,0)->level();

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));

      fe_eval_xwall.read_dof_values(src,0,src,dim);
      fe_eval_xwall.evaluate(true,true);

      double factor = 1.;
      calculate_penalty_parameter(factor);
      //VectorizedArray<value_type> sigmaF = std::abs(fe_eval_xwall.get_normal_volume_fraction()) * (value_type)factor;
      VectorizedArray<value_type> sigmaF = fe_eval_xwall.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = -uM;
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
          Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor =
              fe_eval_xwall.get_symmetric_gradient(q);
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(jump_value,fe_eval_xwall.get_normal_vector(q));


          //we do not want to symmetrize the penalty part
          average_gradient_tensor = average_gradient_tensor - jump_tensor * sigmaF;

          Tensor<1,dim,VectorizedArray<value_type> > average_gradient;
          for (unsigned int comp=0; comp<dim; comp++)
            {
            average_gradient[comp] = average_gradient_tensor[comp][0] *
                fe_eval_xwall.get_normal_vector(q)[0];
              for (unsigned int d=1; d<dim; ++d)
                average_gradient[comp] += average_gradient_tensor[comp][d] *
                  fe_eval_xwall.get_normal_vector(q)[d];
            }
#ifdef SKEWSYMMVISC
          fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.make_symmetric(fe_eval_xwall.eddyvisc[q]*jump_tensor),q);
#else
          fe_eval_xwall.submit_gradient(-0.5*fe_eval_xwall.make_symmetric(fe_eval_xwall.eddyvisc[q]*jump_tensor),q);
#endif
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);

        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          Tensor<1,dim,VectorizedArray<value_type> > jump_value;
          Tensor<1,dim,VectorizedArray<value_type> > average_gradient;// = make_vectorized_array<value_type>(0.0);
          for(unsigned int i=0;i<dim;i++)
          {
            average_gradient[i] = make_vectorized_array(0.);
            jump_value[i] = make_vectorized_array(0.);
          }
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(jump_value,fe_eval_xwall.get_normal_vector(q));

#ifdef SKEWSYMMVISC
          fe_eval_xwall.submit_gradient(0.5*fe_eval_xwall.make_symmetric(fe_eval_xwall.eddyvisc[q]*jump_tensor),q);
#else
          fe_eval_xwall.submit_gradient(-0.5*fe_eval_xwall.make_symmetric(fe_eval_xwall.eddyvisc[q]*jump_tensor),q);
#endif
          fe_eval_xwall.submit_value(-fe_eval_xwall.eddyvisc[q]*average_gradient,q);

        }
      }
      fe_eval_xwall.integrate(true,true);
      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous (const MatrixFree<dim,value_type>                &data,
              parallel::distributed::BlockVector<double>      &dst,
              const std::vector<parallel::distributed::Vector<double> >  &src,
              const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
      // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_xwall.reinit (cell);
      fe_eval_xwall.read_dof_values(src,0,src,dim);
      fe_eval_xwall.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_xwall.get_value(q);
        fe_eval_xwall.submit_value (make_vectorized_array<value_type>(1.0/time_step)*u, q);
      }
      fe_eval_xwall.integrate (true,false);
      fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous_face (const MatrixFree<dim,value_type>                 &,
                parallel::distributed::BlockVector<double>      &,
                const std::vector<parallel::distributed::Vector<double> >  &,
                const std::pair<unsigned int,unsigned int>          &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous_boundary_face (const MatrixFree<dim,value_type>             &data,
                         parallel::distributed::BlockVector<double>    &dst,
                         const std::vector<parallel::distributed::Vector<double> >  &,
                         const std::pair<unsigned int,unsigned int>          &face_range) const
  {

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
#endif

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall.read_cell_data(element_volume));

      double factor = 1.;
      calculate_penalty_parameter(factor);

      VectorizedArray<value_type> sigmaF = fe_eval_xwall.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > g_np;
          for(unsigned int d=0;d<dim;++d)
          {
            AnalyticalSolution<dim> dirichlet_boundary(d,time+time_step);
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array[n] = dirichlet_boundary.value(q_point);
            }
            g_np[d].load(&array[0]);
          }

          g_np *= fe_eval_xwall.eddyvisc[q];
          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(g_np,fe_eval_xwall.get_normal_vector(q));
#ifdef SKEWSYMMVISC
          fe_eval_xwall.submit_gradient(fe_eval_xwall.make_symmetric(jump_tensor),q);
#else
          fe_eval_xwall.submit_gradient(-fe_eval_xwall.make_symmetric(jump_tensor),q);
#endif
          fe_eval_xwall.submit_value(2.0*sigmaF*g_np,q);

        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > h;
          for(unsigned int d=0;d<dim;++d)
          {
            NeumannBoundaryVelocity<dim> neumann_boundary(d,time+time_step);
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array[n] = neumann_boundary.value(q_point);
            }
            h[d].load(&array[0]);
          }
          Tensor<1,dim,VectorizedArray<value_type> > jump_value;
          for(unsigned d=0;d<dim;++d)
            jump_value[d] = 0.0;

          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(jump_value,fe_eval_xwall.get_normal_vector(q));
#ifdef SKEWSYMMVISC
          fe_eval_xwall.submit_gradient(jump_tensor,q);
#else
          fe_eval_xwall.submit_gradient(-jump_tensor,q);
#endif
          fe_eval_xwall.submit_value(fe_eval_xwall.eddyvisc[q]*h,q);
        }
      }

      fe_eval_xwall.integrate(true,true);
      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  precompute_inverse_mass_matrix ()
  {
   std::vector<parallel::distributed::Vector<value_type> > dummy;
  data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_precompute_mass_matrix,
                   this, dummy, dummy);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  xwall_projection ()
  {

    //make sure that this is distributed properly
    (*xwall.ReturnTauWN()).update_ghost_values();

    std::vector<parallel::distributed::Vector<value_type> > tmp(2*dim);
    for (unsigned int i=0;i<dim;i++)
    {
      tmp[i]=solution_n[i];
      tmp[i+dim]=solution_n[i+dim+1];
    }
    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
                   this, solution_n, tmp);
    for (unsigned int i=0;i<dim;i++)
    {
      tmp[i]=solution_nm[i];
      tmp[i+dim]=solution_nm[i+dim+1];
    }
    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
                   this, solution_nm, tmp);
    for (unsigned int i=0;i<dim;i++)
    {
      tmp[i]=solution_nm2[i];
      tmp[i+dim]=solution_nm2[i+dim+1];
    }
    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
                   this, solution_nm2, tmp);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_precompute_mass_matrix (const MatrixFree<dim,value_type>        &data,
      std::vector<parallel::distributed::Vector<value_type> >    &,
      const std::vector<parallel::distributed::Vector<value_type> >  &,
               const std::pair<unsigned int,unsigned int>   &cell_range)
  {

    //initialize routine for non-enriched elements
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      std::vector<FullMatrix<value_type> > matrix;
      {
        FullMatrix<value_type> onematrix(fe_eval_xwall.tensor_dofs_per_cell);
        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
          matrix.push_back(onematrix);
      }
      for (unsigned int j=0; j<fe_eval_xwall.tensor_dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_xwall.evaluate (true,false,false);
        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
        {
  //        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
          fe_eval_xwall.submit_value (fe_eval_xwall.get_value(q), q);
        }
        fe_eval_xwall.integrate (true,false);

        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
            if(fe_eval_xwall.component_enriched(v))
              (matrix[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
            else//this is a non-enriched element
            {
              if(i<fe_eval_xwall.std_dofs_per_cell && j<fe_eval_xwall.std_dofs_per_cell)
                (matrix[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
              else if(i == j)//diagonal
                (matrix[v])(i,j) = 1.0;
            }
      }
//      for (unsigned int i=0; i<10; ++i)
//        std::cout << std::endl;
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        matrix[v].print(std::cout,14,8);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrix[v]).gauss_jordan();
      }
      matrices[cell].reinit(fe_eval_xwall.dofs_per_cell, fe_eval_xwall.dofs_per_cell);
      //now apply vectors to inverse matrix
      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
        for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
        {
          VectorizedArray<value_type> value;
          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
            value[v] = (matrix[v])(i,j);
          matrices[cell](i,j) = value;
        }
    }
  }
  //


  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_project_xwall (const MatrixFree<dim,value_type>        &data,
      std::vector<parallel::distributed::Vector<value_type> >    &dst,
      const std::vector<parallel::distributed::Vector<value_type> >  &src,
               const std::pair<unsigned int,unsigned int>   &cell_range)
  {

  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall_n (data,xwallstatevec[0],*xwall.ReturnTauWN(),0,3);
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall_n.reinit (cell);
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      //now apply vectors to inverse matrix
      for (unsigned int idim = 0; idim < dim; ++idim)
      {
        fe_eval_xwall_n.read_dof_values(src.at(idim),src.at(idim+dim));
        fe_eval_xwall_n.evaluate(true,false);
        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; q++)
          fe_eval_xwall.submit_value(fe_eval_xwall_n.get_value(q),q);
        fe_eval_xwall.integrate(true,false);
        AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
            vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
        fe_eval_xwall.set_dof_values (dst.at(idim),dst.at(idim+dim+1));
      }
    }
  }
  //


  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_inverse_mass_matrix (const parallel::distributed::BlockVector<value_type>  &src,
      parallel::distributed::BlockVector<value_type>      &dst) const
  {
    for (unsigned int i = 0; i<dim; i++)
    {
      dst.block(i)=0;
#ifdef XWALL
      dst.block(i+dim)=0;
#endif
    }

  data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
                   this, dst, src);

  for (unsigned int i = 0; i<dim; i++)
  {
    dst.block(i)*= time_step/gamma0;
#ifdef XWALL
    dst.block(i+dim)*= time_step/gamma0;
#endif
  }

  }
  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_mass_matrix (const MatrixFree<dim,value_type>        &data,
                parallel::distributed::BlockVector<value_type>    &dst,
                const parallel::distributed::BlockVector<value_type>  &src,
                const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
   InverseMassMatrixData<dim,fe_degree,value_type>& mass_data = mass_matrix_data->get();

#ifdef XWALL
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#endif

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
#ifdef XWALL
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      //now apply vectors to inverse matrix
      for (unsigned int idim = 0; idim < dim; ++idim)
      {
        fe_eval_xwall.read_dof_values(src.block(idim),src.block(idim+dim));
        AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
            vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
        fe_eval_xwall.set_dof_values (dst.block(idim),dst.block(idim+dim));
      }
    }
    else
#endif
    {
      mass_data.phi[0].reinit(cell);
      mass_data.phi[0].read_dof_values(src, 0);

      mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
      mass_data.inverse.apply(mass_data.coefficients, dim,
                              mass_data.phi[0].begin_dof_values(),
                              mass_data.phi[0].begin_dof_values());

      mass_data.phi[0].set_dof_values(dst,0);
    }
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_divergence (const MatrixFree<dim,value_type>        &data,
                parallel::distributed::Vector<value_type>    &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>   &cell_range) const
  {

    //initialize routine for non-enriched elements
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi(data,0,0);

//    VectorizedArray<value_type> coefficients[FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type>::tensor_dofs_per_cell]
    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);
#ifdef XWALL
    FEEvaluation<dim,fe_degree,n_q_points_1d_xwall,1,value_type> fe_eval (data,0,3);
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval (data,0,0);
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    {
      fe_eval_xwall.reinit(cell);
      fe_eval.reinit(cell);
      phi.reinit(cell);
      fe_eval_xwall.read_dof_values(src,0,src,dim);
      fe_eval_xwall.evaluate(false,true);

      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; q++)
        fe_eval.submit_value(fe_eval_xwall.get_divergence(q),q);
      fe_eval.integrate(true,false);
      for (unsigned int i=0; i<fe_eval.dofs_per_cell; i++)
        phi.begin_dof_values()[i] = fe_eval.begin_dof_values()[i];

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }
  //

  }
  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_mass_matrix (const MatrixFree<dim,value_type>        &data,
      std::vector<parallel::distributed::Vector<value_type> >    &dst,
      const std::vector<parallel::distributed::Vector<value_type> >  &src,
               const std::pair<unsigned int,unsigned int>   &cell_range) const
  {

    if(dst.size()>dim)
    {
    //initialize routine for non-enriched elements
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> phi(data,0,0);
    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, dim, value_type> inverse(phi);
#ifdef XWALL
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#endif


  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
#ifdef XWALL
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      //now apply vectors to inverse matrix
      for (unsigned int idim = 0; idim < dim; ++idim)
      {
        fe_eval_xwall.read_dof_values(src.at(idim),src.at(idim+dim));
        AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
            vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
        fe_eval_xwall.set_dof_values (dst.at(idim),dst.at(idim+dim));
      }
    }
    else
#endif
    {
      phi.reinit(cell);
      phi.read_dof_values(src,0);

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,dim,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }
  //
    }
    else
    {
      FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi (data,0,0);

      AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);
  #ifdef XWALL
     FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
#endif

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
#ifdef XWALL
      //first, check if we have an enriched element
      //if so, perform the routine for the enriched elements
      fe_eval_xwall.reinit (cell);
      if(fe_eval_xwall.enriched)
      {
        //now apply vectors to inverse matrix
          fe_eval_xwall.read_dof_values(src.at(0),src.at(1));
          AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
            for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
              vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
            fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
          fe_eval_xwall.set_dof_values (dst.at(0),dst.at(1));
      }
      else
  #endif
      {
        phi.reinit(cell);
        phi.read_dof_values(src.at(0));

        inverse.fill_inverse_JxW_values(coefficients);
        inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());

        phi.set_dof_values(dst.at(0));
      }
    }
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >   &src,
              std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<2*number_vorticity_components;++d)
    dst[d] = 0;
  // data.loop
  data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_vorticity,this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_grad_div_projection(const MatrixFree<dim,value_type>                  &data,
                std::vector<parallel::distributed::Vector<value_type> >      &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> velocity(data,xwallstatevec[0],xwallstatevec[1],0,3);
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> phi(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> velocity(data,xwallstatevec[0],xwallstatevec[1],0,0);
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> phi(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

  std::vector<LAPACKFullMatrix<value_type> > matrices(VectorizedArray<value_type>::n_array_elements);
  AlignedVector<VectorizedArray<value_type> > JxW_values(phi.n_q_points);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit(cell);
    velocity.reinit(cell);
    const unsigned int total_dofs_per_cell = phi.dofs_per_cell * dim;
    velocity.read_dof_values(solution_n,0,solution_n,dim);
    velocity.evaluate (true,false);
    VectorizedArray<value_type> volume;
    VectorizedArray<value_type> normmeanvel;
    {
      Tensor<1,dim,VectorizedArray<value_type> > meanvel;
      phi.fill_JxW_values(JxW_values);
      meanvel = JxW_values[0]*velocity.get_value(0);
      volume = JxW_values[0];
      for (unsigned int q=1; q<phi.n_q_points; ++q)
      {
        meanvel += JxW_values[q]*velocity.get_value(q);
        volume += JxW_values[q];
      }
      meanvel /=volume;
      normmeanvel = meanvel.norm();
    }
    velocity.reinit(cell);
    velocity.read_dof_values(src,0,src,dim);

    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      matrices[v].reinit(total_dofs_per_cell, total_dofs_per_cell);

    // compute grad-div parameter
    //use definition Ohlhanskii et al. (2009)
//#ifdef STOKES
    const VectorizedArray<value_type> tau = K*normmeanvel*std::pow(volume,1./(double)dim) + make_vectorized_array<value_type>(VISCOSITY*K);
//    const VectorizedArray<value_type> tau = make_vectorized_array<value_type>(VISCOSITY*K);
//#else
//    const VectorizedArray<value_type> tau =
//      K*normmeanvel*std::pow(volume,1./(double)dim);
//#endif

//    std::cout << "tau" << tau[0] << "  " << tau[1] << std::endl;
//    std::cout << "vel  " << normmeanvel[0] << "  " << normmeanvel[1] << std::endl;

    for (unsigned int j=0; j<total_dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<total_dofs_per_cell; ++i)
        phi.write_cellwise_dof_value(i,make_vectorized_array(0.));
      phi.write_cellwise_dof_value(j,make_vectorized_array(1.));

      phi.evaluate (true,true,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
        const VectorizedArray<value_type> tau_times_div = tau * phi.get_divergence(q);
        Tensor<2,dim,VectorizedArray<value_type> > test;
        for (unsigned int d=0; d<dim; ++d)
          test[d][d] = tau_times_div;
        phi.submit_gradient(test, q);
        phi.submit_value (phi.get_value(q), q);
      }
      phi.integrate (true,true);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        if(phi.component_enriched(v))
          for (unsigned int i=0; i<total_dofs_per_cell; ++i)
            (matrices[v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
        else//this is a non-enriched element
          {
            if(j<phi.std_dofs_per_cell*dim)
              for (unsigned int i=0; i<phi.std_dofs_per_cell*dim; ++i)
                (matrices[v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
            else //diagonal
              (matrices[v])(j,j) = 1.0;
          }
    }

//      for (unsigned int i=0; i<10; ++i)
//        std::cout << std::endl;
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        matrices[v].print(std::cout,14,8);

    //now apply vectors to inverse matrix
//    for (unsigned int q=0; q<phi.n_q_points; ++q)
//    {
//      velocity.submit_value (velocity.get_value(q), q);
//    }
//    velocity.integrate (true,false);

    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
    {
      (matrices[v]).compute_lu_factorization();
      Vector<value_type> vector_input(total_dofs_per_cell);
      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        vector_input(j)=(velocity.read_cellwise_dof_value(j))[v];

//        Vector<value_type> vector_result(total_dofs_per_cell);
      (matrices[v]).apply_lu_factorization(vector_input,false);
//        (matrices[v]).vmult(vector_result,vector_input);
      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        velocity.write_cellwise_dof_value(j,vector_input(j),v);
    }
    velocity.set_dof_values (dst,0,dst,dim);
  }
  }

  namespace internal
  {
    template <typename Number, typename Number2>
    bool all_smaller (const Number a, const Number2 b)
    {
      return a<b;
    }
    template <typename Number, typename Number2>
    bool all_smaller (const VectorizedArray<Number> a, const Number2 b)
    {
      for (unsigned int i=0; i<VectorizedArray<Number>::n_array_elements; ++i)
        if (a[i] >= b)
          return false;
      return true;
    }
    template <typename Number>
    void adjust_division_by_zero (Number &)
    {}
    template <typename Number>
    void adjust_division_by_zero (VectorizedArray<Number> &x)
    {
      for (unsigned int i=0; i<VectorizedArray<Number>::n_array_elements; ++i)
        if (x[i] < 1e-30)
          x[i] = 1;
    }
  }

  template<typename value_type>
  class SolverCGmod
  {
  public:
    SolverCGmod(const unsigned int unknowns,
                const double abs_tol=1.e-12,
                const double rel_tol=1.e-8,
                const unsigned int max_iter = 1e5);

    template <typename Matrix>
    void solve(const Matrix *matrix,  value_type *solution, const value_type *rhs);

  private:
    const double ABS_TOL;
    const double REL_TOL;
    const unsigned int MAX_ITER;
    AlignedVector<value_type> storage;
    value_type *p,*r,*v;
    const unsigned int M;
    value_type l2_norm(const value_type *vector);

    void vector_init(value_type *dst);
    void equ(value_type *dst, const value_type scalar, const value_type *in_vector);
    void equ(value_type *dst, const value_type scalar1, const value_type *in_vector1, const value_type scalar2, const value_type *in_vector2);

    void add(value_type *dst, const value_type scalar, const value_type *in_vector);
    value_type inner_product(const value_type *vector1, const value_type *vector2);
  };
  template<typename value_type>
  SolverCGmod<value_type>::SolverCGmod(const unsigned int unknowns,
                                       const double abs_tol,
                                       const double rel_tol,
                                       const unsigned int max_iter):
  ABS_TOL(abs_tol),
  REL_TOL(rel_tol),
  MAX_ITER(max_iter),
  M(unknowns)
  {
    storage.resize(3*M);
    p = storage.begin();
    r = storage.begin()+M;
    v = storage.begin()+2*M;
  }

  template<typename value_type>
  value_type SolverCGmod< value_type>::l2_norm(const value_type *vector)
  {
    return std::sqrt(inner_product(vector, vector));
  }

  template<typename value_type>
  void SolverCGmod< value_type>::vector_init(value_type *vector)
  {
    for(unsigned int i=0;i<M;++i)
      vector[i] = 0.0;
  }

  template<typename value_type>
  void SolverCGmod< value_type>::equ(value_type *dst, const value_type scalar, const value_type *in_vector)
  {
    for(unsigned int i=0;i<M;++i)
      dst[i] = scalar*in_vector[i];
  }

  template<typename value_type>
  void SolverCGmod< value_type>::equ(value_type *dst, const value_type scalar1, const value_type *in_vector1, const value_type scalar2, const value_type *in_vector2)
  {
    for(unsigned int i=0;i<M;++i)
      dst[i] = scalar1*in_vector1[i]+scalar2*in_vector2[i];
  }

  template<typename value_type>
  void SolverCGmod< value_type>::add(value_type *dst, const value_type scalar, const value_type *in_vector)
  {
    for(unsigned int i=0;i<M;++i)
      dst[i] += scalar*in_vector[i];
  }

  template<typename value_type>
  value_type SolverCGmod< value_type>::inner_product(const value_type *vector1, const value_type *vector2)
  {
    value_type result = value_type();
    for(unsigned int i=0;i<M;++i)
      result += vector1[i]*vector2[i];

    return result;
  }

  template<typename value_type>
  template<typename Matrix>
  void SolverCGmod<value_type>::solve(const Matrix *matrix,
                                      value_type *solution,
                                      const value_type *rhs)
  {
    value_type one;
    one = 1.0;

    // guess initial solution
    vector_init(solution);

    // apply matrix vector product: v = A*solution
    matrix->vmult(v,solution);

    // compute residual: r = rhs-A*solution
    equ(r,one,rhs,-one,v);
    value_type norm_r0 = l2_norm(r);

    // precondition
    matrix->precondition(p,r);
    // else
    //   equ(p,1.,r);

    // compute norm of residual
    value_type norm_r_abs = norm_r0;
    value_type norm_r_rel = one;
    value_type r_times_y = inner_product(p, r);

    unsigned int n_iter = 0;

    while(true)
    {
      // v = A*p
      (*matrix).vmult(v,p);

      value_type p_times_v = inner_product(p,v);
      internal::adjust_division_by_zero(p_times_v);
      internal::adjust_division_by_zero(r_times_y);

      // alpha = (r^T*y) / (p^T*v)
      value_type alpha = (r_times_y)/(p_times_v);

      // solution <- solution + alpha*p
      add(solution,alpha,p);

      // r <- r - alpha*v
      add(r,-alpha,v);

      // calculate residual norm
      norm_r_abs = l2_norm(r);
      norm_r_rel = norm_r_abs / norm_r0;

      // increment iteration counter
      ++n_iter;

      if (internal::all_smaller(norm_r_abs, ABS_TOL) ||
          internal::all_smaller(norm_r_rel, REL_TOL) || (n_iter > MAX_ITER))
        break;

      // precondition
      matrix->precondition(v,r);

      value_type r_times_y_new = inner_product(r,v);

      // beta = (v^T*r) / (p^T*v)
      value_type beta = r_times_y_new / r_times_y;

      // p <- r -beta*p
      equ(p,one,v,beta,p);

      r_times_y = r_times_y_new;
    }

    std::ostringstream message;
    for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; v++)
       message << " v: " << v << "  " << norm_r_abs[v] << " ";
    Assert(n_iter <= MAX_ITER,
           ExcMessage("No convergence of solver in " + Utilities::to_string(MAX_ITER)
                      + "iterations. Residual was " + message.str().c_str()));
  }

  template<int dim, int fe_degree>
  class MatrixProjectionStep
  {
  public:
    static const unsigned int dimension = dim;
    static const unsigned int degree = fe_degree;
    typedef FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> EvalType;

    MatrixProjectionStep(const MatrixFree<dim,double> &matrix_free,
                         const unsigned int fe_no,
                         const unsigned int quad_no)
      :
      fe_eval(matrix_free, fe_no, quad_no),
      inverse(fe_eval)
    {
      coefficients.resize(fe_eval.n_q_points);
    }

    void setup(const unsigned int cell,
               const VectorizedArray<double> tau_grad_div_stab)
    {
      this->tau = tau_grad_div_stab;
      fe_eval.reinit(cell);
      inverse.fill_inverse_JxW_values(coefficients);
    }

    void precondition(VectorizedArray<double> *dst,
                      const VectorizedArray<double> *src) const
    {
      inverse.apply(coefficients, dim, src, dst);
    }

    void vmult(VectorizedArray<double> *dst,
               VectorizedArray<double> *src) const
    {
      Assert(fe_eval.get_shape_info().element_type ==
             dealii::internal::MatrixFreeFunctions::tensor_symmetric ||
             fe_eval.get_shape_info().element_type ==
             dealii::internal::MatrixFreeFunctions::tensor_gausslobatto,
             ExcNotImplemented());

      // get internal evaluator in order to avoid copying data around
      dealii::internal::FEEvaluationImpl<dealii::internal::MatrixFreeFunctions::tensor_symmetric,
                                         dim, fe_degree, fe_degree+1, double> evaluator;
      VectorizedArray<double> *unit_values[dim], *unit_gradients[dim][dim],
        *unit_hessians[dim][dim*(dim+1)/2];
      for (unsigned int c=0; c<dim; ++c)
        {
          unit_values[c] = &fe_eval.begin_values()[c*fe_eval.n_q_points];
          for (unsigned int d=0; d<dim; ++d)
            unit_gradients[c][d] = &fe_eval.begin_gradients()[(c*dim+d)*fe_eval.n_q_points];
          for (unsigned int d=0; d<dim*(dim+1)/2; ++d)
            unit_hessians[c][d] = 0;
        }

      // compute matrix vector product on element
      for (unsigned int c=0; c<dim; ++c)
        evaluator.evaluate(fe_eval.get_shape_info(),
                           &src[c*fe_eval.dofs_per_cell], unit_values[c],
                           unit_gradients[c], unit_hessians[c],
                           true, true, false);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
        {
          VectorizedArray<double> tau_times_div = tau * fe_eval.get_divergence(q);
          fe_eval.submit_divergence(tau_times_div, q);
          fe_eval.submit_value (fe_eval.get_value(q), q);
        }
      for (unsigned int c=0; c<dim; ++c)
        evaluator.integrate(fe_eval.get_shape_info(),
                           &dst[c*fe_eval.dofs_per_cell], unit_values[c],
                           unit_gradients[c],
                           true, true);
    }

  private:
    mutable FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> fe_eval;
    VectorizedArray<double> tau;
    AlignedVector<VectorizedArray<double> > coefficients;
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,double> inverse;
  };



  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_precompute_grad_div_projection(const MatrixFree<dim,value_type>                  &data,
                std::vector<parallel::distributed::Vector<value_type> >      &,
                const std::vector<parallel::distributed::Vector<value_type> >  &,
                const std::pair<unsigned int,unsigned int>            &cell_range)
  {
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> phi(data,xwallstatevec[0],xwallstatevec[1],0,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    //first is div-div matrix, second is mass matrix
    div_matrices[cell].resize(2);
    //div-div matrix
    phi.reinit(cell);
    const unsigned int total_dofs_per_cell = phi.dofs_per_cell * dim;
    div_matrices[cell][0].resize(data.n_components_filled(cell));
    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      div_matrices[cell][0][v].reinit(total_dofs_per_cell, total_dofs_per_cell);

    for (unsigned int j=0; j<total_dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<total_dofs_per_cell; ++i)
        phi.write_cellwise_dof_value(i,make_vectorized_array(0.));
      phi.write_cellwise_dof_value(j,make_vectorized_array(1.));

      phi.evaluate (false,true,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
        const VectorizedArray<value_type> div = phi.get_divergence(q);
        Tensor<2,dim,VectorizedArray<value_type> > test;
        for (unsigned int d=0; d<dim; ++d)
          test[d][d] = div;
        phi.submit_gradient(test, q);
      }
      phi.integrate (false,true);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          (div_matrices[cell][0][v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
    }

    //mass matrix
    phi.reinit(cell);
    div_matrices[cell][1].resize(data.n_components_filled(cell));
    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      div_matrices[cell][1][v].reinit(total_dofs_per_cell, total_dofs_per_cell);
    for (unsigned int j=0; j<total_dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<total_dofs_per_cell; ++i)
        phi.write_cellwise_dof_value(i,make_vectorized_array(0.));
      phi.write_cellwise_dof_value(j,make_vectorized_array(1.));

      phi.evaluate (true,false,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
        phi.submit_value (phi.get_value(q), q);
      phi.integrate (true,false);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        if(phi.component_enriched(v))
          for (unsigned int i=0; i<total_dofs_per_cell; ++i)
            (div_matrices[cell][1][v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
        else//this is a non-enriched element
          {
            if(j<phi.std_dofs_per_cell*dim)
              for (unsigned int i=0; i<phi.std_dofs_per_cell*dim; ++i)
                (div_matrices[cell][1][v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
            else //diagonal
              (div_matrices[cell][1][v])(j,j) = 1.0;
          }
    }
  }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_divu_for_channel_stats(const MatrixFree<dim,value_type>                  &data,
                std::vector<double >      &test,
                const std::vector<parallel::distributed::Vector<value_type> >  &source,
                const std::pair<unsigned int,unsigned int>            &cell_range)
  {
#ifdef XWALL
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> phi(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> phi(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif
  AlignedVector<VectorizedArray<value_type> > JxW_values(phi.n_q_points);
  VectorizedArray<value_type> div_vec = make_vectorized_array(0.);
  VectorizedArray<value_type> vol_vec = make_vectorized_array(0.);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit(cell);
    phi.read_dof_values(source,0,source,dim+1);
    phi.evaluate(false,true);
    phi.fill_JxW_values(JxW_values);

    for (unsigned int q=0; q<phi.n_q_points; ++q)
    {
      vol_vec += JxW_values[q];
      div_vec += JxW_values[q]*std::abs(phi.get_divergence(q));
    }
  }
  value_type div = 0.;
  value_type vol = 0.;
  for (unsigned int v=0;v<VectorizedArray<value_type>::n_array_elements;v++)
  {
    div += div_vec[v];
    vol += vol_vec[v];
  }
  test.at(0)+=div;
  test.at(1)+=vol;

  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
      local_compute_divu_for_channel_stats_face (const MatrixFree<dim,double>              &data,
                  std::vector<double >      &test,
                  const std::vector<parallel::distributed::Vector<double> >  &source,
                  const std::pair<unsigned int,unsigned int>          &face_range)
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,0);
#endif  
AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval_xwall.n_q_points);
  VectorizedArray<value_type> div_vec = make_vectorized_array(0.);
  VectorizedArray<value_type> vol_vec = make_vectorized_array(0.);
  for (unsigned int face=face_range.first; face<face_range.second; ++face)
  {
    fe_eval_xwall.reinit(face);
    fe_eval_xwall.read_dof_values(source,0,source,dim+1);
    fe_eval_xwall.evaluate(true,false);
    fe_eval_xwall_neighbor.reinit(face);
    fe_eval_xwall_neighbor.read_dof_values(source,0,source,dim+1);
    fe_eval_xwall_neighbor.evaluate(true,false);
    fe_eval_xwall.fill_JxW_values(JxW_values);

    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      vol_vec += JxW_values[q];
      div_vec += JxW_values[q]*std::abs((fe_eval_xwall.get_value(q)-fe_eval_xwall_neighbor.get_value(q))*fe_eval_xwall.get_normal_vector(q));
    }
  }
  value_type div = 0.;
  value_type vol = 0.;
  for (unsigned int v=0;v<VectorizedArray<value_type>::n_array_elements;v++)
  {
    div += div_vec[v];
    vol += vol_vec[v];
  }
  test.at(2)+=div;
  test.at(3)+=vol;

  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
      local_compute_divu_for_channel_stats_boundary_face (const MatrixFree<dim,double>              &data,
                  std::vector<double >      &test,
                  const std::vector<parallel::distributed::Vector<double> >  &source,
                  const std::pair<unsigned int,unsigned int>          &face_range)
  {
     ;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_fast_grad_div_projection(const MatrixFree<dim,value_type>                  &data,
                std::vector<parallel::distributed::Vector<value_type> >      &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>            &cell_range) const
  {

//
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> velocity(data,xwallstatevec[0],xwallstatevec[1],0,3);
#else
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> velocity(data,xwallstatevec[0],xwallstatevec[1],0,0);
#endif

   std::vector<LAPACKFullMatrix<value_type> > matrices(VectorizedArray<value_type>::n_array_elements);
   AlignedVector<VectorizedArray<value_type> > JxW_values(velocity.n_q_points);
   for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
   {
     velocity.reinit(cell);
     const unsigned int total_dofs_per_cell = velocity.dofs_per_cell * dim;
     velocity.read_dof_values(solution_n,0,solution_n,dim);
     velocity.evaluate (true,false);
     VectorizedArray<value_type> volume;
     VectorizedArray<value_type> normmeanvel;
     {
       Tensor<1,dim,VectorizedArray<value_type> > meanvel;
       velocity.fill_JxW_values(JxW_values);
       meanvel = JxW_values[0]*velocity.get_value(0);
       volume = JxW_values[0];
       for (unsigned int q=1; q<velocity.n_q_points; ++q)
       {
         meanvel += JxW_values[q]*velocity.get_value(q);
         volume += JxW_values[q];
       }
       meanvel /=volume;
       normmeanvel = meanvel.norm();
     }
     velocity.reinit(cell);
     velocity.read_dof_values(src,0,src,dim);
     for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
       if (matrices[v].m() != total_dofs_per_cell)
         matrices[v].reinit(total_dofs_per_cell, total_dofs_per_cell);
       else
         matrices[v] = 0;

     // compute grad-div parameter
     //use definition Ohlhanskii et al. (2009)
//#ifdef STOKES
     const VectorizedArray<value_type> tau = K*normmeanvel*std::pow(volume,1./(double)dim) + make_vectorized_array<value_type>(VISCOSITY*K);
//     const VectorizedArray<value_type> tau = make_vectorized_array<value_type>(VISCOSITY*K);
//#else
//     const VectorizedArray<value_type> tau =
//       K*normmeanvel*std::pow(volume,1./(double)dim);
//#endif

     //now apply vectors to inverse matrix
//     for (unsigned int q=0; q<velocity.n_q_points; ++q)
//       velocity.submit_value (velocity.get_value(q), q);
//     velocity.integrate (true,false);

     for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
     {
       //mass matrix + tau* div-div matrix
       for (unsigned int i=0; i<total_dofs_per_cell; ++i)
         for (unsigned int j=0; j<total_dofs_per_cell; ++j)
           matrices[v](i,j) = div_matrices[cell][1][v](i,j)+tau[v]*div_matrices[cell][0][v](i,j);

       (matrices[v]).compute_lu_factorization();
       Vector<value_type> vector_input(total_dofs_per_cell);
       for (unsigned int j=0; j<total_dofs_per_cell; ++j)
         vector_input(j)=(velocity.read_cellwise_dof_value(j))[v];

 //        Vector<value_type> vector_result(total_dofs_per_cell);
       (matrices[v]).apply_lu_factorization(vector_input,false);
 //        (matrices[v]).vmult(vector_result,vector_input);
       for (unsigned int j=0; j<total_dofs_per_cell; ++j)
         velocity.write_cellwise_dof_value(j,vector_input(j),v);
     }
     velocity.set_dof_values (dst,0,dst,dim);
   }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_vorticity(const MatrixFree<dim,value_type>                  &data,
                std::vector<parallel::distributed::Vector<value_type> >      &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
//    //TODO Benjamin the vorticity lives only on the standard space
////#ifdef XWALL
////    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,3);
////    FEEvaluation<dim,fe_degree,n_q_points_1d_xwall,number_vorticity_components,value_type> phi(data,0,3);
////#else
////    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
//    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,0);
////    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,number_vorticity_components,value_type> fe_eval_xwall_phi(data,src.at(dim),src.at(dim+1),0,0);
//    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);

//
#ifdef XWALL
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> velocity_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
   FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
   FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);
#else
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,number_vorticity_components,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,0);
   FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
   FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);
#endif
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);
  const unsigned int dofs_per_cell = phi.dofs_per_cell;
  AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
//    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);

//no XWALL but with XWALL routine
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,0);

   //   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_xwall (data,0,0);
#ifdef XWALL
  std::vector<LAPACKFullMatrix<value_type> > matrices(VectorizedArray<value_type>::n_array_elements);
#endif
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
#ifdef XWALL
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_xwall.reinit (cell);
    if(fe_eval_xwall.enriched)
    {
      const unsigned int total_dofs_per_cell = fe_eval_xwall.dofs_per_cell * number_vorticity_components;
      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        matrices[v].reinit(total_dofs_per_cell);
      velocity_xwall.reinit(cell);
      velocity_xwall.read_dof_values(src,0,src,dim+1);
      velocity_xwall.evaluate (false,true,false);

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_xwall.evaluate (true,false,false);
        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
        {
  //        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
          fe_eval_xwall.submit_value (fe_eval_xwall.get_value(q), q);
        }
        fe_eval_xwall.integrate (true,false);

        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
            if(fe_eval_xwall.component_enriched(v))
              (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
            else//this is a non-enriched element
            {
              if(i<fe_eval_xwall.std_dofs_per_cell*number_vorticity_components && j<fe_eval_xwall.std_dofs_per_cell*number_vorticity_components)
                (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
              else if(i == j)//diagonal
                (matrices[v])(i,j) = 1.0;
            }
      }
//      for (unsigned int i=0; i<10; ++i)
//        std::cout << std::endl;
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        matrices[v].print(std::cout,14,8);

      //initialize again to get a clean version
      fe_eval_xwall.reinit (cell);
      //now apply vectors to inverse matrix
      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        fe_eval_xwall.submit_value (velocity_xwall.get_curl(q), q);
//        std::cout << velocity_xwall.get_curl(q)[2][0] << "   "  << velocity_xwall.get_curl(q)[2][1] << std::endl;
      }
      fe_eval_xwall.integrate (true,false);


      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrices[v]).compute_lu_factorization();
        Vector<value_type> vector_input(total_dofs_per_cell);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          vector_input(j)=(fe_eval_xwall.read_cellwise_dof_value(j))[v];

  //        Vector<value_type> vector_result(total_dofs_per_cell);
        (matrices[v]).apply_lu_factorization(vector_input,false);
  //        (matrices[v]).vmult(vector_result,vector_input);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          fe_eval_xwall.write_cellwise_dof_value(j,vector_input(j),v);
      }
      fe_eval_xwall.set_dof_values (dst,0,dst,number_vorticity_components);

    }
    else
#endif
    {
      phi.reinit(cell);
      velocity.reinit(cell);
      velocity.read_dof_values(src,0);
      velocity.evaluate (false,true,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
      Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
//      std::cout << omega[2][0] << "    " << omega[2][1] << std::endl;
        phi.submit_value (omega, q);
      }
      phi.integrate (true,false);

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }

//    else

//    {
//      phi.read_dof_values(src,0);
//
//      inverse.fill_inverse_JxW_values(coefficients);
//      inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());
//
//      phi.set_dof_values(dst,0);
//    }
//  }

  //


  }

  template <int dim, typename FEEval>
  struct CurlCompute
  {
    static
    Tensor<1,dim,VectorizedArray<typename FEEval::number_type> >
    compute(FEEval     &fe_eval,
        const unsigned int   q_point)
    {
    return fe_eval.get_curl(q_point);
    }
  };

  template <typename FEEval>
  struct CurlCompute<2,FEEval>
  {
  static
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> >
    compute(FEEval     &fe_eval,
        const unsigned int   q_point)
    {
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> > rot;
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> > temp = fe_eval.get_gradient(q_point);
    rot[0] = temp[1];
    rot[1] = - temp[0];
    return rot;
    }
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  shift_pressure (parallel::distributed::Vector<value_type>  &pressure)
  {
    parallel::distributed::Vector<value_type> vec1(pressure);
    for(unsigned int i=0;i<vec1.local_size();++i)
      vec1.local_element(i) = 1.;
    AnalyticalSolution<dim> analytical_solution(dim,time+time_step);
    double exact = analytical_solution.value(first_point);
    double current = 0.;
    if (pressure.locally_owned_elements().is_element(dof_index_first_point))
      current = pressure(dof_index_first_point);
    current = Utilities::MPI::sum(current, MPI_COMM_WORLD);
    pressure.add(exact-current,vec1);
  }


  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
             parallel::distributed::Vector<value_type>      &dst)
  {

  dst = 0;
  // data.loop
  data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_face,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_boundary_face,
        this, dst, src);

  if(pure_dirichlet_bc)
    {  pressure_poisson_solver.get_matrix().apply_nullspace_projection(dst);  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure (const MatrixFree<dim,value_type>                &data,
            parallel::distributed::Vector<double>       &dst,
            const std::vector<parallel::distributed::Vector<double> >  &src,
            const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
#ifdef XWALL
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
  FEEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,1,3);
#else
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,1);
  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);
#endif

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_xwall.reinit (cell);
    pressure.reinit (cell);
    fe_eval_xwall.read_dof_values(src,0,src,dim+1);
#ifdef DIVUPARTIAL
    fe_eval_xwall.evaluate (true,false,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      pressure.submit_gradient (fe_eval_xwall.get_value(q)/time_step, q);
    }
    pressure.integrate (false,true);
#else
    fe_eval_xwall.evaluate (false,true,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      pressure.submit_value (-fe_eval_xwall.get_divergence(q)/time_step, q);
    }
    pressure.integrate (true,false);
#endif
    pressure.distribute_local_to_global (dst);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_face (const MatrixFree<dim,value_type>               &data,
                parallel::distributed::Vector<double>      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
    FEFaceEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,true,1,3);
    FEFaceEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure_neighbor (data,false,1,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,1);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,true,1,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure_neighbor (data,false,1,1);
#endif

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_xwall.reinit (face);
    fe_eval_xwall_neighbor.reinit (face);
    pressure.reinit (face);
    pressure_neighbor.reinit (face);
    fe_eval_xwall.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall_neighbor.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall.evaluate (true,false,false);
    fe_eval_xwall_neighbor.evaluate (true,false,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_xwall.get_normal_vector(q);
#ifdef DIVUPARTIAL
      Tensor<1,dim,VectorizedArray<value_type> > meanvel = 0.5*(fe_eval_xwall.get_value(q)+fe_eval_xwall_neighbor.get_value(q));
#else
      Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_xwall.get_value(q)*0.;
#endif
      VectorizedArray<value_type> submitvalue = normal * meanvel;

      pressure.submit_value ((-submitvalue)/time_step, q);
      pressure_neighbor.submit_value (submitvalue/time_step, q);
    }
    pressure.integrate (true,false);
    pressure_neighbor.integrate (true,false);
    pressure.distribute_local_to_global (dst);
    pressure_neighbor.distribute_local_to_global (dst);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_boundary_face (const MatrixFree<dim,value_type>               &data,
                    parallel::distributed::Vector<double>      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const
  {

#ifdef XWALL
//    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_nx (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_n (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_nm (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_nm2 (data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> omega_n(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> omega_nm(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> omega_nm2(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,true,1,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_n (data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_nm (data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_xwall_nm2 (data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_n(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_nm(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> omega_nm2(data,xwallstatevec[0],xwallstatevec[1],true,0,2);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree+(fe_degree+2)/2,1,value_type> pressure (data,true,1,2);
#endif

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
//    fe_eval_xwall_nx.reinit(face);
//    fe_eval_xwall_nx.read_dof_values(src,0,src,dim);
//    fe_eval_xwall_nx.evaluate (true,false);
    fe_eval_xwall.reinit(face);
    fe_eval_xwall.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall.evaluate (true,false);
    fe_eval_xwall_n.reinit(face);
    fe_eval_xwall_n.evaluate_eddy_viscosity(solution_n,face,fe_eval_xwall_n.read_cell_data(element_volume));
    pressure.reinit (face);
    fe_eval_xwall_n.read_dof_values(solution_n,0,solution_n,dim+1);
    fe_eval_xwall_n.evaluate (true,true);
    fe_eval_xwall_nm.reinit (face);
    fe_eval_xwall_nm.read_dof_values(solution_nm,0,solution_nm,dim+1);
    fe_eval_xwall_nm.evaluate (true,true);
    fe_eval_xwall_nm2.reinit (face);
    fe_eval_xwall_nm2.read_dof_values(solution_nm2,0,solution_nm2,dim+1);
    fe_eval_xwall_nm2.evaluate (true,true);

    omega_n.reinit (face);
    omega_n.read_dof_values(vorticity_n,0,vorticity_n,number_vorticity_components);
    omega_n.evaluate (false,true);
    omega_nm.reinit (face);
    omega_nm.read_dof_values(vorticity_nm,0,vorticity_nm,number_vorticity_components);
    omega_nm.evaluate (false,true);
    omega_nm2.reinit (face);
    omega_nm2.read_dof_values(vorticity_nm2,0,vorticity_nm2,number_vorticity_components);
    omega_nm2.evaluate (false,true);

    double factor = pressure_poisson_solver.get_matrix().get_penalty_factor();
    //VectorizedArray<value_type> sigmaF = std::abs(pressure.get_normal_volume_fraction()) * (value_type)factor;
    VectorizedArray<value_type> sigmaF = fe_eval_xwall_n.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()) * (value_type)factor;

    for(unsigned int q=0;q<pressure.n_q_points;++q)
    {
      if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Inflow and wall boundaries
      {
        // p+ =  p-
        Point<dim,VectorizedArray<value_type> > q_points = pressure.quadrature_point(q);
        VectorizedArray<value_type> h;

        Tensor<1,dim,VectorizedArray<value_type> > dudt_np, rhs_np;
        for(unsigned int d=0;d<dim;++d)
        {
          PressureBC_dudt<dim> neumann_boundary_pressure(d,time+time_step);
          RHS<dim> f(d,time+time_step);
          value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
          value_type array_f [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
            array_dudt[n] = neumann_boundary_pressure.value(q_point);
            array_f[n] = f.value(q_point);
          }
          dudt_np[d].load(&array_dudt[0]);
          rhs_np[d].load(&array_f[0]);
        }

        Tensor<1,dim,VectorizedArray<value_type> > normal = pressure.get_normal_vector(q);
        Tensor<1,dim,VectorizedArray<value_type> > u_n = fe_eval_xwall_n.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > grad_u_n = fe_eval_xwall_n.get_gradient(q);
        Tensor<1,dim,VectorizedArray<value_type> > conv_n = grad_u_n * u_n;
        Tensor<1,dim,VectorizedArray<value_type> > u_nm = fe_eval_xwall_nm.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > grad_u_nm = fe_eval_xwall_nm.get_gradient(q);
        Tensor<1,dim,VectorizedArray<value_type> > u_nm2 = fe_eval_xwall_nm2.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > grad_u_nm2 = fe_eval_xwall_nm2.get_gradient(q);
        Tensor<1,dim,VectorizedArray<value_type> > conv_nm = grad_u_nm * u_nm;
        Tensor<1,dim,VectorizedArray<value_type> > conv_nm2 = grad_u_nm2 * u_nm2;
#ifdef CONSCONVPBC
        conv_n += fe_eval_xwall_n.get_divergence(q) * u_n;
        conv_nm += fe_eval_xwall_nm.get_divergence(q) * u_nm;
        conv_nm2 += fe_eval_xwall_nm2.get_divergence(q) * u_nm2;
#endif
//          Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,decltype(omega_n)>::compute(omega_n,q);
//          Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,decltype(omega_nm)>::compute(omega_nm,q);

          // kaiser cluster: decltype() is unknown
#ifdef XWALL
        Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> >::compute(omega_n,q);
        Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> >::compute(omega_nm,q);
        Tensor<1,dim,VectorizedArray<value_type> > rot_nm2 = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> >::compute(omega_nm2,q);
#else
        Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> >::compute(omega_n,q);
        Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> >::compute(omega_nm,q);
        Tensor<1,dim,VectorizedArray<value_type> > rot_nm2 = CurlCompute<dim,FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+(fe_degree+2)/2,number_vorticity_components,value_type> >::compute(omega_nm2,q);
#endif

        h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(conv_n + fe_eval_xwall_n.eddyvisc[q]*rot_n)
                + make_vectorized_array<value_type>(beta[1])*(conv_nm + fe_eval_xwall_n.eddyvisc[q]*rot_nm)
                + make_vectorized_array<value_type>(beta[2])*(conv_nm2 + fe_eval_xwall_n.eddyvisc[q]*rot_nm2));

        // Stokes
#ifdef STOKES
        h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(fe_eval_xwall_n.eddyvisc[q]*rot_n)
                + make_vectorized_array<value_type>(beta[1])*( fe_eval_xwall_n.eddyvisc[q]*rot_nm)
                + make_vectorized_array<value_type>(beta[2])*(fe_eval_xwall_n.eddyvisc[q]*rot_nm2));
#endif

#ifdef DIVUPARTIAL
        Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_xwall.get_value(q);

//      Tensor<1,dim,VectorizedArray<value_type> > g_np;
//        for(unsigned int d=0;d<dim;++d)
//        {
//          AnalyticalSolution<dim> dirichlet_boundary(d,time+time_step);
//          value_type array [VectorizedArray<value_type>::n_array_elements];
//          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//          {
//            Point<dim> q_point;
//            for (unsigned int d=0; d<dim; ++d)
//            q_point[d] = q_points[d][n];
//            array[n] = dirichlet_boundary.value(q_point);
//          }
//          g_np[d].load(&array[0]);
//        }
//        Tensor<1,dim,VectorizedArray<value_type> > meanvel = make_vectorized_array<value_type>(gamma0)*g_np;
#else
        Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_xwall.get_value(q)*0.;
#endif
        VectorizedArray<value_type> submitvalue;
        submitvalue = normal[0]*meanvel[0];
        for (unsigned int i = 1; i<dim;i++)
          submitvalue += normal[i]*meanvel[i];

        pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
        pressure.submit_value(h-(submitvalue)/time_step,q);
      }
      else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
      {
        // p+ = - p- + 2g
        Point<dim,VectorizedArray<value_type> > q_points = pressure.quadrature_point(q);
        VectorizedArray<value_type> g;

        AnalyticalSolution<dim> dirichlet_boundary(dim,time+time_step);
        value_type array [VectorizedArray<value_type>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
          array[n] = dirichlet_boundary.value(q_point);
        }
        g.load(&array[0]);

        Tensor<1,dim,VectorizedArray<value_type> > normal = pressure.get_normal_vector(q);
#ifdef DIVUPARTIAL
        Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_xwall.get_value(q);
#else
        Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_xwall.get_value(q)*0.;
#endif
        VectorizedArray<value_type> submitvalue;
        submitvalue = normal[0]*meanvel[0];
        for (unsigned int i = 1; i<dim;i++)
          submitvalue += normal[i]*meanvel[i];

        pressure.submit_normal_gradient(-g,q);
        pressure.submit_value(2.0 *sigmaF * g - (submitvalue)/time_step,q);
      }
    }
    pressure.integrate(true,true);
    pressure.distribute_local_to_global(dst);
  }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_solve_projection(const MatrixFree<dim,value_type>                               &data,
                         std::vector<parallel::distributed::Vector<value_type> >        &dst,
                         const std::vector<parallel::distributed::Vector<value_type> >  &src,
                         const std::pair<unsigned int,unsigned int>                     &cell_range)
  {
#ifdef XWALL
    AssertThrow(false,
                ExcMessage("XWall should not arrive in iterative projection solver"));
#endif
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data, 0, 0);
    const unsigned int total_dofs_per_cell = fe_eval.dofs_per_cell * dim;

    AlignedVector<VectorizedArray<value_type> > solution(total_dofs_per_cell);
    MatrixProjectionStep<dim,fe_degree> matrix_projection_step(data,0,0);
    SolverCGmod<VectorizedArray<double> > cg_solver(total_dofs_per_cell, 1e-12, 1e-9, 1e4); //cg_solver(total_dofs_per_cell, 1e-12, 1e-9, 1e4);
    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(solution_n,0);
        fe_eval.evaluate (true,false);
        VectorizedArray<value_type> volume;
        VectorizedArray<value_type> normmeanvel;

        Tensor<1,dim,VectorizedArray<value_type> > meanvel;
        fe_eval.fill_JxW_values(JxW_values);
        meanvel = JxW_values[0]*fe_eval.get_value(0);
        volume = JxW_values[0];
        for (unsigned int q=1; q<fe_eval.n_q_points; ++q)
          {
            meanvel += JxW_values[q]*fe_eval.get_value(q);
            volume += JxW_values[q];
          }
        meanvel /=volume;
        normmeanvel = meanvel.norm();

        fe_eval.read_dof_values(src,0);

        // compute grad-div parameter: use definition Ohlhanskii et al. (2009)
        const VectorizedArray<value_type> tau = K*normmeanvel*std::pow(volume,1./(double)dim) + make_vectorized_array<value_type>(VISCOSITY*K);

        matrix_projection_step.setup(cell, tau);
        cg_solver.solve(&matrix_projection_step, solution.begin(), fe_eval.begin_dof_values());
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j] = solution[j];
        fe_eval.set_dof_values (dst,0);
      }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
             std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  
  for(unsigned int d=0;d<dim;++d)
  {
    dst[d] = 0;
  }
  // data.loop
  data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_face,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_boundary_face,
        this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_weak_projection (const parallel::distributed::BlockVector<value_type>      &src,
                  parallel::distributed::BlockVector<value_type>       &dst)
  {
    for(unsigned int d=0;d<dim;++d)
    {
      dst.block(d)=0;
    }
    data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_weak_projection,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_weak_projection_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_weak_projection_boundary_face,
            this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  calculate_stab_fac_weak_projection()
  {
    std::vector<parallel::distributed::Vector<value_type> > dummy;
    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_calculate_stab_fac_weak_projection,
                               this, dummy, dummy);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_calculate_stab_fac_weak_projection(const MatrixFree<dim,value_type>        &data,
                                            std::vector<parallel::distributed::Vector<value_type> > &,
                                            const std::vector<parallel::distributed::Vector<value_type> > &,
                                            const std::pair<unsigned int,unsigned int>   &cell_range)
  {
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval(data,xwallstatevec[0],xwallstatevec[1],0,0);

    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(solution_n,0,solution_n,dim);
      fe_eval.evaluate (true,false);
      VectorizedArray<value_type> volume;
      VectorizedArray<value_type> norm_U_mean;
      {
        Tensor<1,dim,VectorizedArray<value_type> > U_mean;
        fe_eval.fill_JxW_values(JxW_values);
        U_mean = JxW_values[0]*fe_eval.get_value(0);
        volume = JxW_values[0];
        for (unsigned int q=1; q<fe_eval.n_q_points; ++q)
        {
          U_mean += JxW_values[q]*fe_eval.get_value(q);
          volume += JxW_values[q];
        }
        U_mean /=volume;
        norm_U_mean = U_mean.norm();
      }
//      array_stab_fac_weak_projection[cell] = K*norm_U_mean*std::pow(volume,1./(double)dim)/VISCOSITY;
      array_stab_fac_weak_projection[cell] = K*norm_U_mean*std::pow(volume,1./(double)dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_weak_projection (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::BlockVector<double>       &dst,
            const parallel::distributed::BlockVector<double>  &src,
            const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_xwall.reinit(cell);
      fe_eval_xwall.read_dof_values(src,0,src,dim+1);
      fe_eval_xwall.evaluate (true,true,false);
  
      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        VectorizedArray<value_type> tau = array_stab_fac_weak_projection[cell];
        Tensor<1,dim,VectorizedArray<value_type> > velocity = fe_eval_xwall.get_value(q);
        VectorizedArray<value_type > divergence = fe_eval_xwall.get_divergence(q);
        Tensor<2,dim,VectorizedArray<value_type> > unit_times_divU;
          for (unsigned int d=0; d<dim; ++d)
            unit_times_divU[d][d] = divergence;
        fe_eval_xwall.submit_value(velocity, q);
        fe_eval_xwall.submit_gradient(tau*unit_times_divU, q);
      }
      fe_eval_xwall.integrate (true,true);
      fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_weak_projection_face (const MatrixFree<dim,value_type>       &data,
                parallel::distributed::BlockVector<double>      &dst,
                const parallel::distributed::BlockVector<double>   &src,
                const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,0);
   
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall_neighbor.reinit (face);
      fe_eval_xwall.read_dof_values(src,0,src,dim+1);
      fe_eval_xwall.evaluate(true,false);
      fe_eval_xwall_neighbor.read_dof_values(src,0,src,dim+1);
      fe_eval_xwall_neighbor.evaluate(true,false);
     
      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        VectorizedArray<value_type> tau = 0.5 * (fe_eval_xwall.read_cell_data(array_stab_fac_weak_projection)+fe_eval_xwall_neighbor.read_cell_data(array_stab_fac_weak_projection));
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_xwall.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_xwall_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

        fe_eval_xwall.submit_value(tau*jump_value,q);
        fe_eval_xwall_neighbor.submit_value(-tau*jump_value,q);
      }
      fe_eval_xwall.integrate(true,false);
      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
      fe_eval_xwall_neighbor.integrate(true,false);
      fe_eval_xwall_neighbor.distribute_local_to_global(dst,0,dst,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_weak_projection_boundary_face (const MatrixFree<dim,value_type>       &data,
                    parallel::distributed::BlockVector<double>     &dst,
                    const parallel::distributed::BlockVector<double>   &src,
                    const std::pair<unsigned int,unsigned int>  &face_range) const
  {
//    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
//    for(unsigned int face=face_range.first; face<face_range.second; face++)
//    {
//      fe_eval_xwall.reinit (face);
//      fe_eval_xwall.read_dof_values(src,0,src,dim+1);
//      fe_eval_xwall.evaluate(true,false);
//
//      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
//      {
//        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
//        {
//          // on Dirichlet boundaries use u⁺ = - u⁻ + 2g
//          VectorizedArray<value_type> tau = fe_eval_xwall.read_cell_data(array_stab_fac_weak_projection);
//          // jump value = u⁻ - u⁺ = 2u⁻ - 2g
//          Tensor<1,dim,VectorizedArray<value_type> > jump_value =
//              make_vectorized_array<value_type>(2.0)*fe_eval_xwall.get_value(q);
//          fe_eval_xwall.submit_value(tau*jump_value,q);
//        }
//        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
//        {
//          // on Neumann boundaries use u⁺ = u⁻
//          // jump_value = 0 -> do-nothing
//          Tensor<1,dim,VectorizedArray<value_type> > jump_value;
//          fe_eval_xwall.submit_value(jump_value,q);
//        }
//      }
//      fe_eval_xwall.integrate(true,false);
//      fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
//    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                  std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<dim;++d)
  {
    dst[d] = 0;
#ifdef XWALL
    dst[d+dim] = 0;
#endif
  }
  data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_boundary_face,
            this, dst, src);

#ifdef COMPDIV
    divergence_old = 0.;
    divergence_new = 0.;
    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divergence,
                               this, divergence_old, dst);
#endif


    if(K>0.0+1.0e-14)
    {
#if LOWMEMORY == 1 || defined(XWALL)
    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_grad_div_projection,this, dst, dst);
#elif LOWMEMORY == 2
    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_solve_projection,this, dst, dst);
#else
    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_fast_grad_div_projection,this, dst, dst);
#endif
    }
    else
    {
      data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
                                 this, dst, dst);
    }


#ifdef COMPDIV
  data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divergence,
                             this, divergence_new, dst);
#endif
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_projection (const MatrixFree<dim,value_type>              &data,
          std::vector<parallel::distributed::Vector<double> >      &dst,
          const std::vector<parallel::distributed::Vector<double> >  &src,
          const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
#ifdef XWALL
    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,3);
    FEEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,1,3);
#else
  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall (data,xwallstatevec[0],xwallstatevec[1],0,0);
  FEEvaluation<dim,fe_degree_p,fe_degree+1,1,value_type> pressure (data,1,0);
#endif

  const VectorizedArray<value_type> fac = make_vectorized_array(time_step);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_xwall.reinit (cell);
    pressure.reinit (cell);
    pressure.read_dof_values(src,dim);
    fe_eval_xwall.read_dof_values(src,0,src,dim+1);
    fe_eval_xwall.evaluate(true,false);
#ifdef PRESPARTIAL
    pressure.evaluate (true,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      Tensor<2,dim,VectorizedArray<value_type> > test;
      for (unsigned int a=0;a<dim;a++)
            test[a][a] = fac;

      test *= pressure.get_value(q);
      fe_eval_xwall.submit_gradient (test, q);
      fe_eval_xwall.submit_value(fe_eval_xwall.get_value(q),q);
    }
    fe_eval_xwall.integrate (true,true);
#else
    pressure.evaluate (false,true);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      fe_eval_xwall.submit_value(fe_eval_xwall.get_value(q)-fac*pressure.get_gradient(q),q);
    }
    fe_eval_xwall.integrate (true,false);
#endif
    fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_projection_face (const MatrixFree<dim,value_type>               &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,3);
    FEFaceEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,true,1,3);
    FEFaceEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure_neighbor (data,false,1,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,1);
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall_neighbor(data,xwallstatevec[0],xwallstatevec[1],false,0,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,true,1,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure_neighbor (data,false,1,1);
#endif
    const VectorizedArray<value_type> fac = make_vectorized_array(time_step);
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_xwall.reinit (face);
    fe_eval_xwall_neighbor.reinit (face);
    pressure.reinit (face);
    pressure_neighbor.reinit (face);
    pressure.read_dof_values(src,dim);
    pressure_neighbor.read_dof_values(src,dim);
    pressure.evaluate (true,false);
    pressure_neighbor.evaluate (true,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > normal = fac*pressure.get_normal_vector(q);
//      Tensor<1,dim,VectorizedArray<value_type> > meanvel = 0.5*(fe_eval_xwall.get_value(q)+fe_eval_xwall_neighbor.get_value(q));
#ifdef PRESPARTIAL
      VectorizedArray<value_type> meanpres = 0.5*(pressure.get_value(q)+pressure_neighbor.get_value(q));
#else
      VectorizedArray<value_type> meanpres = make_vectorized_array(0.);
#endif
      normal*=meanpres;

      fe_eval_xwall.submit_value (-normal, q);
      fe_eval_xwall_neighbor.submit_value (normal, q);
    }
    fe_eval_xwall.integrate (true,false);
    fe_eval_xwall_neighbor.integrate (true,false);
    fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
    fe_eval_xwall_neighbor.distribute_local_to_global (dst,0,dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_projection_boundary_face (const MatrixFree<dim,value_type>               &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {
#ifdef PRESPARTIAL

#ifdef XWALL
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,3);
    FEFaceEvaluation<dim,fe_degree_p,n_q_points_1d_xwall,1,value_type> pressure (data,true,1,3);
#else
    FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree_p+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,true,1,1);
#endif

  const VectorizedArray<value_type> fac = make_vectorized_array(time_step);
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval_xwall.reinit (face);
    pressure.reinit (face);
    pressure.read_dof_values(src,dim);
    pressure.evaluate (true,false);
    for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > normal = fac*pressure.get_normal_vector(q);
      VectorizedArray<value_type> meanpres = pressure.get_value(q);
      normal*=meanpres;
      fe_eval_xwall.submit_value (-normal, q);
    }
    fe_eval_xwall.integrate (true,false);
    fe_eval_xwall.distribute_local_to_global (dst,0,dst,dim);
  }
#endif

#ifdef WEAK_PROJECTION
//  FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],true,0,0);
//  for(unsigned int face=face_range.first; face<face_range.second; face++)
//  {
//    fe_eval_xwall.reinit (face);
//    fe_eval_xwall.read_dof_values(src,0,src,dim+1);
//    fe_eval_xwall.evaluate(true,false);
//
//    for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
//    {
//      if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
//      {
//        VectorizedArray<value_type> tau = fe_eval_xwall.read_cell_data(array_stab_fac_weak_projection);
//
//        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_xwall.quadrature_point(q);
//        Tensor<1,dim,VectorizedArray<value_type> > g_np;
//        for(unsigned int d=0;d<dim;++d)
//        {
//          AnalyticalSolution<dim> dirichlet_boundary(d,time+time_step);
//          value_type array [VectorizedArray<value_type>::n_array_elements];
//          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//          {
//            Point<dim> q_point;
//            for (unsigned int d=0; d<dim; ++d)
//            q_point[d] = q_points[d][n];
//            array[n] = dirichlet_boundary.value(q_point);
//          }
//          g_np[d].load(&array[0]);
//        }
//        Tensor<1,dim,VectorizedArray<value_type> > jump_value = -make_vectorized_array<value_type>(2.0)*g_np;
//        jump_value = gamma0 * jump_value;
//        // minus sign in following expression is due to the fact that the term has been moved to the right hand side
//        fe_eval_xwall.submit_value(-tau*jump_value,q);
//      }
//      else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
//      {
//        // on Neumann boundaries use velocity + = velocity -
//        // jump_value = 0 -> do-nothing
//        Tensor<1,dim,VectorizedArray<value_type> > jump_value;
//        fe_eval_xwall.submit_value(jump_value,q);
//      }
//    }
//    fe_eval_xwall.integrate(true,false);
//    fe_eval_xwall.distribute_local_to_global(dst,0,dst,dim);
//  }
#endif
  }

  namespace
  {
    template <int dim>
    Point<dim> get_direction()
    {
      Point<dim> direction;
      direction[dim-1] = 1.;
      return direction;
    }

    template <int dim>
    Point<dim> get_center()
    {
      Point<dim> center;
      center[0] = 0.5;
      center[1] = 0.2;
      return center;
    }
  }

  template<int dim>
  class NavierStokesProblem
  {
  public:
  typedef typename NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;
  NavierStokesProblem(const unsigned int n_refinements);
  void run();

  private:
//  Point<dim> grid_transform (const Point<dim> &in);
  void make_grid_and_dofs ();
  void write_output(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
             std::vector<parallel::distributed::Vector<value_type>>   &vorticity,
             XWall<dim,fe_degree,fe_degree_xwall>* xwall,
#ifdef COMPDIV
             parallel::distributed::Vector<value_type>   &div_old,
             parallel::distributed::Vector<value_type>   &div_new,
#endif
             const unsigned int                     timestep_number);

  int numchsamp;
  double udiv_samp;
  double udiff_samp;
  void init_channel_statistics();
  void compute_divu_statistics(std::vector<parallel::distributed::Vector<value_type> >   &vel_hat, NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> & nsoperation);
  void write_divu(std::vector<parallel::distributed::Vector<value_type> >   &vel_hat, NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> & nsoperation, double time, unsigned int time_step_number);
  void calculate_error(std::vector<parallel::distributed::Vector<value_type>> &solution_n, const double delta_t=0.0);
  void calculate_time_step();

  ConditionalOStream pcout;

  double time, time_step;

  std_cxx11::shared_ptr<Manifold<dim> > cylinder_manifold;

  parallel::distributed::Triangulation<dim> triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;
  FE_DGQArbitraryNodes<dim>  fe;
  FE_DGQArbitraryNodes<dim>  fe_p;
  FE_DGQArbitraryNodes<dim>  fe_xwall;
  MappingQ<dim>   mapping;
  DoFHandler<dim>  dof_handler;
  DoFHandler<dim>  dof_handler_p;
  DoFHandler<dim>  dof_handler_xwall;

  const double cfl;
  const unsigned int n_refinements;

  std::set<types::boundary_id> dirichlet_boundary;
  std::set<types::boundary_id> neumann_boundary;
  };

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps):
  pcout (std::cout,
         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  time(START_TIME),
  time_step(0.0),
  cylinder_manifold(dim == 2 ?
                    static_cast<Manifold<dim>*>(new HyperBallBoundary<dim>(get_center<dim>(), 0.05)) :
                    static_cast<Manifold<dim>*>(new CylindricalManifold<dim>(get_direction<dim>(), get_center<dim>()))),
  triangulation(MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe(QGaussLobatto<1>(fe_degree+1)),
  fe_p(QGaussLobatto<1>(fe_degree_p+1)),
  fe_xwall(QGaussLobatto<1>(fe_degree_xwall+1)),
  mapping(fe_degree),
  dof_handler(triangulation),
  dof_handler_p(triangulation),
  dof_handler_xwall(triangulation),
  cfl(CFL/pow(fe_degree,2.0)),
  n_refinements(refine_steps)
  {
  pcout << std::endl << std::endl << std::endl
  << "/******************************************************************/" << std::endl
  << "/*                                                                */" << std::endl
  << "/*     Solver for the incompressible Navier-Stokes equations      */" << std::endl
  << "/*                                                                */" << std::endl
  << "/******************************************************************/" << std::endl
  << std::endl;
  }

  template <int dim>
  Point<dim> grid_transform (const Point<dim> &in)
  {
    Point<dim> out = in;

    out[0] = in(0)-numbers::PI;
#ifdef XWALL    //wall-model
    out[1] =  2.*in(1)-1.;
#else    //no wall model
    out[1] =  std::tanh(GRID_STRETCH_FAC*(2.*in(1)-1.))/std::tanh(GRID_STRETCH_FAC);
#endif
    out[2] = in(2)-0.5*numbers::PI;
    return out;
  }

  void create_triangulation(Triangulation<2> &tria,
                              const bool compute_in_2d = true)
  {
    HyperBallBoundary<2> boundary(Point<2>(0.5,0.2), 0.05);
  Triangulation<2> left, middle, right, tmp, tmp2;
  // old mesh
//  std::vector<unsigned int> ref_1(2, 3);
//  ref_1[1] = 4;
  // old mesh
  // new mesh
  std::vector<unsigned int> ref_1(2, 2);
  ref_1[1] = 2;
  // new mesh

  GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(), Point<2>(0.3, 0.41), false);
  // old mesh
//  std::vector<unsigned int> ref_2(2, 18);
//  ref_2[1] = 4;
  // old mesh
  // new mesh
  std::vector<unsigned int> ref_2(2, 9);
  ref_2[1] = 2;
  // new mesh

  GridGenerator::subdivided_hyper_rectangle(right, ref_2,Point<2>(0.7, 0), Point<2>(2.5, 0.41), false);

  // create middle part first as a hyper shell
  GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
  middle.set_manifold(0, boundary);
  middle.refine_global(1);

  //for (unsigned int v=0; v<middle.get_vertices().size(); ++v)
  //  const_cast<Point<dim> &>(middle.get_vertices()[v]) = 0.4 / 3. * middle.get_vertices()[v];

  // then move the vertices to the points where we want them to be to create a
  // slightly asymmetric cube with a hole
  for (Triangulation<2>::cell_iterator cell = middle.begin();
     cell != middle.end(); ++cell)
    for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
    {
      Point<2> &vertex = cell->vertex(v);
      if (std::abs(vertex[0] - 0.7) < 1e-10 &&
        std::abs(vertex[1] - 0.2) < 1e-10)
      vertex = Point<2>(0.7, 0.205);
      else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
           std::abs(vertex[1] - 0.3) < 1e-10)
      vertex = Point<2>(0.7, 0.41);
      else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
           std::abs(vertex[1] - 0.1) < 1e-10)
      vertex = Point<2>(0.7, 0);
      else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
           std::abs(vertex[1] - 0.4) < 1e-10)
      vertex = Point<2>(0.5, 0.41);
      else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
           std::abs(vertex[1] - 0.0) < 1e-10)
      vertex = Point<2>(0.5, 0.0);
      else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
           std::abs(vertex[1] - 0.3) < 1e-10)
      vertex = Point<2>(0.3, 0.41);
      else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
           std::abs(vertex[1] - 0.1) < 1e-10)
      vertex = Point<2>(0.3, 0);
      else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
           std::abs(vertex[1] - 0.2) < 1e-10)
      vertex = Point<2>(0.3, 0.205);
      else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
           std::abs(vertex[1] - 0.13621) < 1e-4)
      vertex = Point<2>(0.59, 0.11);
      else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
           std::abs(vertex[1] - 0.26379) < 1e-4)
      vertex = Point<2>(0.59, 0.29);
      else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
           std::abs(vertex[1] - 0.13621) < 1e-4)
      vertex = Point<2>(0.41, 0.11);
      else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
           std::abs(vertex[1] - 0.26379) < 1e-4)
      vertex = Point<2>(0.41, 0.29);
    }

  // refine once to create the same level of refinement as in the
  // neighboring domains
  // old mesh
  // middle.refine_global(1);
  // old mesh

  // must copy the triangulation because we cannot merge triangulations with
  // refinement...
  GridGenerator::flatten_triangulation(middle, tmp2);

  if (compute_in_2d)
    GridGenerator::merge_triangulations (tmp2, right, tria);
  else
    {
    GridGenerator::merge_triangulations (left, tmp2, tmp);
    GridGenerator::merge_triangulations (tmp, right, tria);
    }

  // Set the cylinder boundary  to 2, outflow to 1, the rest to 0.
  for (Triangulation<2>::active_cell_iterator cell=tria.begin() ;
     cell != tria.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
    if (cell->face(f)->at_boundary())
    {
      if (std::abs(cell->face(f)->center()[0] - (compute_in_2d ? 0.3 : 0)) < 1e-12)
        cell->face(f)->set_all_boundary_ids(0);
      else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
        cell->face(f)->set_all_boundary_ids(1);
      else if (Point<2>(0.5,0.2).distance(cell->face(f)->center())<=0.05)
      {
        cell->face(f)->set_all_manifold_ids(10);
        cell->face(f)->set_all_boundary_ids(2);
      }
      else
        cell->face(f)->set_all_boundary_ids(0);
    }
  }

  void create_triangulation(Triangulation<3> &tria)
  {
    Triangulation<2> tria_2d;
    create_triangulation(tria_2d, false);
    // new mesh
    GridGenerator::extrude_triangulation(tria_2d, 3, 0.41, tria);
    // new mesh
    // old mesh
//    GridGenerator::extrude_triangulation(tria_2d, 5, 0.41, tria);
    // old mesh

  // Set the cylinder boundary  to 2, outflow to 1, the rest to 0.
  for (Triangulation<3>::active_cell_iterator cell=tria.begin() ;
     cell != tria.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
    if (cell->face(f)->at_boundary())
    {
      if (std::abs(cell->face(f)->center()[0]) < 1e-12)
        cell->face(f)->set_all_boundary_ids(0);
      else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
        cell->face(f)->set_all_boundary_ids(1);
      else if (Point<3>(0.5,0.2,cell->face(f)->center()[2]).distance(cell->face(f)->center())<=0.05)
      {
        cell->face(f)->set_all_manifold_ids(10);
        cell->face(f)->set_all_boundary_ids(2);
      }
      else
        cell->face(f)->set_all_boundary_ids(0);
    }
  }

  template<int dim>
  void NavierStokesProblem<dim>::make_grid_and_dofs ()
  {
    /* --------------- Generate grid ------------------- */
    //turbulent channel flow
#ifdef CHANNEL
    Point<dim> coordinates;
    coordinates[0] = 2.0*numbers::PI;
    coordinates[1] = 1.;
    if (dim == 3)
      coordinates[2] = numbers::PI;
    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
//    const double left = -1.0, right = 1.0;
//    GridGenerator::hyper_cube(triangulation,left,right);
//    const unsigned int base_refinements = n_refinements;
    std::vector<unsigned int> refinements(dim, 1);
    //refinements[0] *= 3;
    GridGenerator::subdivided_hyper_rectangle (triangulation,
        refinements,Point<dim>(),
        coordinates);
    // set boundary indicator
//    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
//    for(;cell!=endc;++cell)
//    {
//    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
//    {
//    //  if ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12)||
//    //      (std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
//     if ((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
//        cell->face(face_number)->set_boundary_id (1);
//    }
//    }
    //periodicity in x- and z-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    triangulation.begin()->face(0)->set_all_boundary_ids(0+10);
    triangulation.begin()->face(1)->set_all_boundary_ids(1+10);
    //periodicity in z-direction, if dim==3
//    for (unsigned int face=4; face<GeometryInfo<dim>::faces_per_cell; ++face)
    triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
    triangulation.begin()->face(5)->set_all_boundary_ids(3+10);

    GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
    GridTools::collect_periodic_faces(triangulation, 2+10, 3+10, 2, periodic_faces);
//    for (unsigned int d=2; d<dim; ++d)
//      GridTools::collect_periodic_faces(triangulation, 2*d+10, 2*d+1+10, d, periodic_faces);
    triangulation.add_periodicity(periodic_faces);
    triangulation.refine_global(n_refinements);

    GridTools::transform (&grid_transform<dim>, triangulation);

    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef VORTEX
    const double left = -0.5, right = 0.5;
    GridGenerator::subdivided_hyper_cube(triangulation,2,left,right);

    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if (((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12) && (cell->face(face_number)->center()(1)<0))||
           ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12) && (cell->face(face_number)->center()(1)>0))||
           ((std::fabs(cell->face(face_number)->center()(1) - left)< 1e-12) && (cell->face(face_number)->center()(0)<0))||
           ((std::fabs(cell->face(face_number)->center()(1) - right)< 1e-12) && (cell->face(face_number)->center()(0)>0)))
          cell->face(face_number)->set_boundary_id (1);
      }
    }
    triangulation.refine_global(n_refinements);

    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef POISEUILLE
    std::vector<unsigned int> repetitions({2,1});
    Point<dim> point1(0.0,-1.0), point2(4.0,1.0);
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - 4.0)< 1e-12))
          cell->face(face_number)->set_boundary_id (1);
      }
    }
    triangulation.refine_global(n_refinements);
    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef KOVASZNAY
    const double left = -1.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
          cell->face(face_number)->set_boundary_id (1);
      }
    }
    triangulation.refine_global(n_refinements);
    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef BELTRAMI
    const double left = -1.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);
    triangulation.refine_global(n_refinements);
    dirichlet_boundary.insert(0);
#endif

#ifdef STOKES
    const double left = 0.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);
    triangulation.refine_global(n_refinements);
    dirichlet_boundary.insert(0);
#endif

#ifdef FLOW_PAST_CYLINDER
    create_triangulation(triangulation);
    triangulation.set_manifold(10, *cylinder_manifold);

    triangulation.refine_global(n_refinements);
    dirichlet_boundary.insert(0);
    dirichlet_boundary.insert(2);
    neumann_boundary.insert(1);
#endif

    pcout << std::endl << "Generating grid for " << dim << "-dimensional problem" << std::endl << std::endl
      << "  number of refinements:" << std::setw(10) << n_refinements << std::endl
      << "  number of cells:      " << std::setw(10) << triangulation.n_global_active_cells() << std::endl
      << "  number of faces:      " << std::setw(10) << triangulation.n_active_faces() << std::endl
      << "  number of vertices:   " << std::setw(10) << triangulation.n_vertices() << std::endl;

    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler_p.distribute_dofs(fe_p);
    dof_handler_xwall.distribute_dofs(fe_xwall);
    //dof_handler.distribute_mg_dofs(fe);
    dof_handler_p.distribute_mg_dofs(fe_p);
    //dof_handler_xwall.distribute_mg_dofs(fe_xwall);

    float ndofs_per_cell_velocity = pow(float(fe_degree+1),dim)*dim;
    float ndofs_per_cell_pressure = pow(float(fe_degree_p+1),dim);
    float ndofs_per_cell_xwall    = pow(float(fe_degree_xwall+1),dim)*dim;
    pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
      << "Velocity:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_velocity << std::endl
      << "  number of dofs (velocity):\t" << std::setw(10) << dof_handler.n_dofs()*dim << std::endl
      << "Pressure:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree_p << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_pressure << std::endl
      << "  number of dofs (pressure):\t" << std::setw(10) << dof_handler_p.n_dofs() << std::endl
      << "Enrichment:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree_xwall << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_xwall << std::endl
      << "  number of dofs (xwall):\t" << std::setw(10) << dof_handler_xwall.n_dofs()*dim << std::endl;
  }


  template <int dim>
  class Postprocessor : public DataPostprocessor<dim>
  {
    static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
  public:
    Postprocessor (const unsigned int partition)
      :
      partition (partition)
    {}

    virtual
    std::vector<std::string>
    get_names() const
    {
      // must be kept in sync with get_data_component_interpretation and
      // compute_derived_quantities_vector
      std::vector<std::string> solution_names (dim, "velocity");
#ifdef CHANNEL
      solution_names.push_back ("tau_w");
      for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back ("velocity_xwall");
#endif
      for (unsigned int d=0; d<number_vorticity_components; ++d)
        solution_names.push_back ("vorticity");
      solution_names.push_back ("owner");

      return solution_names;
    }

    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const
    {
#ifdef CHANNEL
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(2*dim+number_vorticity_components+2, DataComponentInterpretation::component_is_part_of_vector);
      // pressure
      interpretation[dim] = DataComponentInterpretation::component_is_scalar;
      // owner
      interpretation.back() = DataComponentInterpretation::component_is_scalar;
#else
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(dim+number_vorticity_components+1, DataComponentInterpretation::component_is_part_of_vector);
      if(dim==2)
        interpretation[dim] = DataComponentInterpretation::component_is_scalar;
      // owner
      interpretation.back() = DataComponentInterpretation::component_is_scalar;
#endif
      return interpretation;
    }

    virtual
    UpdateFlags
    get_needed_update_flags () const
    {
      return update_values | update_quadrature_points;
    }

    virtual void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &/*duh*/,
                                       const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                       const std::vector<Point<dim> >                  &/*normals*/,
                                       const std::vector<Point<dim> >                  &evaluation_points,
                                       std::vector<Vector<double> >                    &computed_quantities) const
    {
      const unsigned int n_quadrature_points = uh.size();
      Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
#ifdef CHANNEL
      Assert (uh[0].size() == 4*dim+1,                            ExcInternalError());

      for (unsigned int q=0; q<n_quadrature_points; ++q)
        {
          // TODO: fill in wall distance function
          double wdist = 0.0;
          if(evaluation_points[q][1]<0.0)
            wdist = 1.0+evaluation_points[q][1];
          else
            wdist = 1.0-evaluation_points[q][1];
          //todo: add correct utau
          const double enrichment_func = SimpleSpaldingsLaw::SpaldingsLaw(wdist,sqrt(uh[q](dim)));
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](d)
              = (uh[q](d) + uh[q](dim+1+d) * enrichment_func);

          // tau_w
          computed_quantities[q](dim) = uh[q](dim);

          // velocity_xwall
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](dim+1+d) = uh[q](dim+1+d);

          // vorticity
          for (unsigned int d=0; d<number_vorticity_components; ++d)
            computed_quantities[q](2*dim+1+d) = uh[q](2*dim+1+d)+uh[q](2*dim+number_vorticity_components+1+d)*enrichment_func;

          // owner
          computed_quantities[q](2*dim+number_vorticity_components+1) = partition;
        }
#else
      Assert (uh[0].size() == dim+number_vorticity_components,                            ExcInternalError());

      for (unsigned int q=0; q<n_quadrature_points; ++q)
        {
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q](d)
              = uh[q](d);

          // vorticity
          for (unsigned int d=0; d<number_vorticity_components; ++d)
            computed_quantities[q](dim+d) = uh[q](dim+d);
          // owner
          computed_quantities[q](dim+number_vorticity_components) = partition;
        }
#endif
    }

  private:
    const unsigned int partition;
  };


  template<int dim>
  void NavierStokesProblem<dim>::
  write_output(std::vector<parallel::distributed::Vector<value_type> >   &solution_n,
          std::vector<parallel::distributed::Vector<value_type> >   &vorticity,
          XWall<dim,fe_degree,fe_degree_xwall>* xwall,
#ifdef COMPDIV
          parallel::distributed::Vector<value_type>   &div_old,
          parallel::distributed::Vector<value_type>   &div_new,
#endif
          const unsigned int                     output_number)
  {
    const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
    // velocity + xwall dofs
    const FESystem<dim> joint_fe (fe, dim,
#ifdef CHANNEL
                                  *(*xwall).ReturnFE(), 1,
                                  fe_xwall, dim,
#endif
                                  fe, number_vorticity_components
#ifdef CHANNEL
                                  ,fe_xwall, number_vorticity_components
#endif
                                  );
    DoFHandler<dim> joint_dof_handler (dof_handler.get_triangulation());
    joint_dof_handler.distribute_dofs (joint_fe);
    IndexSet joint_relevant_set;
    DoFTools::extract_locally_relevant_dofs(joint_dof_handler, joint_relevant_set);
    parallel::distributed::Vector<double>
      joint_solution (joint_dof_handler.locally_owned_dofs(), joint_relevant_set, MPI_COMM_WORLD);
    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
      loc_vel_dof_indices (fe.dofs_per_cell)
#ifdef CHANNEL
      , loc_pre_dof_indices((*xwall).ReturnFE()->dofs_per_cell),
      loc_vel_xwall_dof_indices(fe_xwall.dofs_per_cell)
#endif
      ;
    typename DoFHandler<dim>::active_cell_iterator
      joint_cell = joint_dof_handler.begin_active(),
      joint_endc = joint_dof_handler.end(),
      vel_cell = dof_handler.begin_active()
#ifdef CHANNEL
      ,pre_cell = (*xwall).ReturnDofHandlerWallDistance()->begin_active(),
      vel_cell_xwall = dof_handler_xwall.begin_active();
      xwall->ReturnTauW()->update_ghost_values()
#endif
;

    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell
#ifdef CHANNEL
    , ++ pre_cell, ++vel_cell_xwall
#endif
    )
      if (joint_cell->is_locally_owned())
      {
        joint_cell->get_dof_indices (loc_joint_dof_indices);
        vel_cell->get_dof_indices (loc_vel_dof_indices);
#ifdef CHANNEL
        pre_cell->get_dof_indices (loc_pre_dof_indices);
        vel_cell_xwall->get_dof_indices (loc_vel_xwall_dof_indices);
#endif
        for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
          switch (joint_fe.system_to_base_index(i).first.first)
            {
            case 0: //velocity
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                solution_n[ joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
#ifdef CHANNEL
            case 1: //tauw, necessary to reconstruct velocity
              Assert (joint_fe.system_to_base_index(i).first.second == 0,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                  (*xwall->ReturnTauW())
                (loc_pre_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 2: //velocity_xwall
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                solution_n[ dim+1+joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_xwall_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
            case 3: //vorticity
#else
            case 1:
#endif
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                vorticity[ joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
#ifdef CHANNEL
            case 4: //vorticity_xwall
              Assert (joint_fe.system_to_base_index(i).first.second < dim,
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                vorticity[ dim + joint_fe.system_to_base_index(i).first.second ]
                (loc_vel_xwall_dof_indices[ joint_fe.system_to_base_index(i).second ]);
              break;
#endif
            default:
              Assert (false, ExcInternalError());
              break;
            }
      }

  joint_solution.update_ghost_values();

  Postprocessor<dim> postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

  DataOut<dim> data_out;
  data_out.attach_dof_handler(joint_dof_handler);
  data_out.add_data_vector(joint_solution, postprocessor);
#ifdef CHANNEL
  (*(*xwall).ReturnWDist()).update_ghost_values();
  data_out.add_data_vector (*(*xwall).ReturnDofHandlerWallDistance(),(*(*xwall).ReturnWDist()), "wdist");
#endif
  solution_n[dim].update_ghost_values();
  data_out.add_data_vector (dof_handler_p,solution_n[dim], "p");
//  data_out.add_data_vector (*(*xwall).ReturnDofHandlerWallDistance(),(*(*xwall).ReturnTauW()), "tauw");
#ifdef COMPDIV
  data_out.add_data_vector (dof_handler,div_old, "div_old");
  data_out.add_data_vector (dof_handler,div_new, "div_new");
#endif

    std::ostringstream filename;
    filename << "output/"
             << output_prefix
             << "_Proc"
             << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
             << "_"
             << output_number
             << ".vtu";

    data_out.build_patches (mapping,5);

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);

    if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
      {
        std::ostringstream filename;
        filename << output_prefix
                 << "_Proc"
                 << i
                 << "_"
                 << output_number
                 << ".vtu";

          filenames.push_back(filename.str().c_str());
      }
      std::string master_name = "output/" + output_prefix + "_" + Utilities::int_to_string(output_number) + ".pvtu";
      std::ofstream master_output (master_name.c_str());
      data_out.write_pvtu_record (master_output, filenames);
    }
  }

  template<int dim>
  void NavierStokesProblem<dim>::
  init_channel_statistics()
  {
    udiv_samp = 0;
    udiff_samp = 0;
    numchsamp = 0;
  }

  template<int dim>
  void NavierStokesProblem<dim>::
  compute_divu_statistics(std::vector<parallel::distributed::Vector<value_type> >   &vel_hathat, NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> & nsoperation)
  {
    numchsamp++;

    std::vector<double > dummy(4,0.0);
    nsoperation.get_data().loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divu_for_channel_stats,
    &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divu_for_channel_stats_face,
    &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divu_for_channel_stats_boundary_face,
                         &nsoperation, dummy, vel_hathat);
    double div = Utilities::MPI::sum (dummy.at(0), MPI_COMM_WORLD);
    double vol = Utilities::MPI::sum (dummy.at(1), MPI_COMM_WORLD);
    double udiff = Utilities::MPI::sum (dummy.at(2), MPI_COMM_WORLD);
    double area = Utilities::MPI::sum (dummy.at(3), MPI_COMM_WORLD);
    udiv_samp += div/vol;
    udiff_samp += udiff/area;
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::ostringstream filename;
      filename << "output/"
               << output_prefix
               << ".flow_statistics_divu";

      std::ofstream f;

      f.open(filename.str().c_str(),std::ios::trunc);
      f<<"average divergence over space and time"<<std::endl;
      f<<"number of samples:   " << numchsamp << std::endl;
      f<<"mean div u_hathat:   " << udiv_samp/numchsamp*6.0/11.0 << std::endl;//the factor 6/11 is gamma0^-1, which is the factor u_hathat is scaled compared to solution_n
      f<<"mean diff u_hathat:  " << udiff_samp/numchsamp*6.0/11.0 << std::endl;
      f.close();
    }
   }

  template<int dim>
  void NavierStokesProblem<dim>::
  write_divu(std::vector<parallel::distributed::Vector<value_type> >   &vel_hathat, NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> & nsoperation, double time, unsigned int time_step_number)
  {
    std::vector<double > dummy(4,0.0);
    nsoperation.get_data().loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divu_for_channel_stats,
    &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divu_for_channel_stats_face,
    &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divu_for_channel_stats_boundary_face,
                         &nsoperation, dummy, vel_hathat);
    double div = Utilities::MPI::sum (dummy.at(0), MPI_COMM_WORLD);
    double vol = Utilities::MPI::sum (dummy.at(1), MPI_COMM_WORLD);
    double udiff = Utilities::MPI::sum (dummy.at(2), MPI_COMM_WORLD);
    double area = Utilities::MPI::sum (dummy.at(3), MPI_COMM_WORLD);
    double udiv = div/vol;
    double udiffx = udiff/area;
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::ostringstream filename;
      filename << "output/"
               << output_prefix
               << ".flow_statistics_divu_timeseries";

      std::ofstream f;
      if(time_step_number==1)
      {
        f.open(filename.str().c_str(),std::ios::trunc);
        f<< "       n       |       t      |     divu     |  (um - up )*n" << std::endl;
      }
      else
        f.open(filename.str().c_str(),std::ios::app);
      f << std::setw(15) <<time_step_number;
      f << std::scientific<<std::setprecision(7) << std::setw(15) << time ;
      f << std::scientific<<std::setprecision(7) << std::setw(15) << udiv*6.0/11.0;
      f << std::scientific<<std::setprecision(7) << std::setw(15) << udiffx*6.0/11.0 << std::endl;
      //the factor 6/11 is gamma0^-1, which is the factor u_hathat is scaled compared to solution_n
      f.close();
    }
  }


  template<int dim>
  void NavierStokesProblem<dim>::
  calculate_error(std::vector<parallel::distributed::Vector<value_type>>  &solution_n,
                  const double                                            delta_t)
  {
  for(unsigned int d=0;d<dim;++d)
  {
    Vector<double> norm_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (mapping, dof_handler,
                       solution_n[d],
                       AnalyticalSolution<dim>(d,time+delta_t),
                       norm_per_cell,
                       QGauss<dim>(fe.degree+2),
                       VectorTools::L2_norm);
    double solution_norm =
      std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
    pcout << "error (L2-norm) velocity u" << d+1 << ":"
        << std::scientific << std::setprecision(5) << std::setw(10) << solution_norm
        << std::endl;
  }
  Vector<double> norm_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping, dof_handler_p,
                     solution_n[dim],
                     AnalyticalSolution<dim>(dim,time+delta_t),
                     norm_per_cell,
                     QGauss<dim>(fe.degree+2),
                     VectorTools::L2_norm);
  double solution_norm =
    std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
  pcout << "error (L2-norm) pressure p: "
      << std::scientific << std::setprecision(5) << std::setw(10) << solution_norm
      << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::calculate_time_step()
  {
    typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
                                                          endc = triangulation.end();

    double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();
    Tensor<1,dim, value_type> velocity;
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
      // calculate minimum diameter
      //diameter = cell->diameter()/std::sqrt(dim); // diameter is the largest diagonal -> divide by sqrt(dim)
      diameter = cell->minimum_vertex_distance();
      if (diameter < min_cell_diameter)
        min_cell_diameter = diameter;
      }
    const double global_min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, MPI_COMM_WORLD);

	  pcout << std::endl << "Temporal discretisation:" << std::endl << std::endl
			<< "  High order dual splitting scheme (3rd order)" << std::endl << std::endl
			<< "Calculation of time step size:" << std::endl << std::endl
			<< "  h_min: " << std::setw(10) << global_min_cell_diameter << std::endl
			<< "  u_max: " << std::setw(10) << MAX_VELOCITY << std::endl
			<< "  CFL:   " << std::setw(7) << CFL << "/p²" << std::endl;

    // cfl = a * time_step / d_min
    //time_step = cfl * global_min_cell_diameter / global_max_cell_a;
    time_step = cfl * global_min_cell_diameter / MAX_VELOCITY;
//    time_step = cfl * 1.0/(MAX_VELOCITY/global_min_cell_diameter+0.0/(global_min_cell_diameter*global_min_cell_diameter));

    // decrease time_step in order to exactly hit END_TIME
    time_step = (END_TIME-START_TIME)/(1+int((END_TIME-START_TIME)/time_step));
#ifdef STOKES
    time_step = TIME_STEP_SIZE; // 0.1/pow(2.0,8);
#endif
    pcout << std::endl << "time step size:\t" << std::setw(10) << time_step << std::endl;

    pcout << std::endl << "further parameters:" << std::endl;
    pcout << " - number of quad points for xwall:     " << n_q_points_1d_xwall << std::endl;
    pcout << " - viscosity:                           " << VISCOSITY << std::endl;
    pcout << " - stab_factor_pressure:                " << stab_factor_pressure << std::endl;
    pcout << " - stab_factor_viscous:                 " << stab_factor_viscous << std::endl;
    pcout << " - penalty factor K:                    " << K << std::endl;
    pcout << " - Smagorinsky constant                 " << CS << std::endl;
    pcout << " - fix tauw to 1.0:                     " << not variabletauw << std::endl;
    pcout << " - max wall distance of xwall:          " << MAX_WDIST_XWALL << std::endl;
    pcout << " - grid stretching if no xwall:         " << GRID_STRETCH_FAC << std::endl;
    pcout << " - prefix:                              " << output_prefix << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::run()
  {
  make_grid_and_dofs();

  calculate_time_step();

  NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>  navier_stokes_operation(mapping,dof_handler, dof_handler_p, dof_handler_xwall, time_step, periodic_faces, dirichlet_boundary, neumann_boundary);

  // prescribe initial conditions
  for(unsigned int d=0;d<dim;++d)
    VectorTools::interpolate(mapping, dof_handler, AnalyticalSolution<dim>(d,time), navier_stokes_operation.solution_n[d]);
  VectorTools::interpolate(mapping, dof_handler_p, AnalyticalSolution<dim>(dim,time), navier_stokes_operation.solution_n[dim]);
  for(unsigned int d=0;d<dim;++d)
    VectorTools::interpolate(mapping, dof_handler_xwall, AnalyticalSolution<dim>(d+dim+1,time), navier_stokes_operation.solution_n[d+dim+1]);

//  navier_stokes_operation.solution_nm = navier_stokes_operation.solution_n;
//  navier_stokes_operation.solution_nm2 = navier_stokes_operation.solution_n;

  for(unsigned int d=0;d<dim;++d)
      VectorTools::interpolate(mapping, dof_handler, AnalyticalSolution<dim>(d,time-time_step), navier_stokes_operation.solution_nm[d]);
    VectorTools::interpolate(mapping, dof_handler_p, AnalyticalSolution<dim>(dim,time-time_step), navier_stokes_operation.solution_nm[dim]);
  for(unsigned int d=0;d<dim;++d)
    VectorTools::interpolate(mapping, dof_handler, AnalyticalSolution<dim>(d,time-time_step*2.0), navier_stokes_operation.solution_nm2[d]);
  VectorTools::interpolate(mapping, dof_handler_p, AnalyticalSolution<dim>(dim,time-time_step*2.0), navier_stokes_operation.solution_nm2[dim]);

  // compute vorticity from initial data at time t = START_TIME
  {
    navier_stokes_operation.compute_vorticity(navier_stokes_operation.solution_n,navier_stokes_operation.vorticity_n);
//    navier_stokes_operation.compute_eddy_viscosity(navier_stokes_operation.solution_n);

//    navier_stokes_operation.vorticity_nm = navier_stokes_operation.vorticity_n;
//    navier_stokes_operation.vorticity_nm2 = navier_stokes_operation.vorticity_n;

    navier_stokes_operation.compute_vorticity(navier_stokes_operation.solution_nm,navier_stokes_operation.vorticity_nm);
    navier_stokes_operation.compute_vorticity(navier_stokes_operation.solution_nm2,navier_stokes_operation.vorticity_nm2);
  }

#ifdef CHANNEL
  StatisticsManager<dim> statistics (dof_handler, &grid_transform<dim>);
#endif

  unsigned int output_number = 0;
  const double EPSILON = 1.0e-10;
  if(OUTPUT_START_TIME < (START_TIME+EPSILON))
  {
    write_output(navier_stokes_operation.solution_n,
          navier_stokes_operation.vorticity_n,
          navier_stokes_operation.ReturnXWall(),
#ifdef COMPDIV
          navier_stokes_operation.divergence_old,
          navier_stokes_operation.divergence_new,
#endif
          output_number);
    pcout << std::endl << "Write output at START_TIME t = " << START_TIME << std::endl;
#ifndef FLOW_PAST_CYLINDER
    calculate_error(navier_stokes_operation.solution_n);
#endif
  }
  output_number++;

  unsigned int time_step_number = 1;

  init_channel_statistics();

  for(;time<(END_TIME-EPSILON)&&time_step_number<=MAX_NUM_STEPS;time+=time_step,++time_step_number)
  {
    navier_stokes_operation.do_timestep(time,time_step,time_step_number);

    if(time_step_number % output_solver_info_every_timesteps ==0)
      pcout << "Step = " << time_step_number << "  t = " << time << std::endl;

    if( (time+time_step-START_TIME) > (output_number*OUTPUT_INTERVAL_TIME-EPSILON) && (time+time_step) > OUTPUT_START_TIME-EPSILON)
    {
//      std::vector<parallel::distributed::Vector<value_type>> output_vector;
//      output_vector.resize(2*dim+1);
//      for(unsigned int d=0;d<dim;++d)
//      {
//        output_vector[d].reinit(navier_stokes_operation.velocity_temp[d]);
//        output_vector[d] = navier_stokes_operation.velocity_temp[d];
//        output_vector[d+dim+1].reinit(navier_stokes_operation.velocity_temp[d]);
//        output_vector[d+dim+1] = navier_stokes_operation.velocity_temp[d];
//      }
//      output_vector[dim] = navier_stokes_operation.solution_n[dim];
//      write_output(output_vector,
//              navier_stokes_operation.ReturnXWall(),
//  #ifdef COMPDIV
//              navier_stokes_operation.divergence_old,
//              navier_stokes_operation.divergence_new,
//  #endif
//              output_number++);

    write_output(navier_stokes_operation.solution_n,
            navier_stokes_operation.vorticity_n,
            navier_stokes_operation.ReturnXWall(),
#ifdef COMPDIV
            navier_stokes_operation.divergence_old,
            navier_stokes_operation.divergence_new,
#endif
            output_number++);
      pcout << std::endl << "Write output at TIME t = " << time+time_step << std::endl;
      calculate_error(navier_stokes_operation.solution_n,time_step);
    }
    else if((time+time_step-START_TIME) > (output_number*OUTPUT_INTERVAL_TIME-EPSILON))
      output_number++;
    if((time+time_step) > STATISTICS_START_TIME-EPSILON && time_step_number % STATISTICS_EVERY == 0)
    {
#ifdef CHANNEL
      statistics.evaluate(navier_stokes_operation.solution_n);
      if(time_step_number % 100 == 0||(time+time_step) > (END_TIME-EPSILON))
        statistics.write_output(output_prefix,VISCOSITY);
#endif
      compute_divu_statistics(navier_stokes_operation.velocity_temp, navier_stokes_operation);
    }
    if(DIVU_TIMESERIES)
      write_divu(navier_stokes_operation.velocity_temp, navier_stokes_operation,time+time_step,time_step_number);
  }
  navier_stokes_operation.analyse_computing_times();
  }
}

int main (int argc, char** argv)
{
  try
    {
      using namespace DG_NavierStokes;
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

      deallog.depth_console(0);

      for(unsigned int refine_steps = refine_steps_min;refine_steps <= refine_steps_max;++refine_steps)
      {
        NavierStokesProblem<dimension> navier_stokes_problem(refine_steps);
        navier_stokes_problem.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
