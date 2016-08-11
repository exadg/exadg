//---------------------------------------------------------------------------
//    $Id: program.cc 56 2015-02-06 13:05:10Z kronbichler $
//    Version: $Name$
//
//    Copyright (C) 2013 - 2015 by Katharina Kormann and Martin Kronbichler
//
//---------------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/integrators/laplace.h>

#include <fstream>
#include <sstream>

#include "../include/InputParameters.h"

#include "../include/DGConvDiffOperation.h"
#include "../include/TimeIntExplRKConvDiff.h"

#include "../include/BoundaryDescriptorConvDiff.h"
#include "../include/FieldFunctionsConvDiff.h"

//#define PROPAGATING_SINE_WAVE
//#define ROTATING_HILL
#define DEFORMING_HILL
//#define DIFFUSIVE_PROBLEM_HOMOGENEOUS_DBC
//#define DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC
//#define DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC_2
//#define PROBLEM_WITH_CONSTANT_RHS
//#define BOUNDARY_LAYER_PROBLEM

  using namespace dealii;

#ifdef PROPAGATING_SINE_WAVE
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::Convection;
  bool const RIGHT_HAND_SIDE = false;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 2;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 3;
  const unsigned int REFINE_STEPS_SPACE_MAX = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 8.0;

  const double CFL_NUMBER = 0.2;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;

  const double DIFFUSIVITY = 0.0;

  const double stab_factor = 1.0;
#endif

#ifdef ROTATING_HILL
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::Convection;
  bool const RIGHT_HAND_SIDE = false;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 5;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 4;
  const unsigned int REFINE_STEPS_SPACE_MAX = 4;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;

  const double CFL_NUMBER = 0.2;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;

  const double DIFFUSIVITY = 0.0;

  const double stab_factor = 1.0;
#endif

#ifdef DEFORMING_HILL
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::Convection;
  bool const RIGHT_HAND_SIDE = false;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 3;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 2;
  const unsigned int REFINE_STEPS_SPACE_MAX = 5;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0; //increase END_TIME for larger deformations of the hill

  const double CFL_NUMBER = 0.2;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME);// /20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = END_TIME-START_TIME;//analytical solution only available at t=START_TIME and t=END_TIME

  const double DIFFUSIVITY = 0.0;

  const double stab_factor = 1.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_DBC
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::Diffusion;
  bool const RIGHT_HAND_SIDE = false;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 3;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 1;
  const unsigned int REFINE_STEPS_SPACE_MAX = 4;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;

  const double CFL_NUMBER = 0.1;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME);///20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = (END_TIME-START_TIME);///20;

  const double DIFFUSIVITY = 0.1;//1.0;

  const double stab_factor = 1.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::Diffusion;
  bool const RIGHT_HAND_SIDE = true;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 2;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 3;
  const unsigned int REFINE_STEPS_SPACE_MAX = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;

  const double CFL_NUMBER = 0.2;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double DIFFUSIVITY = 0.1;//1.0;

  const double stab_factor = 1.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC_2
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::Diffusion;
  bool const RIGHT_HAND_SIDE = false;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 2;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 3;
  const unsigned int REFINE_STEPS_SPACE_MAX = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;

  const double CFL_NUMBER = 0.2;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double DIFFUSIVITY = 0.2;//1.0;

  const double stab_factor = 1.0;
#endif

#ifdef PROBLEM_WITH_CONSTANT_RHS
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::Diffusion;
  bool const RIGHT_HAND_SIDE = true;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 2;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 3;
  const unsigned int REFINE_STEPS_SPACE_MAX = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;

  const double CFL_NUMBER = 0.2;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double DIFFUSIVITY = 1.0;//1.0;

  const double stab_factor = 1.0;
#endif

#ifdef BOUNDARY_LAYER_PROBLEM
  EquationTypeConvDiff const EQUATION_TYPE = EquationTypeConvDiff::ConvectionDiffusion;
  bool const RIGHT_HAND_SIDE = false;
  bool const RUNTIME_OPTIMIZATION = false;

  const unsigned int FE_DEGREE = 2;
  const unsigned int DIMENSION = 2; // dimension >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 4;
  const unsigned int REFINE_STEPS_SPACE_MAX = 4;

  const double START_TIME = 0.0;
  const double END_TIME = 8.0;

  const double CFL_NUMBER = 0.2;
  const double DIFFUSION_NUMBER = 0.01;

  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double OUTPUT_START_TIME = START_TIME;
  const double OUTPUT_INTERVAL_TIME = (END_TIME-START_TIME)/20;

  const double ERROR_CALC_START_TIME = START_TIME;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;

  const double DIFFUSIVITY = 0.25; //0.0025

  const double stab_factor = 1.0;
#endif

  template<int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    AnalyticalSolution (const unsigned int 	n_components = 1,
			  	  	          const double 			  time = 0.)
      :
      Function<dim>(n_components, time)
    {}

    virtual ~AnalyticalSolution(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();
    double result = 1.0;

#ifdef PROPAGATING_SINE_WAVE
    // test case for purely convective problem
    // sine wave that is advected from left to right by a constant velocity field
    result = std::sin(numbers::PI*(p[0]-t));
#endif

#ifdef ROTATING_HILL
    double radius = 0.5;
    double omega = 2.0*numbers::PI;
    double center_x = -radius*std::sin(omega*t);
    double center_y = +radius*std::cos(omega*t);
    result = std::exp(-50*pow(p[0]-center_x,2.0)-50*pow(p[1]-center_y,2.0));
#endif

#ifdef DEFORMING_HILL
    double center_x = 0.5;
    double center_y = 0.75;
    double factor = 50.0;
    result = std::exp(-factor*(pow(p[0]-center_x,2.0)+pow(p[1]-center_y,2.0)));
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_DBC
    for(int d=0;d<dim;d++)
      result *= std::cos(p[d]*numbers::PI/2.0);
    result *= std::exp(-0.5*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC
    // diffusive problem with pure Neumann boundary conditions (homogeneous)
    // method of manufactured solutions:
    // prescribe analytical solution and choose right hand side f such that the residual is equal to zero
    for(int d=0;d<dim;d++)
      result *= std::cos(p[d]*numbers::PI)+1.0;
    result *= std::exp(-2.0*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC_2
    // test case with homogeneous Neumann-BC
  for(int d=0;d<dim;d++)
    result *= std::cos(p[d]*numbers::PI);
  result *= std::exp(-2.0*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
#endif

#ifdef PROBLEM_WITH_CONSTANT_RHS
    result = t;
#endif

#ifdef BOUNDARY_LAYER_PROBLEM
    // prescribe value of phi at left and right boundary
    // neumann boundaries at upper and lower boundary
    // use constant advection velocity from left to right -> boundary layer
    double phi_l = 1.0 , phi_r = 0.0;
    double U = 1.0, L = 2.0;
    double Pe = U*L/DIFFUSIVITY;
    if(t<(START_TIME + 1.0e-8))
      result = phi_l + (phi_r-phi_l)*(0.5+p[0]/L); //0.5*(1.0 - p[0]);
    else
      result = phi_l + (phi_r-phi_l)*(std::exp(Pe*p[0]/L)-std::exp(-Pe/2.0))/(std::exp(Pe/2.0)-std::exp(-Pe/2.0));
#endif

    return result;
  }

  template<int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide (const unsigned int 	n_components = 1,
			  	  	     const double 			  time = 0.)
      :
      Function<dim>(n_components, time)
    {}

    virtual ~RightHandSide(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double RightHandSide<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double t = this->get_time();

#ifdef PROPAGATING_SINE_WAVE
    double result = 0.0;
#endif

#ifdef ROTATING_HILL
    double result = 0.0;
#endif

#ifdef DEFORMING_HILL
    double result = 0.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_DBC
    double result = 0.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC
    // method of manufactured solutions:
    // prescribe analytical solution and choose right hand side f such that the residual is equal to zero
    double result = 0.0;
    for(int d=0;d<dim;++d)
    	result += std::cos(p[d]*numbers::PI)+1;
    result *= - std::pow(numbers::PI,2.0) * DIFFUSIVITY * std::exp(-2.0*DIFFUSIVITY*pow(numbers::PI,2.0)*t);
#endif


#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC_2
    double result = 0.0;
#endif

#ifdef PROBLEM_WITH_CONSTANT_RHS
    double result = 1.0;
#endif

#ifdef BOUNDARY_LAYER_PROBLEM
    double result = 0.0;
#endif

    return result;
  }

  template<int dim>
  class NeumannBoundary : public Function<dim>
  {
  public:
    NeumannBoundary (const unsigned int n_components = 1,
					           const double 			time = 0.)
      :
      Function<dim>(n_components, time)
    {}

    virtual ~NeumannBoundary(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double NeumannBoundary<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
#ifdef PROPAGATING_SINE_WAVE
    double result = 0.0;
#endif

#ifdef ROTATING_HILL
    double result = 0.0;
#endif

#ifdef DEFORMING_HILL
    double result = 0.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_DBC
    double result = 0.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC
    double result = 0.0;
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC_2
    double result = 0.0;
#endif

#ifdef PROBLEM_WITH_CONSTANT_RHS
    double result = 0.0;
#endif

#ifdef BOUNDARY_LAYER_PROBLEM
    double result = 0.0;

//    double right = 1.0;
//    if( fabs(p[0]-right)<1.0e-12 )
//      result = -3.0;
#endif
    return result;
  }

  template<int dim>
  class VelocityField : public Function<dim>
  {
  public:
    VelocityField (const unsigned int n_components = dim,
                   const double       time = 0.)
      :
      Function<dim>(n_components, time)
    {}

    virtual ~VelocityField(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double VelocityField<dim>::value(const Point<dim> &point,const unsigned int component) const
  {
    double value = 0.0;

    double t = this->get_time();
    (void)t;

#ifdef PROPAGATING_SINE_WAVE
    if(component == 0)
      value = 1.0;
#endif

#ifdef ROTATING_HILL
    if(component == 0)
      value = -point[1]*2.0*numbers::PI;
    else if(component ==1)
      value =  point[0]*2.0*numbers::PI;
#endif

#ifdef DEFORMING_HILL
    if(component == 0)
      value =  4.0 * std::sin(numbers::PI*point[0])*std::sin(numbers::PI*point[0])*std::sin(numbers::PI*point[1])*std::cos(numbers::PI*point[1])*std::cos(numbers::PI*t/END_TIME);
    else if(component ==1)
      value = -4.0 * std::sin(numbers::PI*point[0])*std::cos(numbers::PI*point[0])*std::sin(numbers::PI*point[1])*std::sin(numbers::PI*point[1])*std::cos(numbers::PI*t/END_TIME);
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_DBC
    // do nothing
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC
    // do nothing
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC_2
    // do nothing
#endif

#ifdef PROBLEM_WITH_CONSTANT_RHS
    // do nothing
#endif

#ifdef BOUNDARY_LAYER_PROBLEM
    if(component == 0)
      value = 1.0;
#endif

    return value;
  }

  InputParametersConvDiff::InputParametersConvDiff()
    :
    equation_type(EQUATION_TYPE),
    right_hand_side(RIGHT_HAND_SIDE),
    runtime_optimization(RUNTIME_OPTIMIZATION),
    start_time(START_TIME),
    end_time(END_TIME),
    order_time_integrator(4), //TODO
    cfl_number(CFL_NUMBER),
    diffusion_number(DIFFUSION_NUMBER),
    output_start_time(OUTPUT_START_TIME),
    output_interval_time(OUTPUT_INTERVAL_TIME),
    error_calc_start_time(ERROR_CALC_START_TIME),
    error_calc_interval_time(ERROR_CALC_INTERVAL_TIME),
    diffusivity(DIFFUSIVITY),
    IP_factor(1.0)//TODO
  {}

template<int dim>
class PostProcessor
{
public:
  PostProcessor(std_cxx11::shared_ptr< const DGConvDiffOperation<dim, FE_DEGREE, double> >  conv_diff_operation_in,
                InputParametersConvDiff const &param_in)
    :
    conv_diff_operation(conv_diff_operation_in),
    param(param_in),
    output_counter(0),
    error_counter(0)
  {}

  void do_postprocessing(parallel::distributed::Vector<double> const &solution,
                         double const                                time);

private:
  void calculate_error(parallel::distributed::Vector<double> const &solution,
                       double const                                time) const;

  void write_output(parallel::distributed::Vector<double> const &solution,
                    double const                                time) const;

  std_cxx11::shared_ptr< const DGConvDiffOperation<dim, FE_DEGREE, double> >  conv_diff_operation;
  InputParametersConvDiff const & param;

  unsigned int output_counter;
  unsigned int error_counter;
};

template<int dim>
void PostProcessor<dim>::do_postprocessing(parallel::distributed::Vector<double> const &solution_vector,
                                           double const                                time)
{
  const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
  if( time > (param.error_calc_start_time + error_counter*param.error_calc_interval_time - EPSILON))
  {
    calculate_error(solution_vector,time);
    ++error_counter;
  }
  if( time > (param.output_start_time + output_counter*param.output_interval_time - EPSILON))
  {
    write_output(solution_vector,time);
    ++output_counter;
  }
}

template<int dim>
void PostProcessor<dim>::write_output(parallel::distributed::Vector<double> const &solution_vector,
                                      double const                                time) const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4) << time << std::endl;

  DataOut<dim> data_out;

  data_out.attach_dof_handler (conv_diff_operation->get_data().get_dof_handler());
  data_out.add_data_vector (solution_vector, "solution");
  data_out.build_patches (4);

  const std::string filename = "output_conv_diff/solution_" + Utilities::int_to_string (output_counter, 3);

  std::ofstream output_data ((filename + ".vtu").c_str());
  data_out.write_vtu (output_data);
}

template<int dim>
void PostProcessor<dim>::calculate_error(parallel::distributed::Vector<double> const &solution_vector,
                                         double const                                time) const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Calculate error at time t = " << std::scientific << std::setprecision(4) << time << ":" << std::endl;

  Vector<double> norm_per_cell (conv_diff_operation->get_data().get_dof_handler().get_triangulation().n_active_cells());
  VectorTools::integrate_difference (conv_diff_operation->get_data().get_dof_handler(),
                                     solution_vector,
                                     AnalyticalSolution<dim>(1,time),
                                     norm_per_cell,
                                     QGauss<dim>(conv_diff_operation->get_data().get_dof_handler().get_fe().degree+2),
                                     VectorTools::L2_norm);

  double solution_norm = std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));

  pcout << std::endl << "error (L2-norm): "
        << std::setprecision(5) << std::setw(10) << solution_norm
        << std::endl;
}

template<int dim>
class ConvDiffProblem
{
public:
  typedef double value_type;
  ConvDiffProblem(const unsigned int n_refine_space, const unsigned int n_refine_time);
  void solve_problem();

private:
  void create_grid();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;

  InputParametersConvDiff param;

  const unsigned int n_refine_space;
  const unsigned int n_refine_time;

  std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> > field_functions;
  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor;

  std_cxx11::shared_ptr<DGConvDiffOperation<dim,FE_DEGREE, value_type> > conv_diff_operation;
  std_cxx11::shared_ptr<PostProcessor<dim> > postprocessor;
  std_cxx11::shared_ptr<TimeIntExplRKConvDiff<dim, FE_DEGREE, value_type> > time_integrator;
};

template<int dim>
ConvDiffProblem<dim>::ConvDiffProblem(const unsigned int n_refine_space_in,
                                      const unsigned int n_refine_time_in)
  :
  pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(n_refine_space_in),
  n_refine_time(n_refine_time_in)
{
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                     unsteady convection-diffusion equation                      " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;

  // initialize functions (analytical solution, rhs, boundary conditions)
  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>(1,param.start_time));

  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>(1,param.start_time));

  std_cxx11::shared_ptr<Function<dim> > velocity;
  velocity.reset(new VelocityField<dim>(dim,param.start_time));

  field_functions.reset(new FieldFunctionsConvDiff<dim>());
  field_functions->analytical_solution = analytical_solution;
  field_functions->right_hand_side = right_hand_side;
  field_functions->velocity = velocity;

  boundary_descriptor.reset(new BoundaryDescriptorConvDiff<dim>());

  // initialize convection diffusion operation
  conv_diff_operation.reset(new DGConvDiffOperation<dim, FE_DEGREE, value_type>(triangulation,param));

  // initialize postprocessor
  postprocessor.reset(new PostProcessor<dim>(conv_diff_operation,param));

  // initialize time integrator
  time_integrator.reset(new TimeIntExplRKConvDiff<dim, FE_DEGREE, value_type>(conv_diff_operation,postprocessor,param,velocity,n_refine_time));
}

template<int dim>
void ConvDiffProblem<dim>::create_grid()
{
#ifdef PROPAGATING_SINE_WAVE
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);

  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      //use outflow boundary at right boundary
      if ((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12))
       cell->face(face_number)->set_boundary_id(1);
    }
  }
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>(1,param.start_time));
  boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(0,analytical_solution));
  std_cxx11::shared_ptr<Function<dim> > neumann_bc;
  neumann_bc.reset(new NeumannBoundary<dim>(1,param.start_time));
  boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(1,neumann_bc));
#endif

#ifdef ROTATING_HILL
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>(1,param.start_time));
  //problem with pure Dirichlet boundary conditions
  boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(0,analytical_solution));
#endif

#ifdef DEFORMING_HILL
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = 0.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>(1,param.start_time));
  //problem with pure Dirichlet boundary conditions
  boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(0,analytical_solution));
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_DBC
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);

  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>(1,param.start_time));
  //problem with pure Dirichlet boundary conditions
  boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(0,analytical_solution));
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      // apply Neumann BC on all boundaries
      if ((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12) ||
          (std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12))
       cell->face(face_number)->set_boundary_id(1);
    }
  }
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > neumann_bc;
  neumann_bc.reset(new NeumannBoundary<dim>(1,param.start_time));
  boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(1,neumann_bc));
#endif

#ifdef DIFFUSIVE_PROBLEM_HOMOGENEOUS_NBC_2
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      // apply Neumann BC on all boundaries
      if ((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12) ||
          (std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12))
       cell->face(face_number)->set_boundary_id(1);
    }
  }
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > neumann_bc;
  neumann_bc.reset(new NeumannBoundary<dim>(1,param.start_time));
  boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(1,neumann_bc));
#endif

#ifdef PROBLEM_WITH_CONSTANT_RHS
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      // apply Neumann BC on all boundaries
      if ((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12) ||
          (std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12)||
          (std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12))
       cell->face(face_number)->set_boundary_id(1);
    }
  }
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > neumann_bc;
  neumann_bc.reset(new NeumannBoundary<dim>(1,param.start_time));
  boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(1,neumann_bc));
#endif

#ifdef BOUNDARY_LAYER_PROBLEM
  // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);

  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      if ((std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12)||
         (std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12)
        // || (std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12) // Neumann BC at right boundary
         )
        cell->face(face_number)->set_boundary_id (1);
    }
  }
  triangulation.refine_global(n_refine_space);

  std_cxx11::shared_ptr<Function<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolution<dim>(1,param.start_time));
  boundary_descriptor->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(0,analytical_solution));
  std_cxx11::shared_ptr<Function<dim> > neumann_bc;
  neumann_bc.reset(new NeumannBoundary<dim>(1,param.start_time));
  boundary_descriptor->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >(1,neumann_bc));
#endif

  pcout << std::endl << "Generating grid for " << dim << "-dimensional problem" << std::endl << std::endl
    << "  number of refinements:\t" << std::setw(10) << n_refine_space << std::endl
    << "  number of cells:\t" << std::setw(10) << triangulation.n_global_active_cells() << std::endl
    << "  number of faces:\t" << std::setw(10) << triangulation.n_active_faces() << std::endl
    << "  number of vertices:\t" << std::setw(10) << triangulation.n_vertices() << std::endl;
}

template<int dim>
void ConvDiffProblem<dim>::solve_problem()
{
  create_grid();

  conv_diff_operation->setup(boundary_descriptor,field_functions);

  time_integrator->setup();

  time_integrator->timeloop();
}

int main (int argc, char** argv)
{
  try
  {
    //using namespace ConvectionDiffusionProblem;
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    //mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN; refine_steps_space <= REFINE_STEPS_SPACE_MAX; ++refine_steps_space)
    {
      //time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;refine_steps_time <= REFINE_STEPS_TIME_MAX;++refine_steps_time)
      {
        ConvDiffProblem<DIMENSION> conv_diff_problem(refine_steps_space,refine_steps_time);
        conv_diff_problem.solve_problem();
      }
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
