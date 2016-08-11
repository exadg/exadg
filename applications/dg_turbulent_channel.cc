
// Navier-Stokes splitting program
// authors: Niklas Fehn, Benjamin Krank, Martin Kronbichler, LNM
// years: 2015-2016
// application to turbulent channel flow

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

#include "../include/DGNavierStokesDualSplitting.h"
#include "../include/DGNavierStokesDualSplittingXWall.h"
#include "../include/DGNavierStokesCoupled.h"

#include "InputParameters.h"
#include "TimeIntBDFDualSplitting.h"
#include "TimeIntBDFDualSplittingXWall.h"
#include "TimeIntBDFDualSplittingXWallSpalartAllmaras.h"
#include "TimeIntBDFCoupled.h"
#include "PostProcessor.h"
#include "PostProcessorXWall.h"
#include "PrintInputParameters.h"

const unsigned int FE_DEGREE = 5;
const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
const unsigned int FE_DEGREE_XWALL = 1;
const unsigned int N_Q_POINTS_1D_XWALL = 25;
const unsigned int DIMENSION = 2; // DIMENSION >= 2
const unsigned int REFINE_STEPS_SPACE_MIN = 3;
const unsigned int REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;
const double GRID_STRETCH_FAC = 0.001;

void InputParameters::set_input_parameters()
{
  output_prefix = "ch395_l3_k5k1_gt0_sa";
  cfl = 0.1;
  viscosity = 1./395.;

  //xwall
  variabletauw = true;
  dtauw = 1.;
  ml = 1.;
  max_wdist_xwall = 0.25;

  //dual splitting scheme
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  projection_type = ProjectionType::DivergencePenalty;
  order_time_integrator = 3;

  //xwall specific
  spatial_discretization = SpatialDiscretization::DGXWall;
  IP_formulation_viscous = InteriorPenaltyFormulation::NIPG;
  solver_viscous = SolverViscous::GMRES;

  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation; //also default
  divu_integrated_by_parts = true;
  gradp_integrated_by_parts = true;
  pure_dirichlet_bc = true;
  output_solver_info_every_timesteps = 1e1;

  end_time = 50.;
  output_start_time = 0.;
  output_interval_time = 1.;
  statistics_start_time = 49.;
  restart_interval_step = 1e9;
  restart_interval_time = 1.e9;

  max_velocity = 15.;

  //solver tolerances
  rel_tol_pressure = 1.e-4;
  rel_tol_projection = 1.e-5;
  rel_tol_viscous = 1.e-4;

}

  template<int dim>
  class AnalyticalSolutionVelocity : public Function<dim>
  {
  public:
    AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                                const double        time = 0.)
      :
      Function<dim>(n_components, time)
    {}

    virtual ~AnalyticalSolutionVelocity(){};

    virtual double value (const Point<dim>    &p,
                          const unsigned int  component = 0) const;
  };

  template<int dim>
  double AnalyticalSolutionVelocity<dim>::value(const Point<dim>   &p,
                                                const unsigned int component) const
  {
    double t = this->get_time();
    double result = 0.0;
    (void)t;

    if(p[1]<0.9999 && p[1]>-0.9999)
    {
      if(dim==3)
      {
        if(component == 0)
          result = -22.0*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-1.0)*0.5-2./22.*std::sin(p[2]*8.));//*1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0*(t<T? (t/T) : 1.0);
        else if(component ==2)
          result = (pow(p[1],6.0)-1.0)*std::sin(p[0]*8.)*2.;
      }
      else if(component == 0)
        result = -23.0*(pow(p[1],6.0)-1.0);
    }

    return result;
  }

  template<int dim>
  class AnalyticalSolutionPressure : public Function<dim>
  {
  public:
    AnalyticalSolutionPressure (const double time = 0.)
      :
      Function<dim>(1 /*n_components*/, time)
    {}

    virtual ~AnalyticalSolutionPressure(){};

    virtual double value (const Point<dim>   &p,
                          const unsigned int component = 0) const;
  };

  template<int dim>
  double AnalyticalSolutionPressure<dim>::value(const Point<dim>    &,
                                                const unsigned int  /* component */) const
  {
    double result = 0.;
    return result;
  }

  template<int dim>
  class NeumannBoundaryVelocity : public Function<dim>
  {
  public:
    NeumannBoundaryVelocity (const double time = 0.)
      :
      Function<dim>(dim, time)
    {}

    virtual ~NeumannBoundaryVelocity(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double NeumannBoundaryVelocity<dim>::value(const Point<dim> &,const unsigned int ) const
  {
    return 0.;
  }

  template<int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide (const double time = 0.)
      :
      Function<dim>(dim, time)
    {}

    virtual ~RightHandSide(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

    void setup(const double* , double , double );
  };


  template<int dim>
  void RightHandSide<dim>::setup(const double* , double , double )
  {
  }

  template<int dim>
  double RightHandSide<dim>::value(const Point<dim> &,const unsigned int component) const
  {
    //channel flow with periodic bc
    if(component==0)
      return 1.0;
    else
      return 0.0;

  return 0.;
  }

  template<int dim>
  class PressureBC_dudt : public Function<dim>
  {
  public:
    PressureBC_dudt (const double time = 0.)
      :
      Function<dim>(dim, time)
    {}

    virtual ~PressureBC_dudt(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double PressureBC_dudt<dim>::value(const Point<dim> &,const unsigned int ) const
  {
  double result = 0.0;
  return result;
  }

  template <int dim>
  Point<dim> grid_transform (const Point<dim> &in);

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  class PostProcessorChannel: public PostProcessor<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>
  {
  public:

    PostProcessorChannel(
                  std_cxx11::shared_ptr< const DGNavierStokesBase<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall> >  ns_operation,
                  InputParameters const &param_in):
      PostProcessor<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>(ns_operation,param_in),
      statistics_ch(ns_operation->get_dof_handler_u())
    {

    }

    virtual ~PostProcessorChannel(){}

    void setup()
    {
      PostProcessor<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>::setup();
      statistics_ch.setup(&grid_transform<dim>);
    }

    virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                           parallel::distributed::Vector<double> const &pressure,
                           parallel::distributed::Vector<double> const &vorticity,
                           parallel::distributed::Vector<double> const &divergence,
                           double const time,
                           unsigned int const time_step_number)
    {
      PostProcessor<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>::do_postprocessing(velocity,pressure,vorticity,divergence,time,time_step_number);
      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

      if(time > this->param.statistics_start_time-EPSILON && time_step_number % this->param.statistics_every == 0)
      {
        statistics_ch.evaluate(velocity);
        if(time_step_number % 100 == 0 || time > (this->param.end_time-EPSILON))
          statistics_ch.write_output(this->param.output_prefix,this->ns_operation_->get_viscosity());
      }
    };

  protected:
    StatisticsManager<dim> statistics_ch;

  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  class PostProcessorChannelXWall: public PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>
  {
  public:

    PostProcessorChannelXWall(
                  std_cxx11::shared_ptr< DGNavierStokesBase<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall> >  ns_operation,
                  InputParameters const &param_in):
      PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>(ns_operation,param_in),
      statistics_ch(ns_operation->get_dof_handler_u())
    {
    }

    virtual ~PostProcessorChannelXWall(){}

    void setup()
    {
      PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>::setup();
      statistics_ch.setup(&grid_transform<dim>);
    }

    virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                           parallel::distributed::Vector<double> const &pressure,
                           parallel::distributed::Vector<double> const &vorticity,
                           parallel::distributed::Vector<double> const &divergence,
                           double const time,
                           unsigned int const time_step_number)
    {
      PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall>::do_postprocessing(velocity,pressure,vorticity,divergence,time,time_step_number);
      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

      if(time > this->param.statistics_start_time-EPSILON && time_step_number % this->param.statistics_every == 0)
      {
        this->statistics_ch.evaluate_xwall(velocity,
                                        this->ns_operation_xw_->get_dof_handler_wdist(),
                                        this->ns_operation_xw_->get_fe_parameters(),
                                        this->ns_operation_xw_->get_viscosity());
        if(time_step_number % 100 == 0 || time > (this->param.end_time-EPSILON))
          this->statistics_ch.write_output(this->param.output_prefix,this->ns_operation_->get_viscosity());
      }
    };
  protected:
    StatisticsManager<dim> statistics_ch;

  };

  template<int dim>
  class NavierStokesProblem
  {
  public:
    typedef typename DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>::value_type value_type;
    NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time=0);
    void solve_problem(bool do_restart);

  private:
    void create_grid();

    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

    const unsigned int n_refine_space;

    std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity;
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure;

    InputParameters param;

    std_cxx11::shared_ptr<DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> > navier_stokes_operation;

    std_cxx11::shared_ptr<PostProcessor<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> > postprocessor;

    std_cxx11::shared_ptr<TimeIntBDF<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type> > time_integrator;
  };

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time):
  pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space)
  {
    PrintInputParams::Header(pcout);

    param.set_input_parameters();
    param.check_parameters();

    // initialize functions (analytical solution, rhs, boundary conditions)
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
    analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(dim,param.start_time));
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
    analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(param.start_time));

    std_cxx11::shared_ptr<Function<dim> > right_hand_side;
    right_hand_side.reset(new RightHandSide<dim>(param.start_time));

    field_functions.reset(new FieldFunctionsNavierStokes<dim>());
    field_functions->analytical_solution_velocity = analytical_solution_velocity;
    field_functions->analytical_solution_pressure = analytical_solution_pressure;
    field_functions->right_hand_side = right_hand_side;

    boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokes<dim>());
    boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokes<dim>());

    if(param.spatial_discretization == SpatialDiscretization::DGXWall)
    {
      analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(2*dim,param.start_time));
      field_functions->analytical_solution_velocity = analytical_solution_velocity;
      if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessorChannelXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,param));
        // initialize time integrator that depends on both navier_stokes_operation and postprocessor
        time_integrator.reset(new TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
            navier_stokes_operation,postprocessor,param,refine_steps_time));
      }
      else
      {
        AssertThrow(false,ExcMessage("XWall only implemented for the unsteady DualSplitting case up to now"));
      }
    }
    else if(param.spatial_discretization == SpatialDiscretization::DG)
    {
      if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesDualSplitting<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessorChannel<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,param));
        // initialize time integrator that depends on both navier_stokes_operation and postprocessor
        time_integrator.reset(new TimeIntBDFDualSplitting<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
            navier_stokes_operation,postprocessor,param,refine_steps_time));
      }
      else if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessorChannel<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,param));
        // initialize time integrator that depends on both navier_stokes_operation and postprocessor
        time_integrator.reset(new TimeIntBDFCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
            navier_stokes_operation,postprocessor,param,refine_steps_time));
      }
    }
  }

  template <int dim>
  Point<dim> grid_transform (const Point<dim> &in)
  {
    Point<dim> out = in;

    out[0] = in(0)-numbers::PI;
    out[1] =  std::tanh(GRID_STRETCH_FAC*(2.*in(1)-1.))/std::tanh(GRID_STRETCH_FAC);
    if(dim==3)
      out[2] = in(2)-0.5*numbers::PI;
    return out;
  }

  template<int dim>
  void NavierStokesProblem<dim>::create_grid ()
  {
    /* --------------- Generate grid ------------------- */
    //turbulent channel flow
    Point<dim> coordinates;
    coordinates[0] = 2.0*numbers::PI;
    coordinates[1] = 1.0;
    if (dim == 3)
      coordinates[2] = numbers::PI;
    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    std::vector<unsigned int> refinements(dim, 1);
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);

    //periodicity in x- and z-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    triangulation.begin()->face(0)->set_all_boundary_ids(0+10);
    triangulation.begin()->face(1)->set_all_boundary_ids(1+10);
    //periodicity in z-direction
    if (dim == 3)
    {
      triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
      triangulation.begin()->face(5)->set_all_boundary_ids(3+10);
    }

    GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
    if (dim == 3)
      GridTools::collect_periodic_faces(triangulation, 2+10, 3+10, 2, periodic_faces);

    triangulation.add_periodicity(periodic_faces);
    triangulation.refine_global(n_refine_space);

    GridTools::transform (&grid_transform<dim>, triangulation);

    // fill boundary descriptor velocity
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
    analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(dim,param.start_time));
    // Dirichlet boundaries: ID = 0
    boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                      (0,analytical_solution_velocity));

    std_cxx11::shared_ptr<Function<dim> > neumann_bc_velocity;
    neumann_bc_velocity.reset(new NeumannBoundaryVelocity<dim>(param.start_time));
    // Neumann boundaris: ID = 1
    boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (1,neumann_bc_velocity));

    // fill boundary descriptor pressure
    std_cxx11::shared_ptr<Function<dim> > pressure_bc_dudt;
    pressure_bc_dudt.reset(new PressureBC_dudt<dim>(param.start_time));
    // Dirichlet boundaries: ID = 0
    boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                      (0,pressure_bc_dudt));

    std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
    analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(param.start_time));
    // Neumann boundaries: ID = 1
    boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (1,analytical_solution_pressure));

    PrintInputParams::print_spatial_discretization(pcout,dim,n_refine_space,
        triangulation.n_global_active_cells(),triangulation.n_active_faces(),triangulation.n_vertices() );
  }

template<int dim>
void NavierStokesProblem<dim>::solve_problem(bool do_restart)
{
  create_grid();

  navier_stokes_operation->setup(periodic_faces,
                                 boundary_descriptor_velocity,
                                 boundary_descriptor_pressure,
                                 field_functions);

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  time_integrator->setup(do_restart);

  navier_stokes_operation->setup_solvers();

  PrintInputParams::print_solver_parameters(pcout,param);
  PrintInputParams::print_turbulence_parameters(pcout,param,GRID_STRETCH_FAC);
  PrintInputParams::print_linear_solver_tolerances_dual_splitting(pcout,param);
  if(param.spatial_discretization == SpatialDiscretization::DGXWall)
    PrintInputParams::print_xwall_parameters(pcout,param,N_Q_POINTS_1D_XWALL);

  postprocessor->setup();

  time_integrator->timeloop();

}

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    bool do_restart = false;
    if (argc > 1)
    {
      do_restart = std::atoi(argv[1]);
      if(do_restart)
      {
        AssertThrow(REFINE_STEPS_SPACE_MIN == REFINE_STEPS_SPACE_MAX, ExcMessage("Spatial refinement with restart not possible!"));

        //this does in principle work
        //although it doesn't make much sense
        if(REFINE_STEPS_TIME_MIN != REFINE_STEPS_TIME_MAX && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Warning: you are starting from a restart and refine the time steps!" << std::endl;
      }
    }

    //mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;refine_steps_space <= REFINE_STEPS_SPACE_MAX;++refine_steps_space)
    {
      //time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;refine_steps_time <= REFINE_STEPS_TIME_MAX;++refine_steps_time)
      {
        NavierStokesProblem<DIMENSION> navier_stokes_problem(refine_steps_space,refine_steps_time);
        navier_stokes_problem.solve_problem(do_restart);
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
