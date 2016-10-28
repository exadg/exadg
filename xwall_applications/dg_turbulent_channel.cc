
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

#include "../include/DGNavierStokesDualSplittingXWall.h"
#include "../include/DGNavierStokesCoupled.h"
#include "../include/InputParametersNavierStokes.h"

#include "TimeIntBDFDualSplittingXWallSpalartAllmaras.h"
#include "TimeIntBDFCoupled.h"
#include "../include/PostProcessor.h"
#include "../include/PostProcessorXWall.h"
#include "PrintInputParameters.h"

const unsigned int FE_DEGREE = 4;
const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
const unsigned int FE_DEGREE_XWALL = 1;
const unsigned int N_Q_POINTS_1D_XWALL = 25;
const unsigned int DIMENSION = 2; // DIMENSION >= 2
const unsigned int REFINE_STEPS_SPACE = 3;
const double GRID_STRETCH_FAC = 0.001;

template<int dim>
void InputParametersNavierStokes<dim>::set_input_parameters()
{
  output_data.output_prefix = "ch590_l3_k4k1_gt0";
  cfl = 0.16;
  diffusion_number = 0.03;
  viscosity = 1./590.;

  if(N_Q_POINTS_1D_XWALL>1) //enriched
  {
    xwall_turb = XWallTurbulenceApproach::RANSSpalartAllmaras;
    max_wdist_xwall = 0.25;
    spatial_discretization = SpatialDiscretization::DGXWall;
    IP_formulation_viscous = InteriorPenaltyFormulation::NIPG;
    solver_viscous = SolverViscous::GMRES;
    penalty_factor_divergence = 1.0e1;
    turb_stat_data.statistics_start_time = 49.;
    end_time = 50.;
  }
  else //LES
  {
    max_wdist_xwall = -0.25;
    spatial_discretization = SpatialDiscretization::DG;
    IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
    solver_viscous = SolverViscous::PCG;
    penalty_factor_divergence = 1.0e0;
    turb_stat_data.statistics_start_time = 50.;
    end_time = 70.;
  }

  //dual splitting scheme
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  projection_type = ProjectionType::DivergencePenalty;
  order_time_integrator = 2;
  start_with_low_order = true;

  //xwall specific
  IP_factor_viscous = 1.;

  calculation_of_time_step_size = TimeStepCalculation::AdaptiveTimeStepCFL;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation; //also default
  divu_integrated_by_parts = true;
  gradp_integrated_by_parts = true;
  pure_dirichlet_bc = true;
  output_solver_info_every_timesteps = 1e2;
  right_hand_side = true;

  write_restart = true;
  restart_every_timesteps = 1e9;
  restart_interval_time = 1.e9;
  restart_interval_wall_time = 1.e9;

  max_velocity = 15.;

  //solver tolerances
  rel_tol_pressure = 1.e-4;
  rel_tol_projection = 1.e-6;
  rel_tol_viscous = 1.e-4;
}

template<int dim>
class Enrichment:public Function<dim>
{
public:
  Enrichment (const double max_distance) : Function<dim>(1, 0.),
                                           max_distance(max_distance)
                                           {}
  virtual ~Enrichment (){};
  virtual double value (const Point<dim> &p,const unsigned int = 0) const
  {
    if ((p[1] > (1.0-max_distance)) || (p[1] <(-1.0 + max_distance)))
      return 1.;
    else
      return -1.;
  }
  const double max_distance;
};

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

  template<int dim, int fe_degree, int fe_degree_p>
  class PostProcessorChannel: public PostProcessor<dim,fe_degree,fe_degree_p>
  {
  public:

    PostProcessorChannel(DoFHandler<dim> const & dof_handler_u,
                         PostProcessorData<dim> const &pp_data)
        :
      PostProcessor<dim,fe_degree,fe_degree_p>(pp_data),
      statistics_ch(dof_handler_u),
      turb_stat_data(pp_data.turb_stat_data),
      output_data(pp_data.output_data)
    {}

    virtual ~PostProcessorChannel(){}

    virtual void setup(DoFHandler<dim> const                                        &dof_handler_velocity_in,
                       DoFHandler<dim> const                                        &dof_handler_pressure_in,
                       Mapping<dim> const                                           &mapping_in,
                       MatrixFree<dim,double> const                                 &matrix_free_data_in,
                       DofQuadIndexData const                                       &dof_quad_index_data_in,
                       std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution_in)
    {
      PostProcessor<dim,fe_degree,fe_degree_p>::setup(dof_handler_velocity_in,
                                                      dof_handler_pressure_in,
                                                      mapping_in,
                                                      matrix_free_data_in,
                                                      dof_quad_index_data_in,
                                                      analytical_solution_in);
      statistics_ch.setup(&grid_transform<dim>);
    }

    virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                                   parallel::distributed::Vector<double> const &intermediate_velocity,
                                   parallel::distributed::Vector<double> const &pressure,
                                   parallel::distributed::Vector<double> const &vorticity,
                                   parallel::distributed::Vector<double> const &divergence,
                                   double const                                time = 0.0,
                                   int const                                   time_step_number = -1)
    {
      PostProcessor<dim,fe_degree,fe_degree_p>::do_postprocessing(velocity,intermediate_velocity,pressure,vorticity,divergence,time,time_step_number);
      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

      if(time > this->turb_stat_data.statistics_start_time-EPSILON && time_step_number % this->turb_stat_data.statistics_every == 0)
      {
        statistics_ch.evaluate(velocity);
        if(time_step_number % 100 == 0 || time > (this->turb_stat_data.statistics_end_time-EPSILON))
          statistics_ch.write_output(this->output_data.output_prefix,this->turb_stat_data.viscosity);
      }
    };

  protected:
    StatisticsManager<dim> statistics_ch;
    TurbulenceStatisticsData turb_stat_data;
    OutputDataNavierStokes output_data;

  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
  class PostProcessorChannelXWall: public PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>
  {
  public:

    PostProcessorChannelXWall(
        std_cxx11::shared_ptr< DGNavierStokesBase<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule> >  ns_operation,
        PostProcessorData<dim> const &pp_data)
          :
        PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>(ns_operation,pp_data),
        statistics_ch(ns_operation->get_dof_handler_u())
    {}

    virtual ~PostProcessorChannelXWall(){}

    virtual void setup(DoFHandler<dim> const                                        &dof_handler_velocity_in,
                       DoFHandler<dim> const                                        &dof_handler_pressure_in,
                       Mapping<dim> const                                           &mapping_in,
                       MatrixFree<dim,double> const                                 &matrix_free_data_in,
                       DofQuadIndexData const                                       &dof_quad_index_data_in,
                       std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution_in)
    {
      PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>::setup( dof_handler_velocity_in,
                                                                                            dof_handler_pressure_in,
                                                                                            mapping_in,
                                                                                            matrix_free_data_in,
                                                                                            dof_quad_index_data_in,
                                                                                            analytical_solution_in);

      statistics_ch.setup(&grid_transform<dim>);
    }

    virtual void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                                   parallel::distributed::Vector<double> const &intermediate_velocity,
                                   parallel::distributed::Vector<double> const &pressure,
                                   parallel::distributed::Vector<double> const &vorticity,
                                   parallel::distributed::Vector<double> const &vt,
                                   double const                                time = 0.0,
                                   int const                                   time_step_number = -1)
    {
      PostProcessorXWall<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule>::
          do_postprocessing(velocity,intermediate_velocity,pressure,vorticity,vt,time,time_step_number);
      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size

      if(time > this->pp_data.turb_stat_data.statistics_start_time-EPSILON && time_step_number % this->pp_data.turb_stat_data.statistics_every == 0)
      {
        this->statistics_ch.evaluate_xwall(velocity,
                                        this->ns_operation_xw_->get_dof_handler_wdist(),
                                        this->ns_operation_xw_->get_fe_parameters(),
                                        this->pp_data.turb_stat_data.viscosity);
        if(time_step_number % 100 == 0 || time > (this->pp_data.turb_stat_data.statistics_end_time-EPSILON))
          this->statistics_ch.write_output(this->pp_data.output_data.output_prefix,this->pp_data.turb_stat_data.viscosity);
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
    NavierStokesProblem(const unsigned int refine_steps_space, const bool do_restart);
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

    InputParametersNavierStokes<dim> param;

    std_cxx11::shared_ptr<DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> > navier_stokes_operation;

    std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor;

    std_cxx11::shared_ptr<TimeIntBDFNavierStokes<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type> > time_integrator;
  };

  bool exists_test0 (const std::string& name)
  {
      std::ifstream f(name.c_str());
      return f.good();
  }

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps_space, const bool do_restart):
  pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space)
  {
    PrintInputParams::Header(pcout);

    param.set_input_parameters();
    param.check_input_parameters();

    if(do_restart == false)
      AssertThrow(param.start_with_low_order == true,ExcMessage("start with low order unless you do a restart"));

    // initialize functions (analytical solution, rhs, boundary conditions)
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
    analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(dim,param.start_time));
    std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
    analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>(param.start_time));

    std_cxx11::shared_ptr<Function<dim> > right_hand_side;
    right_hand_side.reset(new RightHandSide<dim>(param.start_time));

    field_functions.reset(new FieldFunctionsNavierStokes<dim>());
    field_functions->initial_solution_velocity = analytical_solution_velocity;
    field_functions->initial_solution_pressure = analytical_solution_pressure;
    field_functions->analytical_solution_pressure = analytical_solution_pressure;
    field_functions->right_hand_side = right_hand_side;

    boundary_descriptor_velocity.reset(new BoundaryDescriptorNavierStokes<dim>());
    boundary_descriptor_pressure.reset(new BoundaryDescriptorNavierStokes<dim>());

    bool use_adaptive_time_stepping = false;
    if(param.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
      use_adaptive_time_stepping = true;

    PostProcessorData<dim> pp_data;
    pp_data.output_data = param.output_data;
    pp_data.turb_stat_data = param.turb_stat_data;
    pp_data.turb_stat_data.viscosity = param.viscosity;

    // set next empty output number if we do restart
    if(do_restart)
    {
      unsigned int output_number = 0;
      std::ostringstream filename;
      filename.str("");
      filename.clear();
      filename << "output/"
               << pp_data.output_data.output_prefix
               << "_"
               << output_number
               << ".pvtu";
      while(exists_test0(filename.str()))
      {
        output_number++;
        filename.str("");
        filename.clear();
        filename << "output/"
                 << pp_data.output_data.output_prefix
                 << "_"
                 << output_number
                 << ".pvtu";
      }
      pp_data.output_data.output_counter_start = output_number;
    }

    if(param.spatial_discretization == SpatialDiscretization::DGXWall)
    {
      analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(2*dim,param.start_time));
      field_functions->initial_solution_velocity = analytical_solution_velocity;
      if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        if(param.xwall_turb == XWallTurbulenceApproach::RANSSpalartAllmaras)
        {
          // initialize navier_stokes_operation
          navier_stokes_operation.reset(new DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
              (triangulation,param));
          // initialize postprocessor after initializing navier_stokes_operation
          postprocessor.reset(new PostProcessorChannelXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,pp_data));

          // initialize time integrator that depends on both navier_stokes_operation and postprocessor
          time_integrator.reset(new TimeIntBDFDualSplittingXWallSpalartAllmaras<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
              navier_stokes_operation,postprocessor,param,0,use_adaptive_time_stepping));
        }
        else if(param.xwall_turb == XWallTurbulenceApproach::None)
        {
          // initialize navier_stokes_operation
          navier_stokes_operation.reset(new DGNavierStokesDualSplittingXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
              (triangulation,param));
          // initialize postprocessor after initializing navier_stokes_operation
          postprocessor.reset(new PostProcessorChannelXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,pp_data));

          // initialize time integrator that depends on both navier_stokes_operation and postprocessor
          time_integrator.reset(new TimeIntBDFDualSplittingXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
              navier_stokes_operation,postprocessor,param,0,use_adaptive_time_stepping));
        }
        else if(param.xwall_turb == XWallTurbulenceApproach::Undefined)
        {
          AssertThrow(false,ExcMessage("Turbulence approach for xwall undefined"));
        }
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
        postprocessor.reset(new PostProcessorChannel<dim, FE_DEGREE, FE_DEGREE_P>
                      (navier_stokes_operation->get_dof_handler_u(),pp_data));
        // initialize time integrator that depends on both navier_stokes_operation and postprocessor
        time_integrator.reset(new TimeIntBDFDualSplitting<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
            navier_stokes_operation,postprocessor,param,0,use_adaptive_time_stepping));
      }
      else if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessorChannel<dim, FE_DEGREE, FE_DEGREE_P>
                      (navier_stokes_operation->get_dof_handler_u(),pp_data));
        // initialize time integrator that depends on both navier_stokes_operation and postprocessor
        time_integrator.reset(new TimeIntBDFCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
            navier_stokes_operation,postprocessor,param,0,use_adaptive_time_stepping));
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

  PrintInputParams::print_solver_parameters<dim>(pcout,param);
  PrintInputParams::print_turbulence_parameters<dim>(pcout,param,GRID_STRETCH_FAC);
  PrintInputParams::print_linear_solver_tolerances_dual_splitting<dim>(pcout,param);
  if(param.spatial_discretization == SpatialDiscretization::DGXWall)
    PrintInputParams::print_xwall_parameters<dim>(pcout,param,N_Q_POINTS_1D_XWALL);

  std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution;
  analytical_solution.reset(new AnalyticalSolutionNavierStokes<dim>());
  analytical_solution->velocity = field_functions->initial_solution_velocity;
  analytical_solution->pressure = field_functions->initial_solution_pressure;

  DofQuadIndexData dof_quad_index_data;

  postprocessor->setup(navier_stokes_operation->get_dof_handler_u(),
                       navier_stokes_operation->get_dof_handler_p(),
                       navier_stokes_operation->get_mapping(),
                       navier_stokes_operation->get_data(),
                       dof_quad_index_data,
                       analytical_solution);

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
    }

    NavierStokesProblem<DIMENSION> navier_stokes_problem(REFINE_STEPS_SPACE,do_restart);
    navier_stokes_problem.solve_problem(do_restart);
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
