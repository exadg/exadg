
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

#include <fstream>
#include <sstream>

#include "../include/DGNavierStokesDualSplitting.h"
#include "../include/DGNavierStokesDualSplittingXWall.h"
#include "../include/DGNavierStokesCoupled.h"

#include "../include/InputParameters.h"
#include "TimeIntBDFDualSplitting.h"
#include "TimeIntBDFDualSplittingXWall.h"
#include "TimeIntBDFDualSplittingXWallSpalartAllmaras.h"
#include "TimeIntBDFCoupled.h"
#include "../include/PostProcessor.h"
#include "PostProcessorXWall.h"
#include "PrintInputParameters.h"

#include "DriverSteadyProblems.h"

using namespace dealii;

// specify flow problem that has to be solved
#define VORTEX
//#define STOKES_GUERMOND
//#define STOKES_SHAHBAZI
//#define POISEUILLE
//#define CUETTE
//#define CAVITY
//#define KOVASZNAY
//#define BELTRAMI
//#define FLOW_PAST_CYLINDER


ProblemType PROBLEM_TYPE = ProblemType::Unsteady; //Steady; //Unsteady;
EquationType EQUATION_TYPE = EquationType::NavierStokes; // Stokes; // NavierStokes;
TreatmentOfConvectiveTerm TREATMENT_OF_CONVECTIVE_TERM = TreatmentOfConvectiveTerm::Explicit; // Explicit; // Implicit;

/************* temporal discretization ***********/
// which temporal discretization approach
TemporalDiscretization TEMPORAL_DISCRETIZATION = TemporalDiscretization::BDFDualSplittingScheme; //BDFDualSplittingScheme // BDFCoupledSolution

// type of time step calculation
TimeStepCalculation TIME_STEP_CALCULATION = TimeStepCalculation::ConstTimeStepCFL; //ConstTimeStepUserSpecified; //ConstTimeStepCFL; //AdaptiveTimeStepCFL;
/*************************************************/

/************* spatial discretization ************/
SpatialDiscretization SPATIAL_DISCRETIZATION = SpatialDiscretization::DG; //DG //DGXWall

FormulationViscousTerm FORMULATION_VISCOUS_TERM = FormulationViscousTerm::DivergenceFormulation; //DivergenceFormulation; //LaplaceFormulation;
InteriorPenaltyFormulationViscous IP_FORMULATION_VISCOUS = InteriorPenaltyFormulationViscous::SIPG; //SIPG; //NIPG;

bool const DIVU_INTEGRATED_BY_PARTS = true;//true;
bool const DIVU_USE_BOUNDARY_DATA = false;//true;
bool const GRADP_INTEGRATED_BY_PARTS = true;//true;
bool const GRADP_USE_BOUNDARY_DATA = false;//true;
/*************************************************/

/******** high-order dual splitting scheme *******/
// approach of Leriche et al. to obtain stability in the limit of small time steps
bool const STS_STABILITY = false;

// pressure Poisson equation
SolverPoisson SOLVER_POISSON = SolverPoisson::PCG; //PCG;
PreconditionerPoisson PRECONDITIONER_POISSON = PreconditionerPoisson::GeometricMultigrid; // None; //Jacobi; //GeometricMultigrid;

// multigrid pressure Poisson
MultigridSmoother MULTIGRID_SMOOTHER = MultigridSmoother::Chebyshev; //Chebyshev;
MultigridCoarseGridSolver MULTIGRID_COARSE_GRID_SOLVER = MultigridCoarseGridSolver::coarse_chebyshev_smoother; //coarse_iterative_nopreconditioner; //coarse_iterative_jacobi;

// projection step
ProjectionType PROJECTION_TYPE = ProjectionType::DivergencePenalty; //NoPenalty; //DivergencePenalty; //DivergenceAndContinuityPenalty;
SolverProjection SOLVER_PROJECTION = SolverProjection::PCG; //LU; //PCG;
PreconditionerProjection PRECONDITIONER_PROJECTION = PreconditionerProjection::InverseMassMatrix; //None; //Jacobi; //InverseMassMatrix;

// viscous step
SolverViscous SOLVER_VISCOUS = SolverViscous::PCG; //PCG; //GMRES;
PreconditionerViscous PRECONDITIONER_VISCOUS = PreconditionerViscous::InverseMassMatrix; //None; //Jacobi; //InverseMassMatrix; //GeometricMultigrid;

// multigrid viscous step
MultigridSmoother MULTIGRID_SMOOTHER_VISCOUS = MultigridSmoother::Chebyshev; //Chebyshev;
MultigridCoarseGridSolver MULTIGRID_COARSE_GRID_SOLVER_VISCOUS = MultigridCoarseGridSolver::coarse_chebyshev_smoother;//coarse_chebyshev_smoother; //coarse_iterative_nopreconditioner; //coarse_iterative_jacobi;
/************************************************/

/************ coupled solver ********************/
const bool USE_SYMMETRIC_SADDLE_POINT_MATRIX = true;

// preconditioner
PreconditionerLinearizedNavierStokes PRECONDITIONER_LINEARIZED_NAVIER_STOKES =
    PreconditionerLinearizedNavierStokes::BlockTriangularFactorization; //None; //BlockDiagonal; //BlockTriangular; //BlockTriangularFactorization;
PreconditionerMomentum PRECONDITIONER_MOMENTUM =
    PreconditionerMomentum::InverseMassMatrix; //None; //InverseMassMatrix; //GeometricMultigrid;
PreconditionerSchurComplement PRECONDITIONER_SCHUR_COMPLEMENT =
    PreconditionerSchurComplement::CahouetChabard; //None; //InverseMassMatrix; //GeometricMultigrid; //CahouetChabard;
/************************************************/

#ifdef VORTEX
  const unsigned int FE_DEGREE = 5; //2
  const unsigned int FE_DEGREE_P = FE_DEGREE-1;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 1;
  const unsigned int REFINE_STEPS_SPACE_MAX = 1;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 1.0;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1e6;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool DIVU_TIMESERIES = true;
  const bool COMPUTE_DIVERGENCE = true;
  const bool ANALYTICAL_SOLUTION = true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.1; //0.1;
  const double U_X_MAX = 1.0;
  const double MAX_VELOCITY = 1.4*U_X_MAX;
  const double TIME_STEP_SIZE = 1.e-3;//1.e-2;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 0.025;//0.01;
  // interior penalty method - penalty factor
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;

  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0; //PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = false;

  const double ABS_TOL_NEWTON = 1.0e-20;
  const double REL_TOL_NEWTON = 1.0e-6;
  unsigned int const MAX_ITER_NEWTON = 1e2;
  const double ABS_TOL_LINEAR = 1.0e-20;
  const double REL_TOL_LINEAR = 1.0e-6;
  unsigned int const MAX_ITER_LINEAR = 1e6;

  const double ABS_TOL_PRESSURE = 1.0e-18;
  const double REL_TOL_PRESSURE = 1.0e-12;
  const double ABS_TOL_PROJECTION = 1.0e-18;
  const double REL_TOL_PROJECTION = 1.0e-12;
  const double ABS_TOL_VISCOUS = 1.0e-18;
  const double REL_TOL_VISCOUS = 1.0e-12;

//  const double ABS_TOL_PRESSURE = 1.0e-12;
//  const double REL_TOL_PRESSURE = 1.0e-6;
//  const double ABS_TOL_VISCOUS = 1.0e-12;
//  const double REL_TOL_VISCOUS = 1.0e-6;
//  const double ABS_TOL_PROJECTION = 1.0e-12;
//  const double REL_TOL_PROJECTION = 1.0e-6;

  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e6;

  const std::string OUTPUT_PREFIX = "vortex_flow";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef POISEUILLE
  const unsigned int FE_DEGREE = 3;
  const unsigned int FE_DEGREE_P = FE_DEGREE-1; //FE_DEGREE; //FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 2;
  const unsigned int REFINE_STEPS_SPACE_MAX = 2;

  const double START_TIME = 0.0;
  const double END_TIME = 10.0;
  const double OUTPUT_INTERVAL_TIME = 0.5;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1000.;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true; 
  const bool DIVU_TIMESERIES = false; //true;
  const bool COMPUTE_DIVERGENCE = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 2.0; //0.1;
  const double TIME_STEP_SIZE = 1.0e-3;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 0.1;
  const double H = 1.0;
  const double L = 4.0;

  const double MAX_VELOCITY = 1.0;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;
  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = false;

  const double ABS_TOL_NEWTON = 1.0e-12;
  const double REL_TOL_NEWTON = 1.0e-6;
  unsigned int const MAX_ITER_NEWTON = 1e2;
  const double ABS_TOL_LINEAR = 1.0e-12;
  const double REL_TOL_LINEAR = 1.0e-6;
  unsigned int const MAX_ITER_LINEAR = 1e6;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-6;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-6;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e0;

  const std::string OUTPUT_PREFIX = "poiseuille";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = true;
#endif

#ifdef CUETTE
  const unsigned int FE_DEGREE = 2;
  const unsigned int FE_DEGREE_P = FE_DEGREE-1;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 2;
  const unsigned int REFINE_STEPS_SPACE_MAX = 2;

  const double START_TIME = 0.0;
  const double END_TIME = 10.0;
  const double OUTPUT_INTERVAL_TIME = 0.5;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1000.;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = true; //true;
  const bool COMPUTE_DIVERGENCE = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.1;
  const double TIME_STEP_SIZE = 1.e-1;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 0.01;
  const double H = 1.0;
  const double L = 4.0;

  const double MAX_VELOCITY = 1.0;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;
  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = false;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-6;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-6;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  // show solver performance (wall time, number of iterations) every ... timesteps
  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e0;

  const std::string OUTPUT_PREFIX = "cuette";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = true;
#endif

#ifdef CAVITY
  const unsigned int FE_DEGREE = 6;
  const unsigned int FE_DEGREE_P = FE_DEGREE-1;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 3;
  const unsigned int REFINE_STEPS_SPACE_MAX = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 40.0;
  const double OUTPUT_INTERVAL_TIME = 2.0;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1000.;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = false; //true;
  const bool COMPUTE_DIVERGENCE = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.1;
  const double TIME_STEP_SIZE = 1.e-1;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 4.0e-3;//0.0002;
  const double L = 1.0;

  const double MAX_VELOCITY = 1.0;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;
  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = true;

  const double ABS_TOL_NEWTON = 1.0e-20;
  const double REL_TOL_NEWTON = 1.0e-6;
  unsigned int const MAX_ITER_NEWTON = 1e2;
  const double ABS_TOL_LINEAR = 1.0e-20;
  const double REL_TOL_LINEAR = 1.0e-8;
  unsigned int const MAX_ITER_LINEAR = 1e4;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-6;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-6;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  // show solver performance (wall time, number of iterations) every ... timesteps
  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e5;

  const std::string OUTPUT_PREFIX = "cavity";

  const unsigned int ORDER_TIME_INTEGRATOR = 2;
  const bool START_WITH_LOW_ORDER = true;
#endif

#ifdef KOVASZNAY
  const unsigned int FE_DEGREE = 2;
  const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 3;
  const unsigned int REFINE_STEPS_SPACE_MAX = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1000.;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = false; //true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.01;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 0.025;

  const double MAX_VELOCITY = 3.6;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;
  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = false;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  // show solver performance (wall time, number of iterations) every ... timesteps
  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e5;

  const std::string OUTPUT_PREFIX = "kovasznay";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef BELTRAMI
  const unsigned int FE_DEGREE = 3;
  const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 3;
  const unsigned int REFINE_STEPS_SPACE_MIN = 2;
  const unsigned int REFINE_STEPS_SPACE_MAX = 2;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1000.;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = false; //true;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.01;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 0.1;

  const double MAX_VELOCITY = 3.5;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;
  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = true;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-8;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-8;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e5;

  const std::string OUTPUT_PREFIX = "beltrami";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef STOKES_GUERMOND
  const unsigned int FE_DEGREE = 4;//3
  const unsigned int FE_DEGREE_P = FE_DEGREE-1;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 4;//2
  const unsigned int REFINE_STEPS_SPACE_MAX = 4;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 1.0;//(END_TIME-START_TIME)/10.0;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1.e6;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = false;
  const bool COMPUTE_DIVERGENCE = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.2; // CFL number irrelevant for Stokes flow problem
  const double TIME_STEP_SIZE = 1.0e-1;//2.e-4;//1.e-1;///std::pow(2.,13); //5.0e-4
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 1.0e-6;//0.01;

  const double MAX_VELOCITY = 2.65; // MAX_VELOCITY also irrelevant
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;

  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0; //1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0; //PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = true;

  const double ABS_TOL_NEWTON = 1.0e-12;
  const double REL_TOL_NEWTON = 1.0e-6;
  unsigned int const MAX_ITER_NEWTON = 1e2;
  const double ABS_TOL_LINEAR = 1.0e-12;
  const double REL_TOL_LINEAR = 1.0e-6;
  unsigned int const MAX_ITER_LINEAR = 1e6;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-6;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-6;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  // show solver performance (wall time, number of iterations) every ... timesteps
  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e4;

  const std::string OUTPUT_PREFIX = "stokes";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef STOKES_SHAHBAZI
  const unsigned int FE_DEGREE = 2;//3
  const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 3;//2
  const unsigned int REFINE_STEPS_SPACE_MAX = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 0.1;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1000.;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = false;
  const bool COMPUTE_DIVERGENCE = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.2; // CFL number irrelevant for Stokes flow problem
  const double TIME_STEP_SIZE = 1.e-4;///std::pow(2.,13); //5.0e-4
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 1.0e0;

  const double MAX_VELOCITY = 1.; // MAX_VELOCITY also irrelevant
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;

  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = true;

  const double ABS_TOL_NEWTON = 1.0e-12;
  const double REL_TOL_NEWTON = 1.0e-6;
  unsigned int const MAX_ITER_NEWTON = 1e2;
  const double ABS_TOL_LINEAR = 1.0e-12;
  const double REL_TOL_LINEAR = 1.0e-6;
  unsigned int const MAX_ITER_LINEAR = 1e6;

//  const double ABS_TOL_PRESSURE = 1.0e-16;
//  const double REL_TOL_PRESSURE = 1.0e-12;
//  const double ABS_TOL_VISCOUS = 1.0e-16;
//  const double REL_TOL_VISCOUS = 1.0e-12;
//  const double ABS_TOL_PROJECTION = 1.0e-16;
//  const double REL_TOL_PROJECTION = 1.0e-12;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-6;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-6;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  // show solver performance (wall time, number of iterations) every ... timesteps
  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e5;

  const std::string OUTPUT_PREFIX = "stokes";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = false;
#endif

#ifdef FLOW_PAST_CYLINDER
  const unsigned int FE_DEGREE = 2;
  const unsigned int FE_DEGREE_P = FE_DEGREE-1;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 0;
  const unsigned int REFINE_STEPS_SPACE_MAX = 0;

  const double START_TIME = 0.0;
  const double END_TIME = 8.0;
  const double OUTPUT_INTERVAL_TIME = 0.4;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double RESTART_INTERVAL_TIME = 100.;
  const double RESTART_INTERVAL_WALL_TIME = 1000.;
  const unsigned int RESTART_INTERVAL_STEP = 1e6;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = true;
  const bool COMPUTE_DIVERGENCE = true;
  const int MAX_NUM_STEPS = 1e6;
  const double TIME_STEP_SIZE = 0.1;//1.e-2;
  const double CFL = 0.05;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 0.001;
  const unsigned int TEST_CASE = 1; // 1, 2 or 3
  const double Um = (DIMENSION == 2 ? (TEST_CASE==1 ? 0.3 : 1.5) : (TEST_CASE==1 ? 0.45 : 2.25));
  const double D = 0.1;
  const double H = 0.41;
  const double L1 = 0.3;
  const double L2 = 2.5;
  const double X_C = 0.5;
  const double Y_C = 0.2;

  const double MAX_VELOCITY = Um;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;

  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e1;
  const double PENALTY_FACTOR_CONTINUITY = PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const bool PURE_DIRICHLET_BC = false;

  const double ABS_TOL_NEWTON = 1.0e-12;
  const double REL_TOL_NEWTON = 1.0e-6;
  unsigned int const MAX_ITER_NEWTON = 1e2;
  const double ABS_TOL_LINEAR = 1.0e-12;
  const double REL_TOL_LINEAR = 1.0e-6;
  unsigned int const MAX_ITER_LINEAR = 1e6;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-6;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-6;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-6;

  // show solver performance (wall time, number of iterations) every ... timesteps
  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e0;//1e3;

  const std::string OUTPUT_PREFIX = "fpc_r0_p2";

  const unsigned int ORDER_TIME_INTEGRATOR = 2;
  const bool START_WITH_LOW_ORDER = true;
#endif

  void InputParameters::set_input_parameters()
  {
    problem_type = PROBLEM_TYPE;
    equation_type = EQUATION_TYPE;
    treatment_of_convective_term = TREATMENT_OF_CONVECTIVE_TERM;
    start_time = START_TIME;
    end_time = END_TIME;
    max_number_of_steps = MAX_NUM_STEPS;
    calculation_of_time_step_size = TIME_STEP_CALCULATION;
    cfl = CFL;
    max_velocity = MAX_VELOCITY;
    time_step_size = TIME_STEP_SIZE;
    viscosity = VISCOSITY;
    temporal_discretization = TEMPORAL_DISCRETIZATION;
    spatial_discretization = SPATIAL_DISCRETIZATION;
    order_time_integrator = ORDER_TIME_INTEGRATOR;
    start_with_low_order = START_WITH_LOW_ORDER;
    use_symmetric_saddle_point_matrix = USE_SYMMETRIC_SADDLE_POINT_MATRIX;
    small_time_steps_stability = STS_STABILITY;
    pure_dirichlet_bc = PURE_DIRICHLET_BC;
    penalty_factor_divergence = PENALTY_FACTOR_DIVERGENCE;
    penalty_factor_continuity = PENALTY_FACTOR_CONTINUITY;
    compute_divergence = COMPUTE_DIVERGENCE;
    divu_integrated_by_parts = DIVU_INTEGRATED_BY_PARTS;
    divu_use_boundary_data = DIVU_USE_BOUNDARY_DATA;
    gradp_integrated_by_parts = GRADP_INTEGRATED_BY_PARTS;
    gradp_use_boundary_data = GRADP_USE_BOUNDARY_DATA;
    IP_factor_pressure = IP_FACTOR_PRESSURE;
    IP_factor_viscous = IP_FACTOR_VISCOUS;
    abs_tol_newton = ABS_TOL_NEWTON;
    rel_tol_newton = REL_TOL_NEWTON;
    max_iter_newton = MAX_ITER_NEWTON;
    abs_tol_linear = ABS_TOL_LINEAR;
    rel_tol_linear = REL_TOL_LINEAR;
    max_iter_linear = MAX_ITER_LINEAR;
    abs_tol_pressure = ABS_TOL_PRESSURE;
    rel_tol_pressure = REL_TOL_PRESSURE;
    abs_tol_projection = ABS_TOL_PROJECTION;
    rel_tol_projection = REL_TOL_PROJECTION;
    abs_tol_viscous = ABS_TOL_VISCOUS;
    rel_tol_viscous = REL_TOL_VISCOUS;
    solver_poisson = SOLVER_POISSON;
    preconditioner_poisson = PRECONDITIONER_POISSON;
    multigrid_smoother = MULTIGRID_SMOOTHER;
    multigrid_coarse_grid_solver = MULTIGRID_COARSE_GRID_SOLVER;
    projection_type = PROJECTION_TYPE;
    solver_projection = SOLVER_PROJECTION;
    preconditioner_projection = PRECONDITIONER_PROJECTION;
    formulation_viscous_term = FORMULATION_VISCOUS_TERM;
    IP_formulation_viscous = IP_FORMULATION_VISCOUS;
    solver_viscous = SOLVER_VISCOUS;
    preconditioner_viscous = PRECONDITIONER_VISCOUS;
    multigrid_smoother_viscous = MULTIGRID_SMOOTHER_VISCOUS;
    multigrid_coarse_grid_solver_viscous = MULTIGRID_COARSE_GRID_SOLVER_VISCOUS;
    preconditioner_linearized_navier_stokes = PRECONDITIONER_LINEARIZED_NAVIER_STOKES;
    preconditioner_momentum = PRECONDITIONER_MOMENTUM;
    preconditioner_schur_complement = PRECONDITIONER_SCHUR_COMPLEMENT;
    output_solver_info_every_timesteps = OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS;
    output_start_time = OUTPUT_START_TIME;
    output_interval_time = OUTPUT_INTERVAL_TIME;
    restart_interval_time = RESTART_INTERVAL_TIME;
    restart_interval_wall_time = RESTART_INTERVAL_WALL_TIME;
    restart_interval_step = RESTART_INTERVAL_STEP;
    output_prefix = OUTPUT_PREFIX;
    error_calc_start_time = ERROR_CALC_START_TIME;
    error_calc_interval_time = ERROR_CALC_INTERVAL_TIME;
    analytical_solution_available = ANALYTICAL_SOLUTION;
  }

  template<int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    AnalyticalSolution (const bool    is_velocity,
                        const double  time = 0.)
      :
      Function<dim>(is_velocity ? dim : 1, time),
      is_velocity(is_velocity)
    {}

    AnalyticalSolution (const bool    is_velocity,
                        const double  time,
                        unsigned int n_components)
      :
      Function<dim>(is_velocity ? n_components : 1, time),
      is_velocity(is_velocity)
    {}

    virtual ~AnalyticalSolution(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const bool is_velocity;
  };

  template<int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> &p,const unsigned int given_component) const
  {
    const unsigned int component = is_velocity ? given_component : dim;

    double t = this->get_time();
    double result = 0.0;
    (void)t;

    /*********************** cavity flow ********************************/
#ifdef CAVITY
    // constant velocity
    if(PROBLEM_TYPE == ProblemType::Steady)
    {
      if(component == 0 && (std::abs(p[1]-1.0)<1.0e-15))
        result = 1.0;
    }
    else if(PROBLEM_TYPE == ProblemType::Unsteady)
    {
      const double T = 0.5;
      const double pi = numbers::PI;
      if(component == 0 && (std::abs(p[1]-1.0)<1.0e-15))
        result = t<T ? std::sin(pi/2.*t/T) : 1.0;
    }
#endif
    /********************************************************************/

    /*********************** Cuette flow problem ************************/
#ifdef CUETTE
    // steady
//    if(component == 0)
//          result = ((p[1]+ H/2.)*MAX_VELOCITY);

    // unsteady
    const double T = 1.0;
    const double pi = numbers::PI;
    if(component == 0)
      result = ((p[1]+H/2.)*MAX_VELOCITY)*(t<T ? std::sin(pi/2.*t/T) : 1.0);
#endif
    /********************************************************************/

    /****************** Poisseuille flow problem ************************/
#ifdef POISEUILLE
    // constant velocity profile at inflow
//    double T = 0.5;
//    const double pi = numbers::PI;
//    if(component == 0 && (std::abs(p[0])<1.0e-12))
//      result = MAX_VELOCITY * (t<T ? std::sin(pi/2.*t/T) : 1.0);

    // parabolic velocity profile at inflow - steady
//    const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
//    if(component == 0)
//      result = 1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0;
//    if(component == dim)
//      result = (p[0]-4.0)*pressure_gradient;

    // parabolic velocity profile at inflow - unsteady
    const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
    const double pi = numbers::PI;
    double T = 1.0e0; //0.5;
    if(component == 0)
      result = 1.0/VISCOSITY * pressure_gradient * (pow(p[1],2.0)-1.0)/2.0 * (t<T ? std::sin(pi/2.*t/T) : 1.0);
    if(component == dim)
      result = (p[0]-4.0) * pressure_gradient * (t<T ? std::sin(pi/2.*t/T) : 1.0);

#endif
    /********************************************************************/

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
      result = -U_X_MAX*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if(component == 1)
      result = U_X_MAX*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if(component == dim)
      result = -U_X_MAX*std::cos(2*pi*p[0])*std::cos(2*pi*p[1])*std::exp(-8.0*pi*pi*VISCOSITY*t);
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
#ifdef STOKES_GUERMOND
    const double pi = numbers::PI;
    double sint = std::sin(t);
    double sinx = std::sin(pi*p[0]);
    double siny = std::sin(pi*p[1]);
    double cosx = std::cos(pi*p[0]);
    double sin2x = std::sin(2.*pi*p[0]);
    double sin2y = std::sin(2.*pi*p[1]);
    if (component == 0)
      result = pi*sint*sin2y*std::pow(sinx,2.);
    else if (component == 1)
      result = -pi*sint*sin2x*std::pow(siny,2.);
    else if (component == dim)
    {
      result = cosx*siny*sint;
    }
#endif

#ifdef STOKES_SHAHBAZI
    const double pi = numbers::PI;
    const double a = 2.883356;
    const double lambda = VISCOSITY*(1.+a*a);

    double exp_t = std::exp(-lambda*t);
    double sin_x = std::sin(p[0]);
    double cos_x = std::cos(p[0]);
    double cos_a = std::cos(a);
    double sin_ay = std::sin(a*p[1]);
    double cos_ay = std::cos(a*p[1]);
    double sinh_y = std::sinh(p[1]);
    double cosh_y = std::cosh(p[1]);
    if (component == 0)
      result = exp_t*sin_x*(a*sin_ay-cos_a*sinh_y);
    else if (component == 1)
      result = exp_t*cos_x*(cos_ay+cos_a*cosh_y);
    else if (component == dim)
      result = lambda*cos_a*cos_x*sinh_y*exp_t;
#endif
    /********************************************************************/

    /********************** flow past cylinder **************************/
#ifdef FLOW_PAST_CYLINDER
    if(component == 0 && std::abs(p[0]-(dim==2 ? L1: 0.0))<1.e-12)
    {
      const double pi = numbers::PI;
      const double T = 1.0;
      double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
      if(TEST_CASE < 3)
        result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0);
      if(TEST_CASE == 3)
        result = coefficient * p[1] * (H-p[1]) * std::sin(pi*t/END_TIME);
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
    NeumannBoundaryVelocity (const double time = 0.)
      :
      Function<dim>(dim, time)
    {}

    virtual ~NeumannBoundaryVelocity(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double NeumannBoundaryVelocity<dim>::value(const Point<dim> &p,const unsigned int component) const
  {
    double t = this->get_time();
    double result = 0.0;
    (void)t;

    // Kovasznay flow
#ifdef KOVASZNAY
    // Laplace formulation of viscous term -> prescribe velocity gradient (grad U)*n on Gamma_N
    if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::LaplaceFormulation)
    {
      const double pi = numbers::PI;
      const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);
      if (component == 0)
        result = -lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
      else if (component == 1)
        result = std::pow(lambda,2.0)/2/pi*std::exp(lambda)*std::sin(2*pi*p[1]);
    }
    // Divergence formulation of viscous term -> prescribe (grad U + (grad U) ^T)*n on Gamma_N
    else if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::DivergenceFormulation)
    {
      const double pi = numbers::PI;
      const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);
      if (component == 0)
        result = -2.0*lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
      else if (component == 1)
        result = (std::pow(lambda,2.0)/2/pi+2.0*pi)*std::exp(lambda)*std::sin(2*pi*p[1]);
    }
    else
    {
      AssertThrow();
    }
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
    if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::LaplaceFormulation)
    {
      const double pi = numbers::PI;
      if(component==0)
      {
        if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
          result = U_X_MAX*2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
          result = -U_X_MAX*2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
      else if(component==1)
      {
        if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
          result = -U_X_MAX*2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
          result = U_X_MAX*2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
    }
    // Divergence formulation of viscous term -> prescribe (grad U + (grad U)^T)*n on Gamma_N
    else if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::DivergenceFormulation)
    {
      const double pi = numbers::PI;
      if(component==0)
      {
        if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
          result = -U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
          result = U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
      else if(component==1)
      {
        if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
          result = -U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
          result = U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Formulation of viscous term not specified - possibilities are DIVERGENCE_FORMULATION_VISCOUS and LAPLACE_FORMULATION_VISCOUS"));
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
    RHS (const double time = 0.)
      :
      Function<dim>(dim, time),time(time)
    {}

    virtual ~RHS(){};

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

    void setup(const double* massflows, double oldforce, double integrand);
  private:
    const double time;
  };

  template<int dim>
  double RHS<dim>::value(const Point<dim> &p,const unsigned int component) const
  {
  double t = this->get_time();
  double result = 0.0;
  (void)t;
  (void)p;
  (void)component;

#ifdef STOKES_GUERMOND
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
    result = pi*cost*sin2y*std::pow(sinx,2.)
        - 2.*std::pow(pi,3.)*sint*sin2y*(1.-4.*std::pow(sinx,2.))*VISCOSITY
        - pi*sint*sinx*siny;
  else if (component == 1)
    result = -pi*cost*sin2x*std::pow(siny,2.)
        + 2.*std::pow(pi,3.)*sint*sin2x*(1.-4.*std::pow(siny,2.))*VISCOSITY
        + pi*sint*cosx*cosy;
#endif

  return result;
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
  double PressureBC_dudt<dim>::value(const Point<dim> &p,const unsigned int component) const
  {
  double t = this->get_time();
  (void)t;
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
    result = U_X_MAX*4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
  else if(component == 1)
    result = -U_X_MAX*4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
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
#ifdef STOKES_GUERMOND
  const double pi = numbers::PI;
  double cost = std::cos(t);
  double sinx = std::sin(pi*p[0]);
  double siny = std::sin(pi*p[1]);
  double sin2x = std::sin(2.*pi*p[0]);
  double sin2y = std::sin(2.*pi*p[1]);
  if (component == 0)
    result = pi*cost*sin2y*std::pow(sinx,2.);
  else if (component == 1)
    result = -pi*cost*sin2x*std::pow(siny,2.);
#endif

#ifdef STOKES_SHAHBAZI
    const double pi = numbers::PI;
    const double a = 2.883356;
    const double lambda = VISCOSITY*(1.+a*a);

    double exp_t = std::exp(-lambda*t);
    double sin_x = std::sin(p[0]);
    double cos_x = std::cos(p[0]);
    double cos_a = std::cos(a);
    double sin_ay = std::sin(a*p[1]);
    double cos_ay = std::cos(a*p[1]);
    double sinh_y = std::sinh(p[1]);
    double cosh_y = std::cosh(p[1]);
    if (component == 0)
      result = -lambda*exp_t*sin_x*(a*sin_ay-cos_a*sinh_y);
    else if (component == 1)
      result = -lambda*exp_t*cos_x*(cos_ay+cos_a*cosh_y);
#endif

  // flow past cylinder
#ifdef FLOW_PAST_CYLINDER
  if(component == 0 && std::abs(p[0]-(dim==2 ? L1 : 0.0))<1.e-12)
  {
    const double pi = numbers::PI;
    const double T = 1.0;
    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
    if(TEST_CASE < 3)
      result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? (pi/2./T)*std::cos(pi/2.*t/T) : 0.0);
    if(TEST_CASE == 3)
      result = coefficient * p[1] * (H-p[1]) * std::cos(pi*t/END_TIME)*pi/END_TIME;
    if (dim == 3)
      result *= p[2] * (H-p[2]);
  }
#endif

  return result;
  }

  template<int dim>
  class NavierStokesProblem
  {
  public:
    typedef typename DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>::value_type value_type;
    NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time=0);
    void solve_problem(bool do_restart);

  private:
    void create_grid();
    void print_parameters() const;

    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces;

    const unsigned int n_refine_space;

    std::set<types::boundary_id> dirichlet_boundary;
    std::set<types::boundary_id> neumann_boundary;

    InputParameters param;

    std_cxx11::shared_ptr<DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> > navier_stokes_operation;

    std_cxx11::shared_ptr<PostProcessor<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> > postprocessor;

    std_cxx11::shared_ptr<TimeIntBDF<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type> > time_integrator;

    std_cxx11::shared_ptr<DriverSteadyProblems<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type> > driver_steady;
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

    if(param.spatial_discretization == SpatialDiscretization::DGXWall)
    {
      if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesDualSplittingXWallSpalartAllmaras<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessorXWall<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,param));
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
      if(param.problem_type == ProblemType::Steady)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessor<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,param));
        // initialize driver for steady state problem that depends on both navier_stokes_operation and postprocessor
        driver_steady.reset(new DriverSteadyProblems<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>
            (navier_stokes_operation,postprocessor,param));

      }
      else if(param.problem_type == ProblemType::Unsteady &&
              param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
      {
        // initialize navier_stokes_operation
        navier_stokes_operation.reset(new DGNavierStokesDualSplitting<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
            (triangulation,param));
        // initialize postprocessor after initializing navier_stokes_operation
        postprocessor.reset(new PostProcessor<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,param));
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
        postprocessor.reset(new PostProcessor<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>(navier_stokes_operation,param));
        // initialize time integrator that depends on both navier_stokes_operation and postprocessor
        time_integrator.reset(new TimeIntBDFCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
            navier_stokes_operation,postprocessor,param,refine_steps_time));
      }
    }

  }

#ifdef FLOW_PAST_CYLINDER
  void create_triangulation(Triangulation<2> &tria, const bool compute_in_2d = true)
  {
    HyperBallBoundary<2> boundary(Point<2>(0.5,0.2), 0.05);
    Triangulation<2> left, middle, right, tmp, tmp2;
    std::vector<unsigned int> ref_1(2, 2);
    ref_1[1] = 2;

    GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(), Point<2>(0.3, 0.41), false);
    std::vector<unsigned int> ref_2(2, 9);
    ref_2[1] = 2;

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
    GridGenerator::extrude_triangulation(tria_2d, 3, 0.41, tria);

    // Set the cylinder boundary  to 2, outflow to 1, the rest to 0.
    for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
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
#endif

  template<int dim>
  void NavierStokesProblem<dim>::create_grid ()
  {
    /* --------------- Generate grid ------------------- */

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
    triangulation.refine_global(n_refine_space);

    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef POISEUILLE
    std::vector<unsigned int> repetitions({2,1});
    Point<dim> point1(0.0,-H/2.), point2(L,H/2.);
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
          cell->face(face_number)->set_boundary_id (1);
      }
    }
    triangulation.refine_global(n_refine_space);
    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef CUETTE
    std::vector<unsigned int> repetitions({2,1});
    Point<dim> point1(0.0,-H/2.), point2(L,H/2.);
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
       if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
          cell->face(face_number)->set_boundary_id (1);
      }
    }
    triangulation.refine_global(n_refine_space);
    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef CAVITY
    Point<dim> point1(0.0,0.0), point2(L,L);
    GridGenerator::hyper_rectangle(triangulation,point1,point2);
    triangulation.refine_global(n_refine_space);
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
    triangulation.refine_global(n_refine_space);
    dirichlet_boundary.insert(0);
    neumann_boundary.insert(1);
#endif

#ifdef BELTRAMI
    const double left = -1.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);
    triangulation.refine_global(n_refine_space);
    dirichlet_boundary.insert(0);
#endif

#ifdef STOKES_GUERMOND
    const double left = 0.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);
    triangulation.refine_global(n_refine_space);
    dirichlet_boundary.insert(0);
#endif

#ifdef STOKES_SHAHBAZI
    const double left = -1.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);
    triangulation.refine_global(n_refine_space);
    dirichlet_boundary.insert(0);
#endif

#ifdef FLOW_PAST_CYLINDER
    Point<dim> direction;
    direction[dim-1] = 1.;

    Point<dim> center;
    center[0] = 0.5;
    center[1] = 0.2;

    static std_cxx11::shared_ptr<Manifold<dim> > cylinder_manifold =
      std_cxx11::shared_ptr<Manifold<dim> >(dim == 2 ? static_cast<Manifold<dim>*>(new HyperBallBoundary<dim>(center, 0.05)) :
                                            static_cast<Manifold<dim>*>(new CylindricalManifold<dim>(direction, center)));
    create_triangulation(triangulation);
    triangulation.set_manifold(10, *cylinder_manifold);

    triangulation.refine_global(n_refine_space);
    dirichlet_boundary.insert(0);
    dirichlet_boundary.insert(2);
    neumann_boundary.insert(1);
#endif

    pcout << std::endl
          << "Generating grid for "     << dim << "-dimensional problem" << std::endl << std::endl
          << "  number of refinements:" << std::fixed << std::setw(10) << std::right << n_refine_space << std::endl
          << "  number of cells:      " << std::fixed << std::setw(10) << std::right << triangulation.n_global_active_cells() << std::endl
          << "  number of faces:      " << std::fixed << std::setw(10) << std::right << triangulation.n_active_faces() << std::endl
          << "  number of vertices:   " << std::fixed << std::setw(10) << std::right << triangulation.n_vertices() << std::endl;
  }

template<int dim>
void NavierStokesProblem<dim>::solve_problem(bool do_restart)
{
  create_grid();

  navier_stokes_operation->setup(periodic_faces, dirichlet_boundary, neumann_boundary);

  if(param.problem_type == ProblemType::Unsteady)
  {
    // setup time integrator before calling setup_solvers
    // (this is necessary since the setup of the solvers
    // depends on quantities such as the time_step_size or gamma0!!!)
    time_integrator->setup(do_restart);

    navier_stokes_operation->setup_solvers();

    PrintInputParams::print_solver_parameters(pcout,param);

    postprocessor->setup();

    time_integrator->timeloop();
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady->setup();

    navier_stokes_operation->setup_solvers();

    postprocessor->setup();

    driver_steady->solve_steady_problem();
  }
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
