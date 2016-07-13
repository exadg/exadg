
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

#include "../include/DGNavierStokesDualSplitting.h"
#include "../include/DGNavierStokesCoupled.h"

#include "../include/InputParameters.h"
#include "TimeIntBDFDualSplitting.h"
#include "TimeIntBDFCoupled.h"

#include "DriverSteadyProblems.h"

using namespace dealii;

//#define XWALL

// specify flow problem that has to be solved
//#define VORTEX
#define STOKES_GUERMOND
//#define STOKES_SHAHBAZI
//#define POISEUILLE
//#define CUETTE
//#define CAVITY
//#define KOVASZNAY
//#define BELTRAMI
//#define FLOW_PAST_CYLINDER
//#define CHANNEL


ProblemType PROBLEM_TYPE = ProblemType::Unsteady; //Steady; //Unsteady;
EquationType EQUATION_TYPE = EquationType::Stokes; // Stokes; // NavierStokes;
TreatmentOfConvectiveTerm TREATMENT_OF_CONVECTIVE_TERM = TreatmentOfConvectiveTerm::Explicit; // Explicit; // Implicit;

bool const DIVU_INTEGRATED_BY_PARTS = true; //false;//true;
bool const DIVU_USE_BOUNDARY_DATA = true; //false;//true;
bool const GRADP_INTEGRATED_BY_PARTS = true; //false;//true;
bool const GRADP_USE_BOUNDARY_DATA = true; //false;//true;

/************* temporal discretization ***********/
// which temporal discretization approach
TemporalDiscretization TEMPORAL_DISCRETIZATION = TemporalDiscretization::BDFCoupledSolution; //BDFDualSplittingScheme // BDFCoupledSolution

// type of time step calculation
TimeStepCalculation TIME_STEP_CALCULATION = TimeStepCalculation::ConstTimeStepUserSpecified; //ConstTimeStepUserSpecified; //ConstTimeStepCFL; //AdaptiveTimeStepCFL;
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
FormulationViscousTerm FORMULATION_VISCOUS_TERM = FormulationViscousTerm::DivergenceFormulation; //DivergenceFormulation; //LaplaceFormulation;
InteriorPenaltyFormulationViscous IP_FORMULATION_VISCOUS = InteriorPenaltyFormulationViscous::SIPG; //SIPG; //NIPG;
SolverViscous SOLVER_VISCOUS = SolverViscous::PCG; //PCG; //GMRES;
PreconditionerViscous PRECONDITIONER_VISCOUS = PreconditionerViscous::GeometricMultigrid; //None; //Jacobi; //InverseMassMatrix; //GeometricMultigrid;

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
  const unsigned int FE_DEGREE = 2; //2
  const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 1;
  const unsigned int REFINE_STEPS_SPACE_MAX = 1;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 0.1;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double STATISTICS_START_TIME = 0.0;
  const int STATISTICS_EVERY = 1;
  const bool DIVU_TIMESERIES = true;
  const bool COMPUTE_DIVERGENCE = true;
  const bool ANALYTICAL_SOLUTION = true;
  const int MAX_NUM_STEPS = 1e7;
  const double CFL = 0.1; //0.1;
  const double U_X_MAX = 1.0;
  const double MAX_VELOCITY = 1.4*U_X_MAX;
  const double TIME_STEP_SIZE = 1.e-1;//1.e-2;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 0.025;//0.01;
  // interior penalty method - penalty factor
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;

  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0; //PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool PURE_DIRICHLET_BC = false;

  const double ABS_TOL_NEWTON = 1.0e-12;
  const double REL_TOL_NEWTON = 1.0e-6;
  unsigned int const MAX_ITER_NEWTON = 1e2;
  const double ABS_TOL_LINEAR = 1.0e-12;
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
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
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

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
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
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
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

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
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
  const unsigned int FE_DEGREE = 5;
  const unsigned int FE_DEGREE_P = FE_DEGREE-1;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 2;
  const unsigned int REFINE_STEPS_SPACE_MIN = 4;
  const unsigned int REFINE_STEPS_SPACE_MAX = 4;

  const double START_TIME = 0.0;
  const double END_TIME = 40.0;
  const double OUTPUT_INTERVAL_TIME = 2.0;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool ANALYTICAL_SOLUTION = false;
  const bool DIVU_TIMESERIES = false; //true;
  const bool COMPUTE_DIVERGENCE = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.1;
  const double TIME_STEP_SIZE = 1.e-1;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 1.0e0;//0.0002;
  const double L = 1.0;

  const double MAX_VELOCITY = 1.0;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;
  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
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
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
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

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
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
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
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

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
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
  const unsigned int REFINE_STEPS_SPACE_MIN = 2;//2
  const unsigned int REFINE_STEPS_SPACE_MAX = 2;

  const double START_TIME = 0.0;
  const double END_TIME = 1.0;
  const double OUTPUT_INTERVAL_TIME = 1.0;//(END_TIME-START_TIME)/10.0;
  const double OUTPUT_START_TIME = 0.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
  const bool ANALYTICAL_SOLUTION = true;
  const bool DIVU_TIMESERIES = false;
  const bool COMPUTE_DIVERGENCE = false;
  const int MAX_NUM_STEPS = 1e6;
  const double CFL = 0.2; // CFL number irrelevant for Stokes flow problem
  const double TIME_STEP_SIZE = 1.e-1;//2.e-4;//1.e-1;///std::pow(2.,13); //5.0e-4
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 1.0e-6;//0.01;

  const double MAX_VELOCITY = 2.65; // MAX_VELOCITY also irrelevant
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;

  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0; //1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
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
  const double STATISTICS_START_TIME = 50.0;
  const int STATISTICS_EVERY = 1;
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

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
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
  const double STATISTICS_START_TIME = 50000.0;
  const int STATISTICS_EVERY = 1;
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

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = -10.0;
  const double GRID_STRETCH_FAC = 1.8;
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

#ifdef CHANNEL
  const unsigned int FE_DEGREE = 4;
  const unsigned int FE_DEGREE_P = FE_DEGREE;//FE_DEGREE-1;
  const unsigned int FE_DEGREE_XWALL = 1;
  const unsigned int N_Q_POINTS_1D_XWALL = 1;
  const unsigned int DIMENSION = 3; // DIMENSION >= 2
  const unsigned int REFINE_STEPS_SPACE_MIN = 2;
  const unsigned int REFINE_STEPS_SPACE_MAX = 2;

  const double START_TIME = 0.0;
  const double END_TIME = 70.0;
  const double OUTPUT_INTERVAL_TIME = 2.0;
  const double OUTPUT_START_TIME = 50.0;
  const double ERROR_CALC_INTERVAL_TIME = OUTPUT_INTERVAL_TIME;
  const double ERROR_CALC_START_TIME = OUTPUT_START_TIME;
  const double STATISTICS_START_TIME = 50.0;
  const bool ANALYTICAL_SOLUTION = false;
  const bool COMPUTE_DIVERGENCE = true;
  const bool DIVU_TIMESERIES = true;
  const int STATISTICS_EVERY = 1;
  const int MAX_NUM_STEPS = 1e7;
  const double CFL = 1.0;
  const double TIME_STEP_SIZE = 1.e-3;
  const unsigned int REFINE_STEPS_TIME_MIN = 0;
  const unsigned int REFINE_STEPS_TIME_MAX = 0;

  const double VISCOSITY = 1./180.0;

  const double MAX_VELOCITY = 15.0;
  const double IP_FACTOR_PRESSURE = 1.0;
  const double IP_FACTOR_VISCOUS = IP_FACTOR_PRESSURE;

  // projection step - divergence and continuity penalty factors
  const double PENALTY_FACTOR_DIVERGENCE = 1.0e0;
  const double PENALTY_FACTOR_CONTINUITY = 0.0e0;//PENALTY_FACTOR_DIVERGENCE;//0.0e0;

  const double CS = 0.0; // Smagorinsky constant
  const double ML = 0.0; // mixing-length model for xwall
  const bool VARIABLETAUW = false;
  const double DTAUW = 1.0;

  const double MAX_WDIST_XWALL = 0.2;
  const double GRID_STRETCH_FAC = 1.8;
  const bool PURE_DIRICHLET_BC = true;

  const double ABS_TOL_PRESSURE = 1.0e-12;
  const double REL_TOL_PRESSURE = 1.0e-4;
  const double ABS_TOL_VISCOUS = 1.0e-12;
  const double REL_TOL_VISCOUS = 1.0e-4;
  const double ABS_TOL_PROJECTION = 1.0e-12;
  const double REL_TOL_PROJECTION = 1.0e-4;

  // show solver performance (wall time, number of iterations) every ... timesteps
  const unsigned int OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS = 1e1;

  const std::string OUTPUT_PREFIX = "ch180_l2_p4_CFL_1-0_WeakProj_K_1"; //"ch180_l2_p4_test";

  const unsigned int ORDER_TIME_INTEGRATOR = 3;
  const bool START_WITH_LOW_ORDER = true;
#endif

  InputParameters::InputParameters():
  problem_type(PROBLEM_TYPE),
  equation_type(EQUATION_TYPE),
  treatment_of_convective_term(TREATMENT_OF_CONVECTIVE_TERM),
  start_time(START_TIME),
  end_time(END_TIME),
  max_number_of_steps(MAX_NUM_STEPS),
  calculation_of_time_step_size(TIME_STEP_CALCULATION),
  cfl(CFL),
  max_velocity(MAX_VELOCITY),
  time_step_size(TIME_STEP_SIZE),
  viscosity(VISCOSITY),
  temporal_discretization(TEMPORAL_DISCRETIZATION),
  order_time_integrator(ORDER_TIME_INTEGRATOR),
  start_with_low_order(START_WITH_LOW_ORDER),
  use_symmetric_saddle_point_matrix(USE_SYMMETRIC_SADDLE_POINT_MATRIX),
  small_time_steps_stability(STS_STABILITY),
  pure_dirichlet_bc(PURE_DIRICHLET_BC),
  penalty_factor_divergence(PENALTY_FACTOR_DIVERGENCE),
  penalty_factor_continuity(PENALTY_FACTOR_CONTINUITY),
  compute_divergence(COMPUTE_DIVERGENCE),
  divu_integrated_by_parts(DIVU_INTEGRATED_BY_PARTS),
  divu_use_boundary_data(DIVU_USE_BOUNDARY_DATA),
  gradp_integrated_by_parts(GRADP_INTEGRATED_BY_PARTS),
  gradp_use_boundary_data(GRADP_USE_BOUNDARY_DATA),
  IP_factor_pressure(IP_FACTOR_PRESSURE),
  IP_factor_viscous(IP_FACTOR_VISCOUS),
  abs_tol_newton(ABS_TOL_NEWTON),
  rel_tol_newton(REL_TOL_NEWTON),
  max_iter_newton(MAX_ITER_NEWTON),
  abs_tol_linear(ABS_TOL_LINEAR),
  rel_tol_linear(REL_TOL_LINEAR),
  max_iter_linear(MAX_ITER_LINEAR),
  abs_tol_pressure(ABS_TOL_PRESSURE),
  rel_tol_pressure(REL_TOL_PRESSURE),
  abs_tol_projection(ABS_TOL_PROJECTION),
  rel_tol_projection(REL_TOL_PROJECTION),
  abs_tol_viscous(ABS_TOL_VISCOUS),
  rel_tol_viscous(REL_TOL_VISCOUS),
  solver_poisson(SOLVER_POISSON),
  preconditioner_poisson(PRECONDITIONER_POISSON),
  multigrid_smoother(MULTIGRID_SMOOTHER),
  multigrid_coarse_grid_solver(MULTIGRID_COARSE_GRID_SOLVER),
  projection_type(PROJECTION_TYPE),
  solver_projection(SOLVER_PROJECTION),
  preconditioner_projection(PRECONDITIONER_PROJECTION),
  formulation_viscous_term(FORMULATION_VISCOUS_TERM),
  IP_formulation_viscous(IP_FORMULATION_VISCOUS),
  solver_viscous(SOLVER_VISCOUS),
  preconditioner_viscous(PRECONDITIONER_VISCOUS),
  multigrid_smoother_viscous(MULTIGRID_SMOOTHER_VISCOUS),
  multigrid_coarse_grid_solver_viscous(MULTIGRID_COARSE_GRID_SOLVER_VISCOUS),
  preconditioner_linearized_navier_stokes(PRECONDITIONER_LINEARIZED_NAVIER_STOKES),
  preconditioner_momentum(PRECONDITIONER_MOMENTUM),
  preconditioner_schur_complement(PRECONDITIONER_SCHUR_COMPLEMENT),
  output_solver_info_every_timesteps(OUTPUT_SOLVER_INFO_EVERY_TIMESTEPS),
  output_start_time(OUTPUT_START_TIME),
  output_interval_time(OUTPUT_INTERVAL_TIME),
  output_prefix(OUTPUT_PREFIX),
  error_calc_start_time(ERROR_CALC_START_TIME),
  error_calc_interval_time(ERROR_CALC_INTERVAL_TIME),
  analytical_solution_available(ANALYTICAL_SOLUTION),
  statistics_start_time(STATISTICS_START_TIME),
  statistics_every(STATISTICS_EVERY),
  cs(CS),
  ml(ML),
  variabletauw(VARIABLETAUW),
  dtauw(DTAUW)
  {}

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

    /****************** turbulent channel flow **************************/
#ifdef CHANNEL
    if(component == 0)
    {
      if(p[1]<0.9999 && p[1]>-0.9999)
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

  private:
    const double time;
  };

  template<int dim>
  double RHS<dim>::value(const Point<dim> &p,const unsigned int component) const
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

    virtual UpdateFlags get_needed_update_flags () const
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
          const double enrichment_func = SimpleSpaldingsLaw::SpaldingsLaw(wdist,sqrt(uh[q](dim)),VISCOSITY);
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
      Assert (uh[0].size() == dim+number_vorticity_components, ExcInternalError());

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

  template <int dim>
  Point<dim> grid_transform (const Point<dim> &in);

  template<int dim>
  class PostProcessor
  {
  public:
    typedef typename DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>::value_type value_type;

    PostProcessor(//DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> const &ns_operation,
                  std_cxx11::shared_ptr< const DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> >  ns_operation,
                  InputParameters const &param_in):
      ns_operation_(ns_operation),
      statistics(ns_operation->get_dof_handler_u()),
      param(param_in),
      time_(0.0),
      time_step_number_(1),
      output_counter_(0),
      error_counter_(0),
      num_samp_(0),
      div_samp_(0.0),
      mass_samp_(0.0)
    {

    }

    void setup()
    {
#ifdef CHANNEL
      statistics.setup(&grid_transform<dim>);
#endif
    }

    void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                           parallel::distributed::Vector<double> const &pressure,
                           parallel::distributed::Vector<double> const &vorticity,
                           parallel::distributed::Vector<double> const &divergence,
                           double const time,
                           unsigned int const time_step_number)
    {
      time_ = time;
      time_step_number_ = time_step_number;

      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
      if( time > (param.output_start_time + output_counter_*param.output_interval_time - EPSILON))
      {
        write_output(velocity,pressure,vorticity,divergence);
        ++output_counter_;
      }

      if( (param.analytical_solution_available == true) &&
          (time > (param.error_calc_start_time + error_counter_*param.error_calc_interval_time - EPSILON)) )
      {
        calculate_error(velocity,pressure);
        ++error_counter_;
      }

#ifdef FLOW_PAST_CYLINDER
      compute_lift_and_drag(velocity,pressure,time_step_number_== 1);
      compute_pressure_difference(pressure,time_step_number_ == 1);
#endif

#ifdef CHANNEL
      if(time > param.statistics_start_time-EPSILON && time_step_number % param.statistics_every == 0)
      {
        statistics.evaluate(velocity);
        if(time_step_number % 100 == 0 || time > (param.end_time-EPSILON))
          statistics.write_output(param.output_prefix,ns_operation_.get_viscosity());
      }
#endif
    };

    // postprocessing for steady-state problems
    void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                           parallel::distributed::Vector<double> const &pressure,
                           parallel::distributed::Vector<double> const &vorticity,
                           parallel::distributed::Vector<double> const &divergence)
    {
      write_output(velocity,pressure,vorticity,divergence);
      ++output_counter_;

      if(param.analytical_solution_available == true)
      {
        calculate_error(velocity,pressure);
      }
    };

    void analyze_divergence_error(parallel::distributed::Vector<double> const &velocity_temp,
                                  double const time,
                                  unsigned int const time_step_number)
    {
      time_ = time;
      time_step_number_ = time_step_number;

      const double EPSILON = 1.0e-10; // small number which is much smaller than the time step size
      if(time > param.statistics_start_time-EPSILON && time_step_number % param.statistics_every == 0)
      {
          write_divu_statistics(velocity_temp);
      }

      write_divu_timeseries(velocity_temp);
    }

  private:
    std_cxx11::shared_ptr< const DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL> >  ns_operation_;
    StatisticsManager<dim> statistics;
    InputParameters const & param;

    double time_;
    unsigned int time_step_number_;
    unsigned int output_counter_;
    unsigned int error_counter_;

    int num_samp_;
    double div_samp_;
    double mass_samp_;

    void calculate_error(parallel::distributed::Vector<double> const &velocity,
                         parallel::distributed::Vector<double> const      &pressure);

    void write_output(parallel::distributed::Vector<double> const &velocity,
                      parallel::distributed::Vector<double> const &pressure,
                      parallel::distributed::Vector<double> const &vorticity,
                      parallel::distributed::Vector<double> const &divergence);

    void compute_lift_and_drag(parallel::distributed::Vector<double> const &velocity,
                               parallel::distributed::Vector<double> const &pressure,
                               bool const                                  clear_files) const;

    void compute_pressure_difference(parallel::distributed::Vector<double> const &pressure,
                                     bool const                                  clear_files) const;

    void my_point_value(const Mapping<dim>                                                            &mapping,
                        const DoFHandler<dim>                                                         &dof_handler,
                        const parallel::distributed::Vector<double>                                   &solution,
                        const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >  &cell_point,
                        Vector<double>                                                                &value) const;

    void write_divu_statistics(parallel::distributed::Vector<double> const &velocity_temp);
    void write_divu_timeseries(parallel::distributed::Vector<double> const &velocity_temp);

    void local_compute_divu_for_channel_stats(const MatrixFree<dim,value_type>                &data,
                                              std::vector<double >                            &test,
                                              const parallel::distributed::Vector<value_type> &source,
                                              const std::pair<unsigned int,unsigned int>      &cell_range) const;

    void local_compute_divu_for_channel_stats_face (const MatrixFree<dim,double>                    &data,
                                                    std::vector<double >                            &test,
                                                    const parallel::distributed::Vector<value_type> &source,
                                                    const std::pair<unsigned int,unsigned int>      &face_range) const;

    void local_compute_divu_for_channel_stats_boundary_face (const MatrixFree<dim,double>                    &data,
                                                             std::vector<double >                            &test,
                                                             const parallel::distributed::Vector<value_type> &source,
                                                             const std::pair<unsigned int,unsigned int>       &face_range) const;

  };

  template<int dim>
  void PostProcessor<dim>::
  calculate_error(parallel::distributed::Vector<double> const  &velocity,
                  parallel::distributed::Vector<double> const  &pressure)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Calculate error at time t = " << std::scientific << std::setprecision(4) << time_ << ":" << std::endl;

    Vector<double> error_norm_per_cell_u (ns_operation_->get_dof_handler_u().get_triangulation().n_active_cells());
    Vector<double> solution_norm_per_cell_u (ns_operation_->get_dof_handler_u().get_triangulation().n_active_cells());
    VectorTools::integrate_difference (ns_operation_->get_mapping(),
                                       ns_operation_->get_dof_handler_u(),
                                       velocity,
                                       AnalyticalSolution<dim>(true,time_),
                                       error_norm_per_cell_u,
                                       QGauss<dim>(ns_operation_->get_fe_u().degree+4),//(fe().degree+2),
                                       VectorTools::L2_norm);
    parallel::distributed::Vector<double> dummy_u;
    dummy_u.reinit(velocity);
    VectorTools::integrate_difference (ns_operation_->get_mapping(),
                                       ns_operation_->get_dof_handler_u(),
                                       dummy_u,
                                       AnalyticalSolution<dim>(true,time_),
                                       solution_norm_per_cell_u,
                                       QGauss<dim>(ns_operation_->get_fe_u().degree+4), //(fe().degree+2),
                                       VectorTools::L2_norm);
    double error_norm_u = std::sqrt(Utilities::MPI::sum (error_norm_per_cell_u.norm_sqr(), MPI_COMM_WORLD));
    double solution_norm_u = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell_u.norm_sqr(), MPI_COMM_WORLD));
    if(solution_norm_u > 1.e-12)
      pcout << "  Relative error (L2-norm) velocity u: "
            << std::scientific << std::setprecision(5) << error_norm_u/solution_norm_u << std::endl;
    else
      pcout << "  ABSOLUTE error (L2-norm) velocity u: "
            << std::scientific << std::setprecision(5) << error_norm_u << std::endl;

    Vector<double> error_norm_per_cell_p (ns_operation_->get_dof_handler_u().get_triangulation().n_active_cells());
    Vector<double> solution_norm_per_cell_p (ns_operation_->get_dof_handler_u().get_triangulation().n_active_cells());
    VectorTools::integrate_difference (ns_operation_->get_mapping(),
                                       ns_operation_->get_dof_handler_p(),
                                       pressure,
                                       AnalyticalSolution<dim>(false,time_),
                                       error_norm_per_cell_p,
                                       QGauss<dim>(ns_operation_->get_fe_p().degree+4), //(fe_p.degree+2),
                                       VectorTools::L2_norm);

    parallel::distributed::Vector<double> dummy_p;
    dummy_p.reinit(pressure);
    VectorTools::integrate_difference (ns_operation_->get_mapping(),
                                       ns_operation_->get_dof_handler_p(),
                                       dummy_p,
                                       AnalyticalSolution<dim>(false,time_),
                                       solution_norm_per_cell_p,
                                       QGauss<dim>(ns_operation_->get_fe_p().degree+4), //(fe_p.degree+2),
                                       VectorTools::L2_norm);

    double error_norm_p = std::sqrt(Utilities::MPI::sum (error_norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));
    double solution_norm_p = std::sqrt(Utilities::MPI::sum (solution_norm_per_cell_p.norm_sqr(), MPI_COMM_WORLD));
    if(solution_norm_p > 1.e-12)
      pcout << "  Relative error (L2-norm) pressure p: "
            << std::scientific << std::setprecision(5) << error_norm_p/solution_norm_p << std::endl;
    else
      pcout << "  ABSOLUTE error (L2-norm) pressure p: "
            << std::scientific << std::setprecision(5) << error_norm_p << std::endl;
  }

  template<int dim>
  void PostProcessor<dim>::
  write_output(parallel::distributed::Vector<double> const &velocity,
               parallel::distributed::Vector<double> const &pressure,
               parallel::distributed::Vector<double> const &vorticity,
               parallel::distributed::Vector<double> const &divergence)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "OUTPUT << Write data at time t = " << std::scientific << std::setprecision(4) << time_ << std::endl;

//    const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
//    // velocity + xwall dofs
//    const FESystem<dim> joint_fe (ns_operation_.get_fe_u(), dim,
//#ifdef CHANNEL
//                                  ns_operation_.get_XWall().ReturnFE(), 1,
//                                  ns_operation_.get_fe_xwall(), dim,
//#endif
//                                  ns_operation_.get_fe_u(), number_vorticity_components
//#ifdef CHANNEL
//                                  ,ns_operation_.get_fe_xwall(), number_vorticity_components
//#endif
//                                  );
//    DoFHandler<dim> joint_dof_handler (ns_operation_.get_dof_handler_u().get_triangulation());
//    joint_dof_handler.distribute_dofs (joint_fe);
//    IndexSet joint_relevant_set;
//    DoFTools::extract_locally_relevant_dofs(joint_dof_handler, joint_relevant_set);
//    parallel::distributed::Vector<double>
//      joint_solution (joint_dof_handler.locally_owned_dofs(), joint_relevant_set, MPI_COMM_WORLD);
//    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
//      loc_vel_dof_indices (ns_operation_.get_fe_u().dofs_per_cell)
//#ifdef CHANNEL
//      , loc_pre_dof_indices(ns_operation_.get_XWall().ReturnFE().dofs_per_cell),
//      loc_vel_xwall_dof_indices(ns_operation_.get_fe_xwall().dofs_per_cell)
//#endif
//      ;
//    typename DoFHandler<dim>::active_cell_iterator
//      joint_cell = joint_dof_handler.begin_active(),
//      joint_endc = joint_dof_handler.end(),
//      vel_cell = ns_operation_.get_dof_handler_u().begin_active()
//#ifdef CHANNEL
//      ,pre_cell = ns_operation_.get_XWall().ReturnDofHandlerWallDistance().begin_active(),
//      vel_cell_xwall = ns_operation_.get_dof_handler_xwall().begin_active()
//#endif
//;
//
//    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell
//#ifdef CHANNEL
//    , ++ pre_cell, ++vel_cell_xwall
//#endif
//    )
//      if (joint_cell->is_locally_owned())
//      {
//        joint_cell->get_dof_indices (loc_joint_dof_indices);
//        vel_cell->get_dof_indices (loc_vel_dof_indices);
//#ifdef CHANNEL
//        pre_cell->get_dof_indices (loc_pre_dof_indices);
//        vel_cell_xwall->get_dof_indices (loc_vel_xwall_dof_indices);
//#endif
//        for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
//          switch (joint_fe.system_to_base_index(i).first.first)
//            {
//            case 0: //velocity
//              Assert (joint_fe.system_to_base_index(i).first.second < dim,
//                      ExcInternalError());
//              joint_solution (loc_joint_dof_indices[i]) =
//                velocity.block(joint_fe.system_to_base_index(i).first.second)
//                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
//              break;
//#ifdef CHANNEL
//            case 1: //tauw, necessary to reconstruct velocity
//              Assert (joint_fe.system_to_base_index(i).first.second == 0,
//                      ExcInternalError());
//              joint_solution (loc_joint_dof_indices[i]) =
//                  (ns_operation_.get_fe_parameters().xwallstatevec[1])
//                (loc_pre_dof_indices[ joint_fe.system_to_base_index(i).second ]);
//              break;
//            case 2: //velocity_xwall
//              Assert (joint_fe.system_to_base_index(i).first.second < dim,
//                      ExcInternalError());
//              joint_solution (loc_joint_dof_indices[i]) =
//                velocity.block( dim + joint_fe.system_to_base_index(i).first.second ) //TODO: Benjamin: check indices
//                (loc_vel_xwall_dof_indices[ joint_fe.system_to_base_index(i).second ]);
//              break;
//            case 3: //vorticity
//#else
//            case 1:
//#endif
//              Assert (joint_fe.system_to_base_index(i).first.second < dim,
//                      ExcInternalError());
//              joint_solution (loc_joint_dof_indices[i]) =
//                vorticity.block(joint_fe.system_to_base_index(i).first.second)
//                (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
//              break;
//#ifdef CHANNEL
//            case 4: //vorticity_xwall
//              Assert (joint_fe.system_to_base_index(i).first.second < dim,
//                      ExcInternalError());
//              joint_solution (loc_joint_dof_indices[i]) =
//                vorticity.block(dim + joint_fe.system_to_base_index(i).first.second) //TODO: Benjamin: check indices
//                (loc_vel_xwall_dof_indices[ joint_fe.system_to_base_index(i).second ]);
//              break;
//#endif
//            default:
//              Assert (false, ExcInternalError());
//              break;
//            }
//      }
//
//  joint_solution.update_ghost_values();
//
//  Postprocessor<dim> postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
//
//  DataOut<dim> data_out;
//  data_out.attach_dof_handler(joint_dof_handler);
//  data_out.add_data_vector(joint_solution, postprocessor);
//#ifdef CHANNEL
//  data_out.add_data_vector (ns_operation_.get_XWall().ReturnDofHandlerWallDistance(),ns_operation_.get_fe_parameters().xwallstatevec[0], "wdist");
//#endif

  DataOut<dim> data_out;
  std::vector<std::string> velocity_names (dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    velocity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (ns_operation_->get_dof_handler_u(),velocity, velocity_names, velocity_component_interpretation);

  std::vector<std::string> vorticity_names (dim, "vorticity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    vorticity_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector (ns_operation_->get_dof_handler_u(),vorticity, vorticity_names, vorticity_component_interpretation);

  pressure.update_ghost_values();
  data_out.add_data_vector (ns_operation_->get_dof_handler_p(),pressure, "p");

  if(COMPUTE_DIVERGENCE == true)
  {
    std::vector<std::string> divergence_names (dim, "divergence");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      divergence_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (ns_operation_->get_dof_handler_u(),divergence, divergence_names, divergence_component_interpretation);
  }

  std::ostringstream filename;
  filename << "output/"
           << param.output_prefix
           << "_Proc"
           << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
           << "_"
           << output_counter_
           << ".vtu";

  data_out.build_patches (ns_operation_->get_mapping(),5);

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);

  if ( Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);++i)
    {
      std::ostringstream filename;
      filename << param.output_prefix
               << "_Proc"
               << i
               << "_"
               << output_counter_
               << ".vtu";

        filenames.push_back(filename.str().c_str());
    }
    std::string master_name = "output/" + param.output_prefix + "_" + Utilities::int_to_string(output_counter_) + ".pvtu";
    std::ofstream master_output (master_name.c_str());
    data_out.write_pvtu_record (master_output, filenames);
  }
  }

  template<int dim>
  void PostProcessor<dim>::
  compute_lift_and_drag(parallel::distributed::Vector<double> const &velocity,
                        parallel::distributed::Vector<double> const &pressure,
                        const bool clear_files) const
  {
#ifdef FLOW_PAST_CYLINDER
    FEFaceEvaluation<dim,FE_DEGREE,FE_DEGREE+1,dim,value_type> fe_eval_velocity(ns_operation_->get_data(),true,0,0);
    FEFaceEvaluation<dim,FE_DEGREE_P,FE_DEGREE+1,1,value_type> fe_eval_pressure(ns_operation_->get_data(),true,1,0);

    Tensor<1,dim,value_type> Force;
    for(unsigned int d=0;d<dim;++d)
      Force[d] = 0.0;

    for(unsigned int face=ns_operation_->get_data().n_macro_inner_faces(); face<(ns_operation_->get_data().n_macro_inner_faces()+ns_operation_->get_data().n_macro_boundary_faces()); face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity.read_dof_values(velocity);
      fe_eval_velocity.evaluate(false,true);

      fe_eval_pressure.reinit (face);
      fe_eval_pressure.read_dof_values(pressure);
      fe_eval_pressure.evaluate(true,false);

      if (ns_operation_->get_data().get_boundary_indicator(face) == 2)
      {
        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          VectorizedArray<value_type> pressure = fe_eval_pressure.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
          Tensor<2,dim,VectorizedArray<value_type> > velocity_gradient = fe_eval_velocity.get_gradient(q);
          fe_eval_velocity.submit_value(pressure*normal -  make_vectorized_array<value_type>(ns_operation_->get_viscosity())*
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
    const double U = Um * (dim==2 ? 2./3. : 4./9.);
    if(dim == 2)
      Force *= 2.0/pow(U,2.0)/D;
    else if(dim == 3)
      Force *= 2.0/pow(U,2.0)/D/H;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::string filename_drag, filename_lift;
      filename_drag = "output/drag_refine" + Utilities::int_to_string(ns_operation_->get_dof_handler_u().get_triangulation().n_levels()-1) + "_fedegree" + Utilities::int_to_string(FE_DEGREE) + ".txt";
      filename_lift = "output/lift_refine" + Utilities::int_to_string(ns_operation_->get_dof_handler_u().get_triangulation().n_levels()-1) + "_fedegree" + Utilities::int_to_string(FE_DEGREE) + ".txt";

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
      f_drag << std::scientific << std::setprecision(6) << time_ << "\t" << Force[0] << std::endl;
      f_drag.close();
      f_lift << std::scientific << std::setprecision(6) << time_ << "\t" << Force[1] << std::endl;
      f_lift.close();
    }
#endif
  }

  template<int dim>
  void PostProcessor<dim>::
  compute_pressure_difference(parallel::distributed::Vector<double> const &pressure,
                              const bool                                  clear_files) const
  {
#ifdef FLOW_PAST_CYLINDER
    double pressure_1 = 0.0, pressure_2 = 0.0;
    unsigned int counter_1 = 0, counter_2 = 0;

    Point<dim> point_1, point_2;
    if(dim == 2)
    {
      Point<dim> point_1_2D((X_C-D/2.0),Y_C), point_2_2D((X_C+D/2.0),Y_C);
      point_1 = point_1_2D;
      point_2 = point_2_2D;
    }
    else if(dim == 3)
    {
      Point<dim> point_1_3D((X_C-D/2.0),Y_C,H/2.0), point_2_3D((X_C+D/2.0),Y_C,H/2.0);
      point_1 = point_1_3D;
      point_2 = point_2_3D;
    }

    // parallel computation
    const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
    cell_point_1 = GridTools::find_active_cell_around_point (ns_operation_->get_mapping(),ns_operation_->get_dof_handler_p(), point_1);
    if(cell_point_1.first->is_locally_owned())
    {
      counter_1 = 1;
      //std::cout<< "Point 1 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;

      Vector<double> value(1);
      my_point_value(ns_operation_->get_mapping(),ns_operation_->get_dof_handler_p(),pressure,cell_point_1,value);
      pressure_1 = value(0);
    }
    counter_1 = Utilities::MPI::sum(counter_1,MPI_COMM_WORLD);
    pressure_1 = Utilities::MPI::sum(pressure_1,MPI_COMM_WORLD);
    pressure_1 = pressure_1/counter_1;

    const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> >
    cell_point_2 = GridTools::find_active_cell_around_point (ns_operation_->get_mapping(),ns_operation_->get_dof_handler_p(), point_2);
    if(cell_point_2.first->is_locally_owned())
    {
      counter_2 = 1;
      //std::cout<< "Point 2 found on Processor "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;

      Vector<double> value(1);
      my_point_value(ns_operation_->get_mapping(),ns_operation_->get_dof_handler_p(),pressure,cell_point_2,value);
      pressure_2 = value(0);
    }
    counter_2 = Utilities::MPI::sum(counter_2,MPI_COMM_WORLD);
    pressure_2 = Utilities::MPI::sum(pressure_2,MPI_COMM_WORLD);
    pressure_2 = pressure_2/counter_2;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::string filename = "output/pressure_difference_refine" + Utilities::int_to_string(ns_operation_->get_dof_handler_u().get_triangulation().n_levels()-1) + "_fedegree" + Utilities::int_to_string(FE_DEGREE) + ".txt";

      std::ofstream f;
      if(clear_files)
      {
        f.open(filename.c_str(),std::ios::trunc);
      }
      else
      {
        f.open(filename.c_str(),std::ios::app);
      }
      f << std::scientific << std::setprecision(6) << time_ << "\t" << pressure_1-pressure_2 << std::endl;
      f.close();
    }
#endif
  }

  template<int dim>
  void PostProcessor<dim>::
  my_point_value(const Mapping<dim>                                                           &mapping,
                 const DoFHandler<dim>                                                        &dof_handler,
                 const parallel::distributed::Vector<double>                                  &solution,
                 const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > &cell_point,
                 Vector<double>                                                               &value) const
  {
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,ExcInternalError());

    const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(cell_point.first);

    // then use this to get at the values of the given fe_function at this point
    std::vector<Vector<double> > u_value(1, Vector<double> (fe.n_components()));
    fe_values.get_function_values(solution, u_value);
    value = u_value[0];
  }

  template<int dim>
  void PostProcessor<dim>::
  write_divu_statistics(parallel::distributed::Vector<double> const &velocity_temp)
  {
    ++num_samp_;

    std::vector<double > dst(4,0.0);
    ns_operation_->get_data().loop (&PostProcessor<dim>::local_compute_divu_for_channel_stats,
                                   &PostProcessor<dim>::local_compute_divu_for_channel_stats_face,
                                   &PostProcessor<dim>::local_compute_divu_for_channel_stats_boundary_face,
                                   this, dst, velocity_temp);

    double divergence = Utilities::MPI::sum (dst.at(0), MPI_COMM_WORLD);
    double volume = Utilities::MPI::sum (dst.at(1), MPI_COMM_WORLD);
    double diff_mass = Utilities::MPI::sum (dst.at(2), MPI_COMM_WORLD);
    double mean_mass = Utilities::MPI::sum (dst.at(3), MPI_COMM_WORLD);
    div_samp_ += divergence/volume;
    mass_samp_ += diff_mass/mean_mass;
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::ostringstream filename;
      filename << param.output_prefix << ".divu_statistics";

      std::ofstream f;

      f.open(filename.str().c_str(),std::ios::trunc);
      f << "average divergence over space and time" << std::endl;
      f << "number of samples:   " << num_samp_ << std::endl;
      f << "Mean error incompressibility constraint:   " << div_samp_/num_samp_ << std::endl;
      f << "Mean error mass flux over interior element faces:  " << mass_samp_/num_samp_ << std::endl;
      f.close();
    }
  }

  template<int dim>
  void PostProcessor<dim>::
  write_divu_timeseries(parallel::distributed::Vector<double> const &velocity_temp)
  {
    std::vector<double> dst(4,0.0);
    ns_operation_->get_data().loop (&PostProcessor<dim>::local_compute_divu_for_channel_stats,
                                   &PostProcessor<dim>::local_compute_divu_for_channel_stats_face,
                                   &PostProcessor<dim>::local_compute_divu_for_channel_stats_boundary_face,
                                   this, dst, velocity_temp);

    double divergence = Utilities::MPI::sum (dst.at(0), MPI_COMM_WORLD);
    double volume = Utilities::MPI::sum (dst.at(1), MPI_COMM_WORLD);
    double diff_mass = Utilities::MPI::sum (dst.at(2), MPI_COMM_WORLD);
    double mean_mass = Utilities::MPI::sum (dst.at(3), MPI_COMM_WORLD);
    double div_normalized = divergence/volume;
    double diff_mass_normalized = diff_mass/mean_mass;
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::ostringstream filename;
      filename << param.output_prefix << ".divu_timeseries";

      std::ofstream f;
      if(time_step_number_==1)
      {
        f.open(filename.str().c_str(),std::ios::trunc);
        f << "Error incompressibility constraint:\n\n\t(1,|divu|)_Omega/(1,1)_Omega\n" << std::endl
          << "Error mass flux over interior element faces:\n\n\t(1,|(um - up)*n|)_dOmegaI / (1,|0.5(um + up)*n|)_dOmegaI\n" << std::endl
          << "       n       |       t      |    divergence    |      mass       " << std::endl;
      }
      else
      {
        f.open(filename.str().c_str(),std::ios::app);
      }
      f << std::setw(15) <<time_step_number_;
      f << std::scientific<<std::setprecision(7) << std::setw(15) << time_;
      f << std::scientific<<std::setprecision(7) << std::setw(15) << div_normalized;
      f << std::scientific<<std::setprecision(7) << std::setw(15) << diff_mass_normalized << std::endl;
    }
  }

  template<int dim>
  void PostProcessor<dim>::
  local_compute_divu_for_channel_stats(const MatrixFree<dim,value_type>                &data,
                                       std::vector<double>                             &dst,
                                       const parallel::distributed::Vector<value_type> &source,
                                       const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
#ifdef XWALL
    FEEvaluationXWall<dim,FE_DEGREE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,dim,value_type> phi(data,ns_operation_.get_xwallstatevec()[0],ns_operation_.get_xwallstatevec()[1],0,3);
#else
//    FEEvaluationXWall<dim,FE_DEGREE,FE_DEGREE_XWALL,FE_DEGREE+1,dim,value_type> phi(data,ns_operation_.get_fe_parameters(),0,0);
    FEEvaluation<dim,FE_DEGREE,FE_DEGREE+1,dim,value_type> phi(data,0,0);
#endif
    AlignedVector<VectorizedArray<value_type> > JxW_values(phi.n_q_points);
    VectorizedArray<value_type> div_vec = make_vectorized_array(0.);
    VectorizedArray<value_type> vol_vec = make_vectorized_array(0.);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(source);
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
    dst.at(0) += div;
    dst.at(1) += vol;
  }

  template<int dim>
  void PostProcessor<dim>::
  local_compute_divu_for_channel_stats_face (const MatrixFree<dim,double>                    &data,
                                             std::vector<double >                            &dst,
                                             const parallel::distributed::Vector<value_type> &source,
                                             const std::pair<unsigned int,unsigned int>      &face_range) const
  {
#ifdef XWALL
    FEFaceEvaluationXWall<dim,FE_DEGREE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,dim,value_type> fe_eval_xwall(data,ns_operation_.get_xwallstatevec()[0],ns_operation_.get_xwallstatevec()[1],true,0,3);
    FEFaceEvaluationXWall<dim,FE_DEGREE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,dim,value_type> fe_eval_xwall_neighbor(data,ns_operation_.get_xwallstatevec()[0],ns_operation_.get_xwallstatevec()[1],false,0,3);
#else
//    FEFaceEvaluationXWall<dim,FE_DEGREE,FE_DEGREE_XWALL,FE_DEGREE+1,dim,value_type> fe_eval_xwall(data,ns_operation_.get_fe_parameters(),true,0,0);
//    FEFaceEvaluationXWall<dim,FE_DEGREE,FE_DEGREE_XWALL,FE_DEGREE+1,dim,value_type> fe_eval_xwall_neighbor(data,ns_operation_.get_fe_parameters(),false,0,0);
    FEFaceEvaluation<dim,FE_DEGREE,FE_DEGREE+1,dim,value_type> fe_eval_xwall(data,true,0,0);
    FEFaceEvaluation<dim,FE_DEGREE,FE_DEGREE+1,dim,value_type> fe_eval_xwall_neighbor(data,false,0,0);
#endif
    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval_xwall.n_q_points);
    VectorizedArray<value_type> diff_mass_flux_vec = make_vectorized_array(0.);
    VectorizedArray<value_type> mean_mass_flux_vec = make_vectorized_array(0.);
    for (unsigned int face=face_range.first; face<face_range.second; ++face)
    {
      fe_eval_xwall.reinit(face);
      fe_eval_xwall.read_dof_values(source);
      fe_eval_xwall.evaluate(true,false);
      fe_eval_xwall_neighbor.reinit(face);
      fe_eval_xwall_neighbor.read_dof_values(source);
      fe_eval_xwall_neighbor.evaluate(true,false);
      fe_eval_xwall.fill_JxW_values(JxW_values);

      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
      {
        mean_mass_flux_vec += JxW_values[q]*std::abs(0.5*(fe_eval_xwall.get_value(q)+fe_eval_xwall_neighbor.get_value(q))*fe_eval_xwall.get_normal_vector(q));

        diff_mass_flux_vec += JxW_values[q]*std::abs((fe_eval_xwall.get_value(q)-fe_eval_xwall_neighbor.get_value(q))*fe_eval_xwall.get_normal_vector(q));
      }
    }
    value_type diff_mass_flux = 0.;
    value_type mean_mass_flux = 0.;
    for (unsigned int v=0;v<VectorizedArray<value_type>::n_array_elements;v++)
    {
      diff_mass_flux += diff_mass_flux_vec[v];
      mean_mass_flux += mean_mass_flux_vec[v];
    }
    dst.at(2) += diff_mass_flux;
    dst.at(3) += mean_mass_flux;
  }

  template<int dim>
  void PostProcessor<dim>::
  local_compute_divu_for_channel_stats_boundary_face (const MatrixFree<dim,double>                     &,
                                                      std::vector<double >                             &,
                                                      const parallel::distributed::Vector<value_type>  &,
                                                      const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  template<int dim>
  class NavierStokesProblem
  {
  public:
    typedef typename DGNavierStokesBase<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>::value_type value_type;
    NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time=0);
    void solve_problem();

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

    std_cxx11::shared_ptr<PostProcessor<dim> > postprocessor;

    std_cxx11::shared_ptr<TimeIntBDF<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type> > time_integrator;

    std_cxx11::shared_ptr<DriverSteadyProblems<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type> > driver_steady;
  };

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps_space, const unsigned int refine_steps_time):
  pcout (std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD,dealii::Triangulation<dim>::none,parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space(refine_steps_space)
  {
    pcout << std::endl << std::endl << std::endl
    << "_________________________________________________________________________________" << std::endl
    << "                                                                                 " << std::endl
    << "                High-order discontinuous Galerkin solver for the                 " << std::endl
    << "                     incompressible Navier-Stokes equations                      " << std::endl
    << "                based on a semi-explicit dual-splitting approach                 " << std::endl
    << "_________________________________________________________________________________" << std::endl
    << std::endl;

    if(param.problem_type == ProblemType::Steady)
    {
      // initialize navier_stokes_operation
      navier_stokes_operation.reset(new DGNavierStokesCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL>
          (triangulation,param));
      // initialize postprocessor after initializing navier_stokes_operation
      postprocessor.reset(new PostProcessor<dim>(navier_stokes_operation,param));
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
      postprocessor.reset(new PostProcessor<dim>(navier_stokes_operation,param));
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
      postprocessor.reset(new PostProcessor<dim>(navier_stokes_operation,param));
      // initialize time integrator that depends on both navier_stokes_operation and postprocessor
      time_integrator.reset(new TimeIntBDFCoupled<dim, FE_DEGREE, FE_DEGREE_P, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, value_type>(
          navier_stokes_operation,postprocessor,param,refine_steps_time));
    }
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
    //turbulent channel flow
#ifdef CHANNEL
    Point<dim> coordinates;
    coordinates[0] = 2.0*numbers::PI;
    coordinates[1] = 1.0;
    if (dim == 3)
      coordinates[2] = numbers::PI;
    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
//    const double left = -1.0, right = 1.0;
//    GridGenerator::hyper_cube(triangulation,left,right);
//    const unsigned int base_refinements = n_refine_space;
    std::vector<unsigned int> refinements(dim, 1);
    //refinements[0] *= 3;
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);

//    std::vector<unsigned int> repetitions({2,1,1});
//    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,Point<dim>(),coordinates);

    // set boundary indicator
//    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
//    for(;cell!=endc;++cell)
//    {
//      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
//      {
//        if ((std::fabs(cell->face(face_number)->center()(0) - 0.)< 1e-12))
//          cell->face(face_number)->set_boundary_id(10);
//        if ((std::fabs(cell->face(face_number)->center()(0) - 2.*numbers::PI)< 1e-12))
//          cell->face(face_number)->set_boundary_id(11);
//        if ((std::fabs(cell->face(face_number)->center()(2) - 0.)< 1e-12))
//          cell->face(face_number)->set_boundary_id(12);
//        if ((std::fabs(cell->face(face_number)->center()(2) - numbers::PI)< 1e-12))
//          cell->face(face_number)->set_boundary_id(13);
//      // if ((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
//      //    cell->face(face_number)->set_boundary_id (1);
//      }
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
    triangulation.refine_global(n_refine_space);

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
void NavierStokesProblem<dim>::print_parameters() const
{
  pcout << std::endl << "further parameters:" << std::endl;
  pcout << " - number of quad points for xwall:     " << N_Q_POINTS_1D_XWALL << std::endl;
  pcout << " - viscosity:                           " << VISCOSITY << std::endl;
  pcout << " - IP_factor_pressure:                  " << param.IP_factor_pressure << std::endl;
  pcout << " - IP_factor_viscous:                   " << param.IP_factor_viscous << std::endl;
  pcout << " - penalty factor divergence:           " << param.penalty_factor_divergence << std::endl;
  pcout << " - penalty factor continuity:           " << param.penalty_factor_continuity << std::endl;
  pcout << " - Smagorinsky constant                 " << CS << std::endl;
  pcout << " - fix tauw to 1.0:                     " << not VARIABLETAUW << std::endl;
  pcout << " - max wall distance of xwall:          " << MAX_WDIST_XWALL << std::endl;
  pcout << " - grid stretching if no xwall:         " << GRID_STRETCH_FAC << std::endl;
  pcout << " - prefix:                              " << param.output_prefix << std::endl;
}

template<int dim>
void NavierStokesProblem<dim>::solve_problem()
{
  create_grid();

  navier_stokes_operation->setup(periodic_faces, dirichlet_boundary, neumann_boundary);

  if(param.problem_type == ProblemType::Unsteady)
  {
    // setup time integrator before calling setup_solvers
    // (this is necessary since the setup of the solvers
    // depends on quantities such as the time_step_size or gamma0!!!)
    time_integrator->setup();

    navier_stokes_operation->setup_solvers();

    print_parameters();

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

    //mesh refinements in order to perform spatial convergence tests
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;refine_steps_space <= REFINE_STEPS_SPACE_MAX;++refine_steps_space)
    {
      //time refinements in order to perform temporal convergence tests
      for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;refine_steps_time <= REFINE_STEPS_TIME_MAX;++refine_steps_time)
      {
        NavierStokesProblem<DIMENSION> navier_stokes_problem(refine_steps_space,refine_steps_time);
        navier_stokes_problem.solve_problem();
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
