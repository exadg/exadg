/*
 * DGNavierStokesBase.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_BASE_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_BASE_H_

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include "../../incompressible_navier_stokes/infrastructure/fe_evaluation_wrapper.h"
#include "../../incompressible_navier_stokes/spatial_discretization/navier_stokes_calculators.h"
#include "../../incompressible_navier_stokes/spatial_discretization/navier_stokes_operators.h"
#include "../../incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../../incompressible_navier_stokes/user_interface/field_functions.h"
#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"
#include "../../poisson/boundary_descriptor_laplace.h"
#include "../infrastructure/fe_parameters.h"
#include "operators/matrix_operator_base.h"
#include "operators/inverse_mass_matrix.h"
#include "turbulence_model.h"

#include "solvers_and_preconditioners/iterative_solvers.h"
#include "solvers_and_preconditioners/inverse_mass_matrix_preconditioner.h"



using namespace dealii;

//forward declarations
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule>
class DGNavierStokesDualSplittingXWall;

template<int dim> struct ViscousOperatorData;
template<int dim> struct ConvectiveOperatorData;
template<int dim> struct GradientOperatorData;
template<int dim> struct DivergenceOperatorData;
template<int dim> struct BodyForceOperatorData;

template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class MassMatrixOperator;
template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class ConvectiveOperator;
template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class ViscousOperator;
template<int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class BodyForceOperator;
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class GradientOperator;
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class DivergenceOperator;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number=double>
class DGNavierStokesBase : public MatrixOperatorBase
{
public:
  enum class DofHandlerSelector {
    velocity = 0,
    pressure = 1,
    velocity_scalar = 2,
    n_variants = velocity_scalar+1
  };

  enum class QuadratureSelector {
    velocity = 0,
    pressure = 1,
    velocity_nonlinear = 2,
    n_variants = velocity_nonlinear+1
  };

  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;

  static const unsigned int dof_index_u = static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::velocity);
  static const unsigned int dof_index_p = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  static const unsigned int dof_index_u_scalar = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity_scalar);

  static const unsigned int quad_index_u = static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity);
  static const unsigned int quad_index_p = static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure);
  static const unsigned int quad_index_u_nonlinear = static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::velocity_nonlinear);

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,Number,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  // Constructor
  DGNavierStokesBase(parallel::distributed::Triangulation<dim> const &triangulation,
                     InputParametersNavierStokes<dim> const          &parameter)
    :
    fe_u(new FESystem<dim>(FE_DGQ<dim>(fe_degree),dim)),
    fe_p(fe_degree_p),
    fe_u_scalar(fe_degree),
    mapping(fe_degree), //(fe_degree <= 10 ? fe_degree : 10), //TODO
    dof_handler_u(triangulation),
    dof_handler_p(triangulation),
    dof_handler_u_scalar(triangulation),
    dof_index_first_point(0),
    param(parameter),
    inverse_mass_matrix_operator(nullptr)
  {}

  // Destructor
  virtual ~DGNavierStokesBase()
  {
    data.clear();
  }

  void initialize_boundary_descriptor_laplace();

  virtual void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                                             periodic_face_pairs,
                      std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity,
                      std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure,
                      std::shared_ptr<FieldFunctionsNavierStokes<dim> >      field_functions);

  void apply_mass_matrix(parallel::distributed::Vector<Number>       &dst,
                         parallel::distributed::Vector<Number> const &src) const;

  void apply_mass_matrix_add(parallel::distributed::Vector<Number>       &dst,
                             parallel::distributed::Vector<Number> const &src) const;

  virtual void prescribe_initial_conditions(parallel::distributed::Vector<Number> &velocity,
                                            parallel::distributed::Vector<Number> &pressure,
                                            double const                          evaluation_time) const;


  MatrixFree<dim,Number> const & get_data() const
  {
    return data;
  }

  unsigned int get_dof_index_velocity() const
  {
    return dof_index_u;
  }

  unsigned int get_dof_index_velocity_scalar() const
  {
    return dof_index_u_scalar;
  }

  unsigned int get_quad_index_velocity_linear() const
  {
    return quad_index_u;
  }

  unsigned int get_dof_index_pressure() const
  {
    return dof_index_p;
  }

  unsigned int get_quad_index_pressure() const
  {
    return quad_index_p;
  }

  Mapping<dim> const & get_mapping() const
  {
    return mapping;
  }

  FESystem<dim> const & get_fe_u() const
  {
    return *fe_u;
  }

  FE_DGQ<dim> const & get_fe_p() const
  {
    return fe_p;
  }

  DoFHandler<dim> const & get_dof_handler_u() const
  {
    return dof_handler_u;
  }

  DoFHandler<dim> const & get_dof_handler_u_scalar() const
  {
    return dof_handler_u_scalar;
  }

  DoFHandler<dim> const & get_dof_handler_p() const
  {
    return dof_handler_p;
  }

  double get_viscosity() const
  {
    return viscous_operator.get_const_viscosity();
  }

  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator_data;
  }

  ViscousOperatorData<dim> const & get_viscous_operator_data() const
  {
    return viscous_operator_data;
  }

  ConvectiveOperatorData<dim> const & get_convective_operator_data() const
  {
    return convective_operator_data;
  }

  GradientOperatorData<dim> const & get_gradient_operator_data() const
  {
    return gradient_operator_data;
  }

  DivergenceOperatorData<dim> const & get_divergence_operator_data() const
  {
    return divergence_operator_data;
  }

  std::shared_ptr<FieldFunctionsNavierStokes<dim> > const get_field_functions() const
  {
    return field_functions;
  }

  // initialization of vectors
  void initialize_vector_velocity(parallel::distributed::Vector<Number> &src) const
  {
    this->data.initialize_dof_vector(src,dof_index_u);
  }

  void initialize_vector_velocity_scalar(parallel::distributed::Vector<Number> &src) const
  {
    this->data.initialize_dof_vector(src,dof_index_u_scalar);
  }

  void initialize_vector_vorticity(parallel::distributed::Vector<Number> &src) const
  {
    this->data.initialize_dof_vector(src,dof_index_u);
  }

  void initialize_vector_pressure(parallel::distributed::Vector<Number> &src) const
  {
    this->data.initialize_dof_vector(src,dof_index_p);
  }

  // special case: pure Dirichlet boundary conditions
  // if analytical solution is available: shift pressure so that the numerical pressure solution
  // coincides the the analytical pressure solution in an arbitrary point.
  // Note that the parameter 'eval_time' is only needed for unsteady problems.
  void  shift_pressure (parallel::distributed::Vector<Number> &pressure,
                        double const                              &eval_time = 0.0) const;

  // special case: pure Dirichlet boundary conditions
  // if analytical solution is available: shift pressure so that the numerical pressure solution
  // has a mean value identical to the "exact pressure solution" obtained by interpolation of analytical solution.
  // Note that the parameter 'eval_time' is only needed for unsteady problems.
  void  shift_pressure_mean_value (parallel::distributed::Vector<Number> &pressure,
                                   double const                              &eval_time = 0.0) const;

  // special case: pure Dirichlet boundary conditions
  // if no analytical solution is available: set mean value of pressure vector to zero
  void apply_zero_mean (parallel::distributed::Vector<Number>  &dst) const;

  // vorticity
  void compute_vorticity (parallel::distributed::Vector<Number>       &dst,
                          const parallel::distributed::Vector<Number> &src) const;

  // divergence
  void compute_divergence (parallel::distributed::Vector<Number>       &dst,
                           const parallel::distributed::Vector<Number> &src) const;

  // velocity_magnitude
  void compute_velocity_magnitude (parallel::distributed::Vector<Number>       &dst,
                                   const parallel::distributed::Vector<Number> &src) const;

  // streamfunction
  void compute_streamfunction (parallel::distributed::Vector<Number>       &dst,
                               const parallel::distributed::Vector<Number> &src) const;

  // Q criterion
  void compute_q_criterion (parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const;

  void evaluate_convective_term (parallel::distributed::Vector<Number>       &dst,
                                 parallel::distributed::Vector<Number> const &src,
                                 Number const                                evaluation_time) const;

  // TODO OIF splitting
//  void evaluate_negative_convective_term_and_apply_inverse_mass_matrix (
//                                 parallel::distributed::Vector<Number>       &dst,
//                                 parallel::distributed::Vector<Number> const &src,
//                                 Number const                                evaluation_time) const;

  void evaluate_negative_convective_term_and_apply_inverse_mass_matrix (
                            parallel::distributed::Vector<Number>       &dst,
                            parallel::distributed::Vector<Number> const &src,
                            Number const                                evaluation_time,
                            parallel::distributed::Vector<Number> const &velocity) const;

  // inverse velocity mass matrix
  void apply_inverse_mass_matrix (parallel::distributed::Vector<Number>       &dst,
                                  parallel::distributed::Vector<Number> const &src) const;

  /*
   *  Update turbulence model, i.e., calculate turbulent viscosity
   */
  void update_turbulence_model (parallel::distributed::Vector<Number> const &velocity);

  parallel::distributed::Vector<Number> & get_viscosity_dof_vector()
  {
    return viscous_operator.get_viscosity_dof_vector();
  }

private:
  virtual void create_dofs();

  virtual void data_reinit(typename MatrixFree<dim,Number>::AdditionalData & additional_data);

protected:
  MatrixFree<dim,Number> data;

  std::shared_ptr<FESystem<dim> > fe_u;
  FE_DGQ<dim> fe_p;
  FE_DGQ<dim> fe_u_scalar;

  MappingQGeneric<dim> mapping;

  DoFHandler<dim> dof_handler_u;
  DoFHandler<dim> dof_handler_p;
  DoFHandler<dim> dof_handler_u_scalar;

  Point<dim> first_point;
  types::global_dof_index dof_index_first_point;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs;

  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure;
  std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;

  // In case of projection type Navier-Stokes solvers this variable
  // is needed to solve the pressure Poisson equation.
  // In that case, also the functions specified in BoundaryDescriptorLaplace are relevant.
  // In case of the coupled solver it is used for the block preconditioner
  // (or more precisely for the Schur-complement preconditioner and the GMG method
  // used to approximately invert the Laplace operator).
  // In that case, the functions specified in BoundaryDescriptorLaplace are irrelevant.
  std::shared_ptr<BoundaryDescriptorLaplace<dim> > boundary_descriptor_laplace;

  InputParametersNavierStokes<dim> const &param;

  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;
  GradientOperatorData<dim> gradient_operator_data;
  DivergenceOperatorData<dim> divergence_operator_data;

  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> mass_matrix_operator;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> convective_operator;
  std::shared_ptr<InverseMassMatrixOperator<dim,fe_degree,Number,dim> > inverse_mass_matrix_operator;
  std::shared_ptr<InverseMassMatrixOperator<dim,fe_degree,Number,1> > inverse_velocity_mass_matrix_operator_scalar;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> viscous_operator;
  BodyForceOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> body_force_operator;
  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> gradient_operator;
  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> divergence_operator;

  VorticityCalculator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> vorticity_calculator;
  DivergenceCalculator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> divergence_calculator;
  VelocityMagnitudeCalculator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> velocity_magnitude_calculator;
  QCriterionCalculator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> q_criterion_calculator;

private:
  // turbulence modeling LES
  TurbulenceModel<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> turbulence_model;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
initialize_boundary_descriptor_laplace()
{
  boundary_descriptor_laplace.reset(new BoundaryDescriptorLaplace<dim>());

  // Dirichlet BCs for pressure
  this->boundary_descriptor_laplace->dirichlet = boundary_descriptor_pressure->dirichlet_bc;

  // Neumann BCs for pressure
  // Note: for the dual splitting scheme, neumann_bc contains functions corresponding
  //       to dudt term required in pressure NBC.
  // Here: set this functions explicitly to ZeroFunction when filling the boundary
  //       descriptor for the Laplace operator because these inhomogeneous
  //       boundary conditions have to be implemented seperately
  //       and can not be applied by the Laplace operator.
  for (typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::
       const_iterator it = boundary_descriptor_pressure->neumann_bc.begin();
       it != boundary_descriptor_pressure->neumann_bc.end(); ++it)
  {
    std::shared_ptr<Function<dim> > zero_function;
    zero_function.reset(new ZeroFunction<dim>(1));
    boundary_descriptor_laplace->neumann.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
      (it->first,zero_function));
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                              periodic_face_pairs,
       std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity_in,
       std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure_in,
       std::shared_ptr<FieldFunctionsNavierStokes<dim> >      field_functions_in)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup Navier-Stokes operation ..." << std::endl << std::flush;

  this->periodic_face_pairs = periodic_face_pairs;
  this->boundary_descriptor_velocity = boundary_descriptor_velocity_in;
  this->boundary_descriptor_pressure = boundary_descriptor_pressure_in;
  this->field_functions = field_functions_in;

  initialize_boundary_descriptor_laplace();

  create_dofs();

  // initialize matrix_free_data
  typename MatrixFree<dim,Number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::partition_partition;
  additional_data.build_face_info = true;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points | update_normal_vectors |
                                          update_values);

  additional_data.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values |
                                                      update_quadrature_points | update_normal_vectors |
                                                      update_values);

  additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values |
                                                         update_quadrature_points | update_normal_vectors |
                                                         update_values);

  data_reinit(additional_data);

  // mass matrix operator
  mass_matrix_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  mass_matrix_operator.initialize(data,mass_matrix_operator_data);

  // inverse mass matrix operator
  inverse_mass_matrix_operator.reset(new InverseMassMatrixOperator<dim,fe_degree,Number,dim>());
  inverse_mass_matrix_operator->initialize(data,dof_index_u,quad_index_u);

  // inverse mass matrix operator velocity scalar
  inverse_velocity_mass_matrix_operator_scalar.reset(new InverseMassMatrixOperator<dim,fe_degree,Number,1>());
  inverse_velocity_mass_matrix_operator_scalar->initialize(data,dof_index_u_scalar,quad_index_u);

  // body force operator
  BodyForceOperatorData<dim> body_force_operator_data;
  body_force_operator_data.dof_index = dof_index_u;
  body_force_operator_data.rhs = field_functions->right_hand_side;
  body_force_operator.initialize(data,body_force_operator_data);

  // gradient operator
  gradient_operator_data.dof_index_velocity = dof_index_u;
  gradient_operator_data.dof_index_pressure = dof_index_p;
  gradient_operator_data.integration_by_parts_of_gradP = param.gradp_integrated_by_parts;
  gradient_operator_data.use_boundary_data = param.gradp_use_boundary_data;
  gradient_operator_data.bc = boundary_descriptor_pressure;
  gradient_operator.initialize(data,gradient_operator_data);

  // divergence operator
  divergence_operator_data.dof_index_velocity = dof_index_u;
  divergence_operator_data.dof_index_pressure = dof_index_p;
  divergence_operator_data.integration_by_parts_of_divU = param.divu_integrated_by_parts;
  divergence_operator_data.use_boundary_data = param.divu_use_boundary_data;
  divergence_operator_data.bc = boundary_descriptor_velocity;
  divergence_operator.initialize(data,divergence_operator_data);

  // convective operator
  convective_operator_data.dof_index = dof_index_u;
  convective_operator_data.bc = boundary_descriptor_velocity;
  convective_operator_data.use_outflow_bc = param.use_outflow_bc_convective_term;
  convective_operator.initialize(data,convective_operator_data);

  // viscous operator
  viscous_operator_data.formulation_viscous_term = param.formulation_viscous_term;
  viscous_operator_data.penalty_term_div_formulation = param.penalty_term_div_formulation;
  viscous_operator_data.IP_formulation_viscous = param.IP_formulation_viscous;
  viscous_operator_data.IP_factor_viscous = param.IP_factor_viscous;
  viscous_operator_data.bc = boundary_descriptor_velocity;
  viscous_operator_data.dof_index = dof_index_u;
  viscous_operator_data.viscosity = param.viscosity;
  viscous_operator.initialize(mapping,data,viscous_operator_data);

  // turbulence model
  if(this->param.use_turbulence_model == true)
  {
    // make sure that viscous coefficients are initialized
    viscous_operator.initialize_viscous_coefficients();

    // initialize turbulence model
    TurbulenceModelData model_data;
    model_data.turbulence_model = this->param.turbulence_model;
    model_data.constant = this->param.turbulence_model_constant;
    turbulence_model.initialize(data,mapping,viscous_operator,model_data);
  }

  // vorticity
  vorticity_calculator.initialize(data,dof_index_u);
  // divergence
  divergence_calculator.initialize(data,dof_index_u,dof_index_u_scalar);
  // velocity magnitude
  velocity_magnitude_calculator.initialize(data,dof_index_u,dof_index_u_scalar);
  // q criterion
  q_criterion_calculator.initialize(data,dof_index_u,dof_index_u_scalar);

  if(this->param.pure_dirichlet_bc == true &&
     this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
  {
    dof_index_first_point = 0;
    for(unsigned int d=0;d<dim;++d)
      first_point[d] = 0.0;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      typename DoFHandler<dim>::active_cell_iterator first_cell;
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(), endc = dof_handler_p.end();
      bool processor_has_active_cells = false;
      for(;cell!=endc;++cell)
      {
        if (cell->is_locally_owned())
        {
          first_cell = cell;
          processor_has_active_cells = true;
          break;
        }
      }
      AssertThrow(processor_has_active_cells == true,ExcMessage("No active cells on Processor with ID=0"));

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
    {
      first_point[d] = Utilities::MPI::sum(first_point[d],MPI_COMM_WORLD);
    }
  }

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
create_dofs()
{
  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_u.distribute_mg_dofs(*fe_u);
  dof_handler_p.distribute_dofs(fe_p);
  dof_handler_p.distribute_mg_dofs(fe_p);
  dof_handler_u_scalar.distribute_dofs(fe_u_scalar);
  dof_handler_u_scalar.distribute_mg_dofs(fe_u_scalar); // probably, we don't need this

  unsigned int ndofs_per_cell_velocity = Utilities::fixed_int_power<fe_degree+1,dim>::value*dim;
  unsigned int ndofs_per_cell_pressure = Utilities::fixed_int_power<fe_degree_p+1,dim>::value;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:"
        << std::endl << std::endl << std::flush;

  pcout << "Velocity:" << std::endl;
  print_parameter(pcout,"degree of 1D polynomials",fe_degree);
  print_parameter(pcout,"number of dofs per cell",ndofs_per_cell_velocity);
  print_parameter(pcout,"number of dofs (total)",dof_handler_u.n_dofs());

  pcout << "Pressure:" << std::endl;
  print_parameter(pcout,"degree of 1D polynomials",fe_degree_p);
  print_parameter(pcout,"number of dofs per cell",ndofs_per_cell_pressure);
  print_parameter(pcout,"number of dofs (total)",dof_handler_p.n_dofs());

  pcout << std::flush;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
data_reinit(typename MatrixFree<dim,Number>::AdditionalData &additional_data)
{
  std::vector<const DoFHandler<dim> * >  dof_handler_vec;

  dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::n_variants));
  dof_handler_vec[dof_index_u] = &dof_handler_u;
  dof_handler_vec[dof_index_p] = &dof_handler_p;
  dof_handler_vec[dof_index_u_scalar] = &dof_handler_u_scalar;

  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  constraint_matrix_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(DofHandlerSelector::n_variants));
  ConstraintMatrix constraint_u, constraint_p, constraint_u_scalar;
  constraint_u.close();
  constraint_p.close();
  constraint_u_scalar.close();
  constraint_matrix_vec[dof_index_u] = &constraint_u;
  constraint_matrix_vec[dof_index_p] = &constraint_p;
  constraint_matrix_vec[dof_index_u_scalar] = &constraint_u_scalar;

  std::vector<Quadrature<1> > quadratures;

  // resize quadratures
  quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type>(QuadratureSelector::n_variants));
  // velocity
  quadratures[quad_index_u] = QGauss<1>(fe_degree+1);
  // pressure
  quadratures[quad_index_p] = QGauss<1>(fe_degree_p+1);
  // exact integration of nonlinear convective term
  quadratures[quad_index_u_nonlinear] = QGauss<1>(fe_degree + (fe_degree+2)/2);

  data.reinit (mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
prescribe_initial_conditions(parallel::distributed::Vector<Number> &velocity,
                             parallel::distributed::Vector<Number> &pressure,
                             double const                          evaluation_time) const
{
  this->field_functions->initial_solution_velocity->set_time(evaluation_time);
  this->field_functions->initial_solution_pressure->set_time(evaluation_time);

  // This is necessary if Number == float
  parallel::distributed::Vector<double> velocity_double;
  parallel::distributed::Vector<double> pressure_double;
  velocity_double = velocity;
  pressure_double = pressure;

  VectorTools::interpolate(mapping, dof_handler_u, *(this->field_functions->initial_solution_velocity), velocity_double);
  VectorTools::interpolate(mapping, dof_handler_p, *(this->field_functions->initial_solution_pressure), pressure_double);

  velocity = velocity_double;
  pressure = pressure_double;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
apply_mass_matrix (parallel::distributed::Vector<Number>       &dst,
                   parallel::distributed::Vector<Number> const &src) const
{
  this->mass_matrix_operator.apply(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
apply_mass_matrix_add (parallel::distributed::Vector<Number>       &dst,
                       parallel::distributed::Vector<Number> const &src) const
{
  this->mass_matrix_operator.apply_add(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
shift_pressure (parallel::distributed::Vector<Number>  &pressure,
                double const                           &eval_time) const
{
  AssertThrow(this->param.error_data.analytical_solution_available == true,
              ExcMessage("The function shift_pressure is intended to be used only if an analytical solution is available!"));

  parallel::distributed::Vector<Number> vec1(pressure);
  for(unsigned int i=0;i<vec1.local_size();++i)
    vec1.local_element(i) = 1.;
  this->field_functions->analytical_solution_pressure->set_time(eval_time);
  double const exact = this->field_functions->analytical_solution_pressure->value(first_point);
  double current = 0.;
  if (pressure.locally_owned_elements().is_element(dof_index_first_point))
    current = pressure(dof_index_first_point);
  current = Utilities::MPI::sum(current, MPI_COMM_WORLD);
  pressure.add(exact-current,vec1);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
shift_pressure_mean_value (parallel::distributed::Vector<Number>  &pressure,
                           double const                           &eval_time) const
{
  AssertThrow(this->param.error_data.analytical_solution_available == true,
              ExcMessage("The function shift_pressure_mean_value is intended to be used only if an analytical solution is available!"));

  // one cannot use Number as template here since Number might be float
  // while analytical_solution_pressure is of type Function<dim,double>
  parallel::distributed::Vector<double> vec_double;
  vec_double = pressure; //initialize

  this->field_functions->analytical_solution_pressure->set_time(eval_time);
  VectorTools::interpolate(mapping, dof_handler_p, *(this->field_functions->analytical_solution_pressure), vec_double);

  double const exact = vec_double.mean_value();
  double const current = pressure.mean_value();

  parallel::distributed::Vector<Number> vec_temp2(pressure);
  for(unsigned int i=0;i<vec_temp2.local_size();++i)
    vec_temp2.local_element(i) = 1.;

  pressure.add(exact-current,vec_temp2);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
apply_zero_mean (parallel::distributed::Vector<Number>  &vector) const
{
  const Number mean_value = vector.mean_value();
  vector.add(-mean_value);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
compute_vorticity (parallel::distributed::Vector<Number>       &dst,
                   const parallel::distributed::Vector<Number> &src) const
{
  vorticity_calculator.compute_vorticity(dst,src);

  inverse_mass_matrix_operator->apply(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
compute_divergence (parallel::distributed::Vector<Number>       &dst,
                    const parallel::distributed::Vector<Number> &src) const
{
  divergence_calculator.compute_divergence(dst,src);

  inverse_velocity_mass_matrix_operator_scalar->apply(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
compute_velocity_magnitude (parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
{
  velocity_magnitude_calculator.compute(dst,src);

  inverse_velocity_mass_matrix_operator_scalar->apply(dst,dst);
}

/*
 *  Streamfunction psi (2D only): defined as u1 = d(psi)/dx2, u2 = - d(psi)/dx1
 *
 *  Vorticity: omega = du2/dx1 - du1/dx2
 *
 *  --> laplace(psi) = (d²/dx1²+d²/dx2²)(psi)
 *                   = d(d(psi)/dx1)/dx1 + d(d(psi)/dx2)/dx2
 *                   = d(-u2)/dx1 + d(u1)/dx2 = - omega
 *
 *  or
 *      - laplace(psi) = omega
 *
 *  with homogeneous Dirichlet BC's (assumption: boundary == streamline)
 */
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
compute_streamfunction (parallel::distributed::Vector<Number>       &dst,
                        parallel::distributed::Vector<Number> const &src) const
{
  AssertThrow(dim==2, ExcMessage("Calculation of streamfunction can only be used for dim==2."));

  // compute rhs vector
  StreamfunctionCalculatorRHSOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> rhs_operator;
  rhs_operator.initialize(data,dof_index_u,dof_index_u_scalar);
  parallel::distributed::Vector<Number> rhs;
  this->initialize_vector_velocity_scalar(rhs);
  rhs_operator.apply(rhs,src);

  // setup Laplace operator for scalar velocity vector
  LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.laplace_dof_index = this->get_dof_index_velocity_scalar();
  laplace_operator_data.laplace_quad_index = this->get_quad_index_velocity_linear();
  std::shared_ptr<BoundaryDescriptorLaplace<dim> > boundary_descriptor_streamfunction;
  boundary_descriptor_streamfunction.reset(new BoundaryDescriptorLaplace<dim>());

  // fill boundary descriptor: Assumption: only Dirichlet BC's
  boundary_descriptor_streamfunction->dirichlet = boundary_descriptor_velocity->dirichlet_bc;

  AssertThrow(boundary_descriptor_velocity->neumann_bc.empty() == true,
      ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                 "not implemented for this type of boundary conditions."));
  AssertThrow(boundary_descriptor_velocity->symmetry_bc.empty() == true,
      ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                 "not implemented for this type of boundary conditions."));

  laplace_operator_data.bc = boundary_descriptor_streamfunction;
  laplace_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;

  LaplaceOperator<dim,fe_degree, Number> laplace_operator;
  laplace_operator.reinit(this->data,this->mapping,laplace_operator_data);

  // setup preconditioner
  std::shared_ptr<PreconditionerBase<Number> > preconditioner;

  // use multigrid preconditioner with Chebyshev smoother
  MultigridData mg_data;

  // use single precision for multigrid
  typedef float MultigridNumber;
  typedef MyMultigridPreconditionerLaplace<dim, Number,
      LaplaceOperator<dim, fe_degree, MultigridNumber>,
      LaplaceOperatorData<dim> > MULTIGRID;

  preconditioner.reset(new MULTIGRID());

  std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner);

  mg_preconditioner->initialize(mg_data,
                                this->dof_handler_u_scalar,
                                this->mapping,
                                laplace_operator_data,
                                laplace_operator_data.bc->dirichlet);

  // setup solver
  CGSolverData solver_data;
  solver_data.solver_tolerance_rel = 1.e-10;
  solver_data.use_preconditioner = true;

  CGSolver<LaplaceOperator<dim, fe_degree, Number>,
           PreconditionerBase<Number>,
           parallel::distributed::Vector<Number> >
    poisson_solver(laplace_operator,*preconditioner,solver_data);

  // solve Poisson problem
  poisson_solver.solve(dst,rhs);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
compute_q_criterion (parallel::distributed::Vector<Number>       &dst,
                     const parallel::distributed::Vector<Number> &src) const
{
  q_criterion_calculator.compute(dst,src);

  inverse_velocity_mass_matrix_operator_scalar->apply(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
apply_inverse_mass_matrix (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
{
  inverse_mass_matrix_operator->apply(dst,src);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
evaluate_convective_term (parallel::distributed::Vector<Number>       &dst,
                          parallel::distributed::Vector<Number> const &src,
                          Number const                                evaluation_time) const
{
  convective_operator.evaluate(dst,src,evaluation_time);
}

//TODO OIF splitting
//template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
//void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
//evaluate_negative_convective_term_and_apply_inverse_mass_matrix (
//                          parallel::distributed::Vector<Number>       &dst,
//                          parallel::distributed::Vector<Number> const &src,
//                          Number const                                evaluation_time) const
//{
//  convective_operator.evaluate(dst,src,evaluation_time);
//
//  // shift convective term to the rhs of the equation
//  dst *= -1.0;
//
//  inverse_mass_matrix_operator->apply(dst,dst);
//}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
evaluate_negative_convective_term_and_apply_inverse_mass_matrix (
                          parallel::distributed::Vector<Number>       &dst,
                          parallel::distributed::Vector<Number> const &src,
                          Number const                                evaluation_time,
                          parallel::distributed::Vector<Number> const &velocity) const
{
  convective_operator.evaluate_oif(dst,src,evaluation_time,velocity);

  // shift convective term to the rhs of the equation
  dst *= -1.0;

  inverse_mass_matrix_operator->apply(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
update_turbulence_model (parallel::distributed::Vector<Number> const &velocity)
{
  // calculate turbulent viscosity locally in each cell and face quadrature point
  turbulence_model.calculate_turbulent_viscosity(velocity);
}

template<typename Operator, typename value_type>
class ConvectiveOperatorNavierStokes
{
public:
  ConvectiveOperatorNavierStokes(std::shared_ptr<Operator> operation_in)
    :
    underlying_operator(operation_in)
  {}

  // TODO OIF splitting
//  void evaluate(parallel::distributed::Vector<value_type>       &dst,
//                parallel::distributed::Vector<value_type> const &src,
//                value_type const                                evaluation_time) const
//  {
//    underlying_operator->evaluate_negative_convective_term_and_apply_inverse_mass_matrix(dst,src,evaluation_time);
//  }

  void evaluate(parallel::distributed::Vector<value_type>       &dst,
                parallel::distributed::Vector<value_type> const &src,
                value_type const                                evaluation_time,
                parallel::distributed::Vector<value_type> const &velocity) const
  {
    underlying_operator->evaluate_negative_convective_term_and_apply_inverse_mass_matrix(dst,src,evaluation_time,velocity);
  }

  void initialize_dof_vector(parallel::distributed::Vector<value_type> &src) const
  {
    underlying_operator->initialize_vector_velocity(src);
  }

private:
  std::shared_ptr<Operator> underlying_operator;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_BASE_H_ */
