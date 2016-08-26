/*
 * DGNavierStokesBase.h
 *
 *  Created on: Jun 27, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESBASE_H_
#define INCLUDE_DGNAVIERSTOKESBASE_H_

#include <deal.II/matrix_free/operators.h>

#include "FEEvaluationWrapper.h"
#include "FE_Parameters.h"

#include "InverseMassMatrix.h"
#include "NavierStokesOperators.h"

#include "../include/BoundaryDescriptorNavierStokes.h"
#include "../include/FieldFunctionsNavierStokes.h"
#include "InputParametersNavierStokes.h"

using namespace dealii;

//forward declarations
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesDualSplittingXWall;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesBase
{
public:
  enum class DofHandlerSelector {
    velocity = 0,
    pressure = 1,
    n_variants = pressure+1
  };

  enum class QuadratureSelector {
    velocity = 0,
    pressure = 1,
    velocity_nonlinear = 2,
    n_variants = velocity_nonlinear+1
  };

  typedef double value_type;
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
  static const bool is_xwall = (n_q_points_1d_xwall>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+1;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  // constructor
  DGNavierStokesBase(parallel::distributed::Triangulation<dim> const &triangulation,
                     InputParametersNavierStokes const               &parameter)
    :
    // fe_u(FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree+1)),dim),
    fe_u(new FESystem<dim>(FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree+1)),dim)),
    fe_p(QGaussLobatto<1>(fe_degree_p+1)),
    mapping(fe_degree),
    dof_handler_u(triangulation),
    dof_handler_p(triangulation),
    evaluation_time(0.0),
    time_step(1.0),
    scaling_factor_time_derivative_term(1.0),
    viscosity(parameter.viscosity),
    dof_index_first_point(0),
    param(parameter),
    fe_param(param),
    inverse_mass_matrix_operator(nullptr)
  {}

  // destructor
  virtual ~DGNavierStokesBase()
  {
    data.clear();
  }

  void fill_dbc_and_nbc_sets(std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor);

  virtual void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                                                  periodic_face_pairs,
                      std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
                      std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
                      std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> >     field_functions);

  virtual void setup_solvers () = 0;

  virtual void prescribe_initial_conditions(parallel::distributed::Vector<value_type> &velocity,
                                            parallel::distributed::Vector<value_type> &pressure,
                                            double const                              evaluation_time) const;

  // getters
  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }

  unsigned int get_dof_index_velocity() const
  {
    return static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  }

  unsigned int get_quad_index_velocity_linear() const
  {
    return static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity);
  }

  unsigned int get_dof_index_pressure() const
  {
    return static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  }

  unsigned int get_quad_index_pressure() const
  {
    return static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure);
  }

  Mapping<dim> const & get_mapping() const
  {
    return mapping;
  }

  FESystem<dim> const & get_fe_u() const
  {
    return *fe_u;
  }

  FE_DGQArbitraryNodes<dim> const & get_fe_p() const
  {
    return fe_p;
  }

  DoFHandler<dim> const & get_dof_handler_u() const
  {
    return dof_handler_u;
  }

  DoFHandler<dim> const & get_dof_handler_p() const
  {
    return dof_handler_p;
  }

  double get_viscosity() const
  {
    return viscosity;
  }

  FEParameters<dim> const & get_fe_parameters() const
  {
    return fe_param;
  }

  value_type get_scaling_factor_time_derivative_term() const
  {
    return scaling_factor_time_derivative_term;
  }

  std::set<types::boundary_id> get_dirichlet_boundary() const
  {
    return dirichlet_boundary;
  }

  std::set<types::boundary_id> get_neumann_boundary() const
  {
    return neumann_boundary;
  }

//  const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > get_periodic_face_pairs() const
//  {
//    return periodic_face_pairs;
//  }

  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator_data;
  }

  ViscousOperatorData<dim> const & get_viscous_operator_data() const
  {
    return viscous_operator_data;
  }

  GradientOperatorData<dim> const & get_gradient_operator_data() const
  {
    return gradient_operator_data;
  }

  DivergenceOperatorData<dim> const & get_divergence_operator_data() const
  {
    return divergence_operator_data;
  }

  // setters
  void set_scaling_factor_time_derivative_term(double const value)
  {
    scaling_factor_time_derivative_term = value;
  }

  void set_evaluation_time(double const eval_time)
  {
    evaluation_time = eval_time;
  }

  void set_time_step(double const time_step_in)
  {
    time_step = time_step_in;
  }

  // initialization of vectors
  void initialize_vector_velocity(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
  }

  void initialize_vector_vorticity(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
  }

  void initialize_vector_pressure(parallel::distributed::Vector<value_type> &src) const
  {
    this->data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure));
  }

  // special case: pure Dirichlet boundary conditions
  // if analytical solution is available: shift pressure so that the numerical pressure solution
  // coincides the the analytical pressure solution in an arbitrary point
  void  shift_pressure (parallel::distributed::Vector<value_type> &pressure) const;

  // special case: pure Dirichlet boundary conditions
  // if no analytical solution is available: set mean value of pressure vector to zero
  void apply_zero_mean (parallel::distributed::Vector<value_type>  &dst) const;

  // vorticity
  void compute_vorticity (parallel::distributed::Vector<value_type>       &dst,
                          const parallel::distributed::Vector<value_type> &src) const;

  // divergence
  void compute_divergence (parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src) const;

  void evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                                 parallel::distributed::Vector<value_type> const &src,
                                 value_type const                                evaluation_time) const;

protected:
  MatrixFree<dim,value_type> data;

  std_cxx11::shared_ptr< FESystem<dim> > fe_u;
  FE_DGQArbitraryNodes<dim> fe_p;

  MappingQGeneric<dim> mapping;

  DoFHandler<dim>  dof_handler_u;
  DoFHandler<dim>  dof_handler_p;

  double evaluation_time;
  double time_step;
  double scaling_factor_time_derivative_term;

  const double viscosity;

  Point<dim> first_point;
  types::global_dof_index dof_index_first_point;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs;

  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity;
  std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure;
  std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions;

  std::set<types::boundary_id> dirichlet_boundary;
  std::set<types::boundary_id> neumann_boundary;

  InputParametersNavierStokes const &param;

  FEParameters<dim> fe_param;

  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;
  GradientOperatorData<dim> gradient_operator_data;
  DivergenceOperatorData<dim> divergence_operator_data;

  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> mass_matrix_operator;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> convective_operator;
  std_cxx11::shared_ptr< InverseMassMatrixOperator<dim,fe_degree,value_type> > inverse_mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> viscous_operator;
  BodyForceOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, value_type> body_force_operator;
  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> gradient_operator;
  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> divergence_operator;

private:
  virtual void create_dofs();

  virtual void data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data);

  // compute vorticity
  void local_compute_vorticity (const MatrixFree<dim,value_type>                 &data,
                                parallel::distributed::Vector<value_type>        &dst,
                                const parallel::distributed::Vector<value_type>  &src,
                                const std::pair<unsigned int,unsigned int>       &cell_range) const;

  // divergence
  void local_compute_divergence (const MatrixFree<dim,value_type>                &data,
                                 parallel::distributed::Vector<value_type>       &dst,
                                 const parallel::distributed::Vector<value_type> &src,
                                 const std::pair<unsigned int,unsigned int>      &cell_range) const;

};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
fill_dbc_and_nbc_sets(std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor)
{
  // Dirichlet boundary conditions: copy Dirichlet boundary ID's from boundary_descriptor.dirichlet_bc (map) to dirichlet_boundary (set)
  for (typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::
       const_iterator it = boundary_descriptor->dirichlet_bc.begin();
       it != boundary_descriptor->dirichlet_bc.end(); ++it)
  {
    dirichlet_boundary.insert(it->first);
  }

  // Neumann boundary conditions: copy Neumann boundary ID's from boundary_descriptor.neumann_bc (map) to neumann_boundary (set)
  for (typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::
       const_iterator it = boundary_descriptor->neumann_bc.begin();
       it != boundary_descriptor->neumann_bc.end(); ++it)
  {
    neumann_boundary.insert(it->first);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                                   periodic_face_pairs,
       std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity_in,
       std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure_in,
       std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> >     field_functions_in)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Setup Navier-Stokes operation ..." << std::endl;

  this->periodic_face_pairs = periodic_face_pairs;
  this->boundary_descriptor_velocity = boundary_descriptor_velocity_in;
  this->boundary_descriptor_pressure = boundary_descriptor_pressure_in;
  this->field_functions = field_functions_in;

  fill_dbc_and_nbc_sets(this->boundary_descriptor_velocity);

  create_dofs();

  // initialize matrix_free_data
  typename MatrixFree<dim,value_type>::AdditionalData additional_data;
  additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme = MatrixFree<dim,value_type>::AdditionalData::partition_partition;
  additional_data.build_face_info = true;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points | update_normal_vectors |
                                          update_values);
  additional_data.periodic_face_pairs_level_0 = periodic_face_pairs;

  data_reinit(additional_data);

  // mass matrix operator
  mass_matrix_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  mass_matrix_operator.initialize(data,fe_param,mass_matrix_operator_data);

  // inverse mass matrix operator
  inverse_mass_matrix_operator.reset(new InverseMassMatrixOperator<dim,fe_degree,value_type>());
  inverse_mass_matrix_operator->initialize(data,
          static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
          static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity));

  // body force operator
  BodyForceOperatorData<dim> body_force_operator_data;
  body_force_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  body_force_operator_data.rhs = field_functions->right_hand_side;
  body_force_operator.initialize(data,fe_param,body_force_operator_data);

  // gradient operator
  gradient_operator_data.dof_index_velocity = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  gradient_operator_data.dof_index_pressure = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  gradient_operator_data.integration_by_parts_of_gradP = param.gradp_integrated_by_parts;
  gradient_operator_data.use_boundary_data = param.gradp_use_boundary_data;
  gradient_operator_data.bc = boundary_descriptor_pressure;
  gradient_operator.initialize(data,fe_param,gradient_operator_data);

  // divergence operator
  divergence_operator_data.dof_index_velocity = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  divergence_operator_data.dof_index_pressure = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure);
  divergence_operator_data.integration_by_parts_of_divU = param.divu_integrated_by_parts;
  divergence_operator_data.use_boundary_data = param.divu_use_boundary_data;
  divergence_operator_data.bc = boundary_descriptor_velocity;
  divergence_operator.initialize(data,fe_param,divergence_operator_data);

  // convective operator
  convective_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  convective_operator_data.bc = boundary_descriptor_velocity;
  convective_operator.initialize(data,fe_param,convective_operator_data);

  // viscous operator
  viscous_operator_data.formulation_viscous_term = param.formulation_viscous_term;
  viscous_operator_data.IP_formulation_viscous = param.IP_formulation_viscous;
  viscous_operator_data.IP_factor_viscous = param.IP_factor_viscous;
  viscous_operator_data.bc = boundary_descriptor_velocity;
  viscous_operator_data.dof_index = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity);
  viscous_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;
  viscous_operator_data.viscosity = param.viscosity;
  viscous_operator.initialize(mapping,data,fe_param,viscous_operator_data);

  dof_index_first_point = 0;
  for(unsigned int d=0;d<dim;++d)
    first_point[d] = 0.0;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    typename DoFHandler<dim>::active_cell_iterator first_cell;
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_p.begin_active(), endc = dof_handler_p.end();
    for(;cell!=endc;++cell)
    {
      if (cell->is_locally_owned())
      {
        first_cell = cell;
        break;
      }
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
  {
    first_point[d] = Utilities::MPI::sum(first_point[d],MPI_COMM_WORLD);
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
create_dofs()
{
  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_u.distribute_mg_dofs(*fe_u);
  dof_handler_p.distribute_dofs(fe_p);
  dof_handler_p.distribute_mg_dofs(fe_p);

  unsigned int ndofs_per_cell_velocity = Utilities::fixed_int_power<fe_degree+1,dim>::value*dim;
  unsigned int ndofs_per_cell_pressure = Utilities::fixed_int_power<fe_degree_p+1,dim>::value;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl << std::endl;

  pcout << "Velocity:" << std::endl;
  print_parameter(pcout,"degree of 1D polynomials",fe_degree);
  print_parameter(pcout,"number of dofs per cell",ndofs_per_cell_velocity);
  print_parameter(pcout,"number of dofs (total)",dof_handler_u.n_dofs());

  pcout << "Pressure:" << std::endl;
  print_parameter(pcout,"degree of 1D polynomials",fe_degree_p);
  print_parameter(pcout,"number of dofs per cell",ndofs_per_cell_pressure);
  print_parameter(pcout,"number of dofs (total)",dof_handler_p.n_dofs());


}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
data_reinit(typename MatrixFree<dim,value_type>::AdditionalData &additional_data)
{
  std::vector<const DoFHandler<dim> * >  dof_handler_vec;

  dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::n_variants));
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity)] = &dof_handler_u;
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure)] = &dof_handler_p;

  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  constraint_matrix_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::n_variants));
  ConstraintMatrix constraint_u, constraint_p;
  constraint_u.close();
  constraint_p.close();
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity)] = &constraint_u;
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure)] = &constraint_p;

  std::vector<Quadrature<1> > quadratures;

  // resize quadratures
  quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::n_variants));
  // velocity
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity)]
              = QGauss<1>(fe_degree+1);
  // pressure
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure)]
              = QGauss<1>(fe_degree_p+1);
  // exact integration of nonlinear convective term
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity_nonlinear)]
              = QGauss<1>(fe_degree + (fe_degree+2)/2);

  data.reinit (mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
prescribe_initial_conditions(parallel::distributed::Vector<value_type> &velocity,
                             parallel::distributed::Vector<value_type> &pressure,
                             double const                              evaluation_time) const
{
  this->field_functions->initial_solution_velocity->set_time(evaluation_time);
  this->field_functions->initial_solution_pressure->set_time(evaluation_time);

  VectorTools::interpolate(mapping, dof_handler_u, *(this->field_functions->initial_solution_velocity), velocity);
  VectorTools::interpolate(mapping, dof_handler_p, *(this->field_functions->initial_solution_pressure), pressure);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
shift_pressure (parallel::distributed::Vector<value_type>  &pressure) const
{
  AssertThrow(this->param.analytical_solution_available == true,
              ExcMessage("The function shift_pressure is intended to be used only if an analytical solution is available!"));

  parallel::distributed::Vector<value_type> vec1(pressure);
  for(unsigned int i=0;i<vec1.local_size();++i)
    vec1.local_element(i) = 1.;
  this->field_functions->analytical_solution_pressure->set_time(evaluation_time);
  double exact = this->field_functions->analytical_solution_pressure->value(first_point);
  double current = 0.;
  if (pressure.locally_owned_elements().is_element(dof_index_first_point))
    current = pressure(dof_index_first_point);
  current = Utilities::MPI::sum(current, MPI_COMM_WORLD);
  pressure.add(exact-current,vec1);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
apply_zero_mean (parallel::distributed::Vector<value_type>  &vector) const
{
  const value_type mean_value = vector.mean_value();
  vector.add(-mean_value);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
compute_vorticity (parallel::distributed::Vector<value_type>       &dst,
                   const parallel::distributed::Vector<value_type> &src) const
{
  dst = 0;

  data.cell_loop (&DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_vorticity,this, dst, src);

  inverse_mass_matrix_operator->apply_inverse_mass_matrix(dst,dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_compute_vorticity(const MatrixFree<dim,value_type>                 &data,
                        parallel::distributed::Vector<value_type>        &dst,
                        const parallel::distributed::Vector<value_type>  &src,
                        const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEval_Velocity_Velocity_linear velocity(data,fe_param,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit(cell);
    velocity.read_dof_values(src);
    velocity.evaluate (false,true,false);
    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
      // omega_vector is a vector with dim components
      // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
      // for dim=2: omega_vector[0] = omega,
      //            omega_vector[1] = 0
      Tensor<1,dim,VectorizedArray<value_type> > omega_vector;
      for (unsigned int d=0; d<number_vorticity_components; ++d)
        omega_vector[d] = omega[d];
      velocity.submit_value (omega_vector, q);
    }
    velocity.integrate (true,false);
    velocity.distribute_local_to_global(dst);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
compute_divergence (parallel::distributed::Vector<value_type>       &dst,
                    const parallel::distributed::Vector<value_type> &src) const
{
  dst = 0;

  data.cell_loop(&DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divergence,
                             this, dst, src);

  inverse_mass_matrix_operator->apply_inverse_mass_matrix(dst,dst);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_compute_divergence (const MatrixFree<dim,value_type>                 &data,
                          parallel::distributed::Vector<value_type>        &dst,
                          const parallel::distributed::Vector<value_type>  &src,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval_velocity.reinit(cell);
    fe_eval_velocity.read_dof_values(src);
    fe_eval_velocity.evaluate(false,true);

    for (unsigned int q=0; q<fe_eval_velocity.n_q_points; q++)
    {
      Tensor<1,dim,VectorizedArray<value_type> > div_vector;
        div_vector[0] = fe_eval_velocity.get_divergence(q);
      fe_eval_velocity.submit_value(div_vector,q);
    }
    fe_eval_velocity.integrate(true,false);
    fe_eval_velocity.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
evaluate_convective_term (parallel::distributed::Vector<value_type>       &dst,
                          parallel::distributed::Vector<value_type> const &src,
                          value_type const                                evaluation_time) const
{
  convective_operator.evaluate(dst,src,evaluation_time);
}


#endif /* INCLUDE_DGNAVIERSTOKESBASE_H_ */
