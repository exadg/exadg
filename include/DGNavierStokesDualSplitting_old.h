/*
 * DG_NavierStokesDualSplitting.h
 *
 *  Created on: May 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_
#define INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_

#include <deal.II/matrix_free/operators.h>

#include "FEEvaluationWrapper.h"
#include "XWall.h"
#include "InputParameters.h"
#include "FE_Parameters.h"


#include "InverseMassMatrix.h"
#include "HelmholtzSolver.h"
#include "ProjectionSolver.h"
#include "poisson_solver.h"


using namespace dealii;

//forward declarations
template<int dim> class AnalyticalSolution;
template<int dim> class RHS;
template<int dim> class NeumannBoundaryVelocity;
template<int dim> class PressureBC_dudt;

enum class DofHandlerSelector{
  velocity = 0,
  pressure = 1,
  wdist_tauw = 2,
  enriched = 3,
  n_variants = enriched+1
};

enum class QuadratureSelector{
  velocity = 0,
  pressure = 1,
  velocity_nonlinear = 2,
  enriched = 3,
  n_variants = enriched+1
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class NavierStokesOperation
{
public:
  typedef double value_type;
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;
  static const bool is_xwall = false;//(fe_degree_xwall>0) ? true : false;
  static const unsigned int n_actual_q_points_vel_nonlinear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+(fe_degree+2)/2;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+1;

  /*
   * nomenclature typdedef FEEvaluationWrapper:
   * FEEval_name1_name2 : name1 specifies the dof handler, name2 the quadrature formula
   * example: FEEval_Pressure_Velocity_linear: dof handler for pressure (scalar quantity),
   * quadrature formula with fe_degree_velocity+1 quadrature points
   */

  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_nonlinear;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,is_xwall> FEEval_Velocity_scalar_Velocity_linear;
  typedef FEEvaluationWrapper<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,false> FEEval_Pressure_Velocity_linear;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,number_vorticity_components,value_type,is_xwall> FEEval_Vorticity_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_linear,1,value_type,false> FEFaceEval_Pressure_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree_p,fe_degree_xwall,n_actual_q_points_vel_nonlinear,1,value_type,false> FEFaceEval_Pressure_Velocity_nonlinear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,number_vorticity_components,value_type,is_xwall> FEFaceEval_Vorticity_Velocity_nonlinear;

  NavierStokesOperation(parallel::distributed::Triangulation<dim> const & triangulation,
                        InputParameters const                           & parameter);

  void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
              std::set<types::boundary_id> dirichlet_bc_indicator,
              std::set<types::boundary_id> neumann_bc_indicator);

  void setup_solvers (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs);

  ~NavierStokesOperation()
  {
    data.clear();
  }

  // convection
  void  rhs_convection (const parallel::distributed::BlockVector<value_type>  &src,
                        parallel::distributed::BlockVector<value_type>        &dst) const;

  void  res_impl_convection (const parallel::distributed::BlockVector<value_type> &temp,
                             const parallel::distributed::BlockVector<value_type> &src,
                             parallel::distributed::BlockVector<value_type>       &dst) const;

  unsigned int solve_implicit_convective_step(parallel::distributed::BlockVector<value_type> &solution,
                                              parallel::distributed::BlockVector<value_type> &residual_convection,
                                              parallel::distributed::BlockVector<value_type> &delta_velocity,
                                              parallel::distributed::BlockVector<value_type> const &temp);

  void apply_linearized_convection (const parallel::distributed::BlockVector<value_type> &src,
                                    parallel::distributed::BlockVector<value_type>       &dst) const;

  void  compute_rhs (parallel::distributed::BlockVector<value_type>  &dst) const;

  // rhs pressure
  void  rhs_pressure (const parallel::distributed::BlockVector<value_type>  &src,
                      parallel::distributed::Vector<value_type>             &dst) const;

  void rhs_pressure_divergence_term (const parallel::distributed::BlockVector<value_type>  &src,
                                     parallel::distributed::Vector<value_type>             &dst) const;

  void rhs_pressure_BC_term (const parallel::distributed::BlockVector<value_type> &src,
                             parallel::distributed::Vector<value_type>            &dst) const;

  void rhs_pressure_convective_term (const parallel::distributed::BlockVector<value_type>  &src,
                                     parallel::distributed::Vector<value_type>             &dst) const;

  void rhs_pressure_viscous_term (const parallel::distributed::BlockVector<value_type>  &src,
                                  parallel::distributed::Vector<value_type>             &dst) const;

  void apply_nullspace_projection (parallel::distributed::Vector<value_type>  &dst) const;

  //shift pressure (pure Dirichlet BC case)
  void  shift_pressure (parallel::distributed::Vector<value_type> &pressure) const;

  unsigned int solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                               const parallel::distributed::Vector<value_type>  &src) const;

  // projection
  unsigned int solve_projection (parallel::distributed::BlockVector<value_type>       &dst,
                                 const parallel::distributed::BlockVector<value_type> &src,
                                 const parallel::distributed::BlockVector<value_type> &velocity_n,
                                 double const                                         cfl) const;

  void rhs_projection (const parallel::distributed::BlockVector<value_type> &src_velocity,
                       const parallel::distributed::Vector<value_type>      &src_pressure,
                       parallel::distributed::BlockVector<value_type>       &dst) const;


  // viscous
  unsigned int solve_viscous (parallel::distributed::BlockVector<value_type>       &dst,
                              const parallel::distributed::BlockVector<value_type> &src);

  void  rhs_viscous (const parallel::distributed::BlockVector<value_type> &src,
                     parallel::distributed::BlockVector<value_type>       &dst) const;

  // vorticity
  void compute_vorticity (const parallel::distributed::BlockVector<value_type>  &src,
                          parallel::distributed::BlockVector<value_type>        &dst) const;

  // divergence
  void compute_divergence (const parallel::distributed::BlockVector<value_type> &src,
                           parallel::distributed::Vector<value_type>            &dst,
                           const bool                                           apply_inv_mass_matrix) const;

//  void precompute_inverse_mass_matrix();

//  void xwall_projection();

  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }

  MappingQ<dim> const & get_mapping() const
  {
    return mapping;
  }

  FE_DGQArbitraryNodes<dim> const & get_fe_u() const
  {
    return fe_u;
  }

  FE_DGQArbitraryNodes<dim> const & get_fe_p() const
  {
    return fe_p;
  }

  FE_DGQArbitraryNodes<dim> const & get_fe_xwall() const
  {
    return fe_xwall;
  }

  DoFHandler<dim> const & get_dof_handler_u() const
  {
    return dof_handler_u;
  }

  DoFHandler<dim> const & get_dof_handler_p() const
  {
    return dof_handler_p;
  }

  DoFHandler<dim> const & get_dof_handler_xwall() const
  {
    return dof_handler_xwall;
  }

  std::vector<parallel::distributed::Vector<value_type> > const & get_xwallstatevec() const
  {
    return fe_param.xwallstatevec;
  }

  XWall<dim,fe_degree,fe_degree_xwall> const & get_XWall() const
  {
    return xwall;
  }

  double get_viscosity() const
  {
    return viscosity;
  }

  FEParameters<value_type> const & get_fe_parameters() const
  {
    return fe_param;
  }

  void set_gamma0(double const gamma0_in)
  {
    gamma0 = gamma0_in;
  }

  void set_time(double const current_time)
  {
    time = current_time;
  }

  void set_time_step(double const time_step_in)
  {
    time_step = time_step_in;
  }

  void initialize_block_vector_velocity(parallel::distributed::BlockVector<value_type> &src) const
  {
    src.reinit(dim);

    data.initialize_dof_vector(src.block(0),
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for(unsigned int d=1;d<dim;++d)
      src.block(d) = src.block(0);

    src.collect_sizes();
  }
  void initialize_scalar_vector_velocity(parallel::distributed::Vector<value_type> &src) const
  {
    data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
  }

  void initialize_vector_pressure(parallel::distributed::Vector<value_type> &src) const
  {
    data.initialize_dof_vector(src,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));
  }

  void initialize_block_vector_vorticity(parallel::distributed::BlockVector<value_type> &src) const
  {
    src.reinit(number_vorticity_components);

    data.initialize_dof_vector(src.block(0),
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for (unsigned int d=1;d<number_vorticity_components;++d)
      src.block(d) = src.block(0);

    src.collect_sizes();
  }

  void prescribe_initial_conditions(parallel::distributed::BlockVector<value_type> &velocity,
                                    parallel::distributed::Vector<value_type> &pressure,
                                    double const evaluation_time) const;

private:
  MatrixFree<dim,value_type> data;

  FE_DGQArbitraryNodes<dim>  fe_u;
  FE_DGQArbitraryNodes<dim>  fe_p;
  FE_DGQArbitraryNodes<dim>  fe_xwall;

  MappingQ<dim> mapping;

  DoFHandler<dim>  dof_handler_u;
  DoFHandler<dim>  dof_handler_p;
  DoFHandler<dim>  dof_handler_xwall;

  double time, time_step;
  double gamma0;
  const double viscosity;

  PoissonSolver<dim> pressure_poisson_solver;

  std_cxx11::shared_ptr<ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> > projection_solver;

  HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> helmholtz_operator;
  HelmholtzSolver<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> helmholtz_solver;

  AlignedVector<VectorizedArray<value_type> > element_volume;

  Point<dim> first_point;
  types::global_dof_index dof_index_first_point;

  std::vector<Table<2,VectorizedArray<value_type> > > matrices;

  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;

  std::set<types::boundary_id> dirichlet_boundary;
  std::set<types::boundary_id> neumann_boundary;

  InputParameters const & param;

  FEParameters<value_type> fe_param;

  XWall<dim,fe_degree,fe_degree_xwall> xwall;

  parallel::distributed::BlockVector<value_type> velocity_linear;

  void create_dofs();

  // explicit convective step
  void local_rhs_convection (const MatrixFree<dim,value_type>                     &data,
                             parallel::distributed::BlockVector<value_type>       &dst,
                             const parallel::distributed::BlockVector<value_type> &src,
                             const std::pair<unsigned int,unsigned int>           &cell_range) const;

  void local_rhs_convection_face (const MatrixFree<dim,value_type>                      &data,
                                  parallel::distributed::BlockVector<value_type>        &dst,
                                  const parallel::distributed::BlockVector<value_type>  &src,
                                  const std::pair<unsigned int,unsigned int>            &face_range) const;

  void local_rhs_convection_boundary_face(const MatrixFree<dim,value_type>                      &data,
                                          parallel::distributed::BlockVector<value_type>        &dst,
                                          const parallel::distributed::BlockVector<value_type>  &src,
                                          const std::pair<unsigned int,unsigned int>            &face_range) const;

  // implicit convective step
  void local_res_impl_convection_const_part (const MatrixFree<dim,value_type>                       &data,
                                             parallel::distributed::BlockVector<value_type>         &dst,
                                             const parallel::distributed::BlockVector<value_type>   &src,
                                             const std::pair<unsigned int,unsigned int>             &cell_range) const;

  void local_res_impl_convection (const MatrixFree<dim,value_type>                      &data,
                                  parallel::distributed::BlockVector<value_type>        &dst,
                                  const parallel::distributed::BlockVector<value_type>  &src,
                                  const std::pair<unsigned int,unsigned int>            &cell_range) const;

  void local_res_impl_convection_face (const MatrixFree<dim,value_type>                     &data,
                                       parallel::distributed::BlockVector<value_type>       &dst,
                                       const parallel::distributed::BlockVector<value_type> &src,
                                       const std::pair<unsigned int,unsigned int>           &face_range) const;

  void local_res_impl_convection_boundary_face(const MatrixFree<dim,value_type>                     &data,
                                               parallel::distributed::BlockVector<value_type>       &dst,
                                               const parallel::distributed::BlockVector<value_type> &src,
                                               const std::pair<unsigned int,unsigned int>           &face_range) const;

  // linearized convective problem
  void local_apply_linearized_convection (const MatrixFree<dim,value_type>                 &data,
                                          parallel::distributed::BlockVector<double>       &dst,
                                          const parallel::distributed::BlockVector<double> &src,
                                          const std::pair<unsigned int,unsigned int>       &cell_range) const;

  void local_apply_linearized_convection_face (const MatrixFree<dim,value_type>                 &data,
                                               parallel::distributed::BlockVector<double>       &dst,
                                               const parallel::distributed::BlockVector<double> &src,
                                               const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_apply_linearized_convection_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                                        parallel::distributed::BlockVector<double>       &dst,
                                                        const parallel::distributed::BlockVector<double> &src,
                                                        const std::pair<unsigned int,unsigned int>       &face_range) const;

  // body force term
  void local_compute_rhs (const MatrixFree<dim,value_type>                      &data,
                          parallel::distributed::BlockVector<value_type>        &dst,
                          const parallel::distributed::BlockVector<value_type>  &,
                          const std::pair<unsigned int,unsigned int>            &cell_range) const;

  // rhs pressure: divergence term
  void local_rhs_pressure_divergence_term (const MatrixFree<dim,value_type>                     &data,
                                           parallel::distributed::Vector<value_type>            &dst,
                                           const parallel::distributed::BlockVector<value_type> &src,
                                           const std::pair<unsigned int,unsigned int>           &cell_range) const;

  void local_rhs_pressure_divergence_term_face (const MatrixFree<dim,value_type>                      &data,
                                                parallel::distributed::Vector<value_type>             &dst,
                                                const parallel::distributed::BlockVector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>            &face_range) const;

  void local_rhs_pressure_divergence_term_boundary_face(const MatrixFree<dim,value_type>                      &data,
                                                        parallel::distributed::Vector<value_type>             &dst,
                                                        const parallel::distributed::BlockVector<value_type>  &src,
                                                        const std::pair<unsigned int,unsigned int>            &face_range) const;

  // rhs pressure: BC term
  void local_rhs_pressure_BC_term (const MatrixFree<dim,value_type>                     &data,
                                   parallel::distributed::Vector<value_type>            &dst,
                                   const parallel::distributed::BlockVector<value_type> &src,
                                   const std::pair<unsigned int,unsigned int>           &cell_range) const;

  void local_rhs_pressure_BC_term_face (const MatrixFree<dim,value_type>                      &data,
                                        parallel::distributed::Vector<value_type>             &dst,
                                        const parallel::distributed::BlockVector<value_type>  &src,
                                        const std::pair<unsigned int,unsigned int>            &face_range) const;

  void local_rhs_pressure_BC_term_boundary_face(const MatrixFree<dim,value_type>                      &data,
                                                parallel::distributed::Vector<value_type>             &dst,
                                                const parallel::distributed::BlockVector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>            &face_range) const;

  // rhs pressure: convective term
  void local_rhs_pressure_convective_term (const MatrixFree<dim,value_type>                     &data,
                                           parallel::distributed::Vector<value_type>            &dst,
                                           const parallel::distributed::BlockVector<value_type> &src,
                                           const std::pair<unsigned int,unsigned int>           &cell_range) const;

  void local_rhs_pressure_convective_term_face (const MatrixFree<dim,value_type>                      &data,
                                                parallel::distributed::Vector<value_type>             &dst,
                                                const parallel::distributed::BlockVector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>            &face_range) const;

  void local_rhs_pressure_convective_term_boundary_face(const MatrixFree<dim,value_type>                      &data,
                                                        parallel::distributed::Vector<value_type>             &dst,
                                                        const parallel::distributed::BlockVector<value_type>  &src,
                                                        const std::pair<unsigned int,unsigned int>            &face_range) const;

  // rhs pressure: viscous term
  void local_rhs_pressure_viscous_term (const MatrixFree<dim,value_type>                     &data,
                                        parallel::distributed::Vector<value_type>            &dst,
                                        const parallel::distributed::BlockVector<value_type> &src,
                                        const std::pair<unsigned int,unsigned int>           &cell_range) const;

  void local_rhs_pressure_viscous_term_face (const MatrixFree<dim,value_type>                      &data,
                                             parallel::distributed::Vector<value_type>             &dst,
                                             const parallel::distributed::BlockVector<value_type>  &src,
                                             const std::pair<unsigned int,unsigned int>            &face_range) const;

  void local_rhs_pressure_viscous_term_boundary_face(const MatrixFree<dim,value_type>                      &data,
                                                     parallel::distributed::Vector<value_type>             &dst,
                                                     const parallel::distributed::BlockVector<value_type>  &src,
                                                     const std::pair<unsigned int,unsigned int>            &face_range) const;

  // projection step
  void local_rhs_projection_mass_term (const MatrixFree<dim,value_type>                     &data,
                                       parallel::distributed::BlockVector<value_type>       &dst,
                                       const parallel::distributed::BlockVector<value_type> &src,
                                       const std::pair<unsigned int,unsigned int>           &cell_range) const;

  void local_rhs_projection_gradient_term (const MatrixFree<dim,value_type>                &data,
                                           parallel::distributed::BlockVector<value_type>  &dst,
                                           const parallel::distributed::Vector<value_type> &src,
                                           const std::pair<unsigned int,unsigned int>      &cell_range) const;

  void local_rhs_projection_gradient_term_face (const MatrixFree<dim,value_type>                 &data,
                                                parallel::distributed::BlockVector<value_type>   &dst,
                                                const parallel::distributed::Vector<value_type>  &src,
                                                const std::pair<unsigned int,unsigned int>       &face_range) const;

  void local_rhs_projection_gradient_term_boundary_face (const MatrixFree<dim,value_type>                 &data,
                                                         parallel::distributed::BlockVector<value_type>   &dst,
                                                         const parallel::distributed::Vector<value_type>  &src,
                                                         const std::pair<unsigned int,unsigned int>       &face_range) const;

  //viscous
  void local_rhs_viscous (const MatrixFree<dim,value_type>                      &data,
                          parallel::distributed::BlockVector<value_type>        &dst,
                          const parallel::distributed::BlockVector<value_type>  &src,
                          const std::pair<unsigned int,unsigned int>            &cell_range) const;

  void local_rhs_viscous_face (const MatrixFree<dim,value_type>                     &data,
                               parallel::distributed::BlockVector<value_type>       &dst,
                               const parallel::distributed::BlockVector<value_type> &src,
                               const std::pair<unsigned int,unsigned int>           &face_range) const;

  void local_rhs_viscous_boundary_face(const MatrixFree<dim,value_type>                     &data,
                                       parallel::distributed::BlockVector<value_type>       &dst,
                                       const parallel::distributed::BlockVector<value_type> &src,
                                       const std::pair<unsigned int,unsigned int>           &face_range) const;

  // compute vorticity
  void local_compute_vorticity (const MatrixFree<dim,value_type>                      &data,
                                parallel::distributed::BlockVector<value_type>        &dst,
                                const parallel::distributed::BlockVector<value_type>  &src,
                                const std::pair<unsigned int,unsigned int>            &cell_range) const;

  // divergence
  void local_compute_divergence (const MatrixFree<dim,value_type>                     &data,
                                 parallel::distributed::Vector<value_type>            &dst,
                                 const parallel::distributed::BlockVector<value_type> &src,
                                 const std::pair<unsigned int,unsigned int>           &cell_range) const;


//  void local_precompute_mass_matrix(const MatrixFree<dim,value_type>                &data,
//                      std::vector<parallel::distributed::Vector<value_type> >    &,
//                      const std::vector<parallel::distributed::Vector<value_type> >  &,
//                      const std::pair<unsigned int,unsigned int>          &cell_range);
//
//  void local_project_xwall(const MatrixFree<dim,value_type>                &data,
//                      parallel::distributed::BlockVector<value_type>    &dst,
//                      const std::vector<parallel::distributed::Vector<value_type> >  &src,
//                      const std::pair<unsigned int,unsigned int>          &cell_range);

};

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::NavierStokesOperation(
                                                                       parallel::distributed::Triangulation<dim> const & triangulation,
                                                                       InputParameters const & parameter)
    :
    fe_u(QGaussLobatto<1>(fe_degree+1)),
    fe_p(QGaussLobatto<1>(fe_degree_p+1)),
    fe_xwall(QGaussLobatto<1>(fe_degree_xwall+1)),
    mapping(fe_degree),
    dof_handler_u(triangulation),
    dof_handler_p(triangulation),
    dof_handler_xwall(triangulation),
    time(0.0),
    time_step(1.0),
    gamma0(1.0),
    viscosity(parameter.viscosity),
    element_volume(0),
    dof_index_first_point(0),
    param(parameter),
    fe_param(param),
    xwall(dof_handler_u,&data,element_volume,fe_param)
  {}

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
         std::set<types::boundary_id> dirichlet_bc_indicator,
         std::set<types::boundary_id> neumann_bc_indicator)
  {
    dirichlet_boundary = dirichlet_bc_indicator;
    neumann_boundary = neumann_bc_indicator;

    create_dofs();

    xwall.initialize_constraints(periodic_face_pairs);

    // initialize matrix_free_data
    typename MatrixFree<dim,value_type>::AdditionalData additional_data;
    additional_data.mpi_communicator = MPI_COMM_WORLD;
    additional_data.tasks_parallel_scheme = MatrixFree<dim,value_type>::AdditionalData::partition_partition;
    additional_data.build_face_info = true;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points | update_normal_vectors |
                                            update_values);
    additional_data.periodic_face_pairs_level_0 = periodic_face_pairs;

    std::vector<const DoFHandler<dim> * >  dof_handler_vec;

    dof_handler_vec.resize(static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::n_variants));
    dof_handler_vec[static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity)] = &dof_handler_u;
    dof_handler_vec[static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure)] = &dof_handler_p;
    dof_handler_vec[static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::wdist_tauw)] = &xwall.ReturnDofHandlerWallDistance();
    dof_handler_vec[static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::enriched)] = &dof_handler_xwall;

    ConstraintMatrix constraint, constraint_p;
    constraint.close();
    constraint_p.close();
    std::vector<const ConstraintMatrix *> constraint_matrix_vec;
    constraint_matrix_vec.push_back(&constraint);
    constraint_matrix_vec.push_back(&constraint_p);
    constraint_matrix_vec.push_back(&xwall.ReturnConstraintMatrix());
    constraint_matrix_vec.push_back(&constraint);

    std::vector<Quadrature<1> > quadratures;

    // resize quadratures
    quadratures.resize(static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::n_variants));

    // velocity
    quadratures[static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity)]
                = QGauss<1>(fe_degree+1);
    // pressure
    quadratures[static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::pressure)]
                = QGauss<1>(fe_degree_p+1);
    // exact integration of nonlinear convective term
    quadratures[static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity_nonlinear)]
                = QGauss<1>(fe_degree + (fe_degree+2)/2);
    // enrichment
    quadratures[static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::enriched)]
                = QGauss<1>(n_q_points_1d_xwall);

    data.reinit (mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);

    inverse_mass_matrix_operator.initialize(data,
            static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
            static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity));

    if(param.convective_step_implicit == true)
      initialize_block_vector_velocity(velocity_linear);

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
      first_point[d] = Utilities::MPI::sum(first_point[d],MPI_COMM_WORLD);

  #ifdef XWALL
    matrices.resize(data.n_macro_cells());
  #endif

    QGauss<dim> quadrature(fe_degree+1);
    FEValues<dim> fe_values(mapping, dof_handler_u.get_fe(), quadrature, update_JxW_values);
    element_volume.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
    for (unsigned int i=0; i<data.n_macro_cells()+data.n_macro_ghost_cells(); ++i)
    {
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
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  prescribe_initial_conditions(parallel::distributed::BlockVector<value_type> &velocity,
                               parallel::distributed::Vector<value_type> &pressure,
                               double const evaluation_time) const
  {
    for(unsigned int d=0;d<dim;++d)
      VectorTools::interpolate(mapping, dof_handler_u, AnalyticalSolution<dim>(d,evaluation_time), velocity.block(d));
    VectorTools::interpolate(mapping, dof_handler_p, AnalyticalSolution<dim>(dim,evaluation_time), pressure);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  setup_solvers (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs)
  {
    // PCG - solver for pressure Poisson equation
    PoissonSolverData<dim> solver_data;
    solver_data.poisson_dof_index = 1;
    solver_data.poisson_quad_index = 1;
    solver_data.periodic_face_pairs_level0 = periodic_face_pairs;
    solver_data.penalty_factor = param.IP_factor_pressure;

    // TODO
    /*
     * approach of Ferrer et al.: increase penalty parameter when reducing the time step
     * in order to improve stability in the limit of small time steps
     */

    /*
    double dt_ref = 0.1;
    solver_data.penalty_factor = param.IP_factor_pressure/time_step*dt_ref;
    */

    solver_data.solver_tolerance_abs = param.abs_tol_pressure;
    solver_data.solver_tolerance = param.rel_tol_pressure;
    solver_data.dirichlet_boundaries = neumann_boundary;
    solver_data.neumann_boundaries = dirichlet_boundary;
    solver_data.coarse_solver = PoissonSolverData<dim>::coarse_chebyshev_smoother;//coarse_chebyshev_smoother;//coarse_iterative_jacobi;
    pressure_poisson_solver.initialize(mapping, data, solver_data);

    // initialize projection solver
    ProjectionOperatorData projection_operator_data;
    projection_operator_data.penalty_parameter_divergence = param.penalty_factor_divergence;
    projection_operator_data.penalty_parameter_continuity = param.penalty_factor_continuity;
    projection_operator_data.solve_stokes_equations = param.solve_stokes_equations;

    if(param.projection_type == ProjectionType::NoPenalty)
    {
      projection_solver.reset(new ProjectionSolverNoPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
          data,
          fe_param,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
          static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
          projection_operator_data));
    }
    else if(param.projection_type == ProjectionType::DivergencePenalty &&
            param.solver_projection == SolverProjection::LU)
    {
      projection_solver.reset(new DirectProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
                                data,
                                fe_param,
                                static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
                                projection_operator_data));
    }
    else if(param.projection_type == ProjectionType::DivergencePenalty &&
            param.solver_projection == SolverProjection::PCG)
    {
      ProjectionSolverData projection_solver_data;
      projection_solver_data.solver_tolerance_abs = param.abs_tol_projection;
      projection_solver_data.solver_tolerance_rel = param.rel_tol_projection;
      projection_solver_data.solver_projection = param.solver_projection;
      projection_solver_data.preconditioner_projection = param.preconditioner_projection;

      projection_solver.reset(new IterativeProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
                                data,
                                fe_param,
                                static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
                                static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
                                projection_operator_data,
                                projection_solver_data));
    }
    else if(param.projection_type == ProjectionType::DivergenceAndContinuityPenalty)
    {
      ProjectionSolverData projection_solver_data;
      projection_solver_data.solver_tolerance_abs = param.abs_tol_projection;
      projection_solver_data.solver_tolerance_rel = param.rel_tol_projection;
      projection_solver_data.solver_projection = param.solver_projection;
      projection_solver_data.preconditioner_projection = param.preconditioner_projection;

      projection_solver.reset(new IterativeProjectionSolverDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
                                data,
                                fe_param,
                                static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
                                static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity),
                                projection_operator_data,
                                projection_solver_data));
    }

    // initialize viscous solver
    HelmholtzOperatorData<dim,value_type> helmholtz_operator_data;
    helmholtz_operator_data.formulation_viscous_term = param.formulation_viscous_term;
    helmholtz_operator_data.IP_formulation_viscous = param.IP_formulation_viscous;
    helmholtz_operator_data.IP_factor_viscous = param.IP_factor_viscous;
    helmholtz_operator_data.dirichlet_boundaries = dirichlet_boundary;
    helmholtz_operator_data.neumann_boundaries = neumann_boundary;
    helmholtz_operator_data.dof_index = static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity);

    helmholtz_operator.initialize(mapping,data,fe_param,helmholtz_operator_data);
    helmholtz_operator.set_mass_matrix_coefficient(gamma0/time_step);
    helmholtz_operator.set_constant_viscosity(viscosity);
    // helmholtz_operator.set_variable_viscosity(viscosity);

    HelmholtzSolverData helmholtz_solver_data;
    helmholtz_solver_data.solver_viscous = param.solver_viscous;
    helmholtz_solver_data.preconditioner_viscous = param.preconditioner_viscous;
    helmholtz_solver_data.solver_tolerance_abs = param.abs_tol_viscous;
    helmholtz_solver_data.solver_tolerance_rel = param.rel_tol_viscous;

    helmholtz_solver.initialize(helmholtz_operator,helmholtz_solver_data,data,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity));
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler_u.distribute_dofs(fe_u);
    dof_handler_p.distribute_dofs(fe_p);
    dof_handler_xwall.distribute_dofs(fe_xwall);
    //dof_handler.distribute_mg_dofs(fe_u);
    dof_handler_p.distribute_mg_dofs(fe_p);
    //dof_handler_xwall.distribute_mg_dofs(fe_xwall);

    float ndofs_per_cell_velocity = pow(float(fe_degree+1),dim)*dim;
    float ndofs_per_cell_pressure = pow(float(fe_degree_p+1),dim);
    float ndofs_per_cell_xwall    = pow(float(fe_degree_xwall+1),dim)*dim;

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
      << "Velocity:" << std::endl
      << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree << std::endl
      << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << (int)ndofs_per_cell_velocity << std::endl
      << "  number of dofs (velocity):\t" << std::fixed << std::setw(10) << std::right << dof_handler_u.n_dofs()*dim << std::endl
      << "Pressure:" << std::endl
      << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree_p << std::endl
      << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << (int)ndofs_per_cell_pressure << std::endl
      << "  number of dofs (pressure):\t" << std::fixed << std::setw(10) << std::right << dof_handler_p.n_dofs() << std::endl
      << "Enrichment:" << std::endl
      << "  degree of 1D polynomials:\t" << std::fixed << std::setw(10) << std::right << fe_degree_xwall << std::endl
      << "  number of dofs per cell:\t"  << std::fixed << std::setw(10) << std::right << (int)ndofs_per_cell_xwall << std::endl
      << "  number of dofs (xwall):\t"   << std::fixed << std::setw(10) << std::right << dof_handler_xwall.n_dofs()*dim << std::endl;
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  struct LinearizedConvectionMatrix : public Subscriptor
  {
    void initialize(NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> &ns_op)
    {
      ns_operation = &ns_op;
    }
    void vmult (parallel::distributed::BlockVector<double> &dst,
        const parallel::distributed::BlockVector<double> &src) const
    {
      ns_operation->apply_linearized_convection(src,dst);
    }
    NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> *ns_operation;
  };

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  unsigned int NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  solve_implicit_convective_step(parallel::distributed::BlockVector<value_type> &solution,
                                 parallel::distributed::BlockVector<value_type> &residual_convection,
                                 parallel::distributed::BlockVector<value_type> &delta_velocity,
                                 parallel::distributed::BlockVector<value_type> const &sum_alphai_ui)
  {
    res_impl_convection(sum_alphai_ui,solution,residual_convection);
    for (unsigned int d=0; d<dim; ++d)
    {
      residual_convection.block(d) *= -1.0; // multiply by -1.0 since rhs of linearized problem is -1.0 * (residual of nonlinear equation)
    }
    value_type norm_r = residual_convection.l2_norm();
    value_type norm_r_0 = norm_r;
    // std::cout << "Norm of nonlinear residual: " << norm_r << std::endl;

    const double ABSTOL_NEWTON = 1.e-12;
    const double RELTOL_NEWTON = 1.e-6;
    const unsigned int MAXITER_NEWTON = 1e3;

    // Newton iteration
    unsigned int n_iter = 0;
    while(norm_r > ABSTOL_NEWTON && norm_r/norm_r_0 > RELTOL_NEWTON && n_iter < MAXITER_NEWTON)
    {
      // solve linearized problem
      for(unsigned int d=0;d<dim;++d)
        velocity_linear.block(d) = solution.block(d);

      // reset increment
      delta_velocity = 0.0;

      const double ABSTOL_LINEAR_CONVECTION = 1.e-12;
      const double RELTOL_LINEAR_CONVECTION = 1.e-6;
      const unsigned int MAXITER_LINEAR = 1e5;
      ReductionControl solver_control_conv (MAXITER_LINEAR, ABSTOL_LINEAR_CONVECTION, RELTOL_LINEAR_CONVECTION);
      SolverFGMRES<parallel::distributed::BlockVector<value_type> > linear_solver_conv (solver_control_conv);
      LinearizedConvectionMatrix<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> linearized_convection_matrix;
      linearized_convection_matrix.initialize(*this);
      InverseMassMatrixPreconditionerVelocity<dim,fe_degree,value_type> preconditioner_conv(data,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
          static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity));
      try
      {
        linear_solver_conv.solve (linearized_convection_matrix, delta_velocity, residual_convection, preconditioner_conv); //PreconditionIdentity());
        /*
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "Linear solver:" << std::endl;
          std::cout << "  Number of iterations: " << solver_control_conv.last_step() << std::endl;
          std::cout << "  Initial value: " << solver_control_conv.initial_value() << std::endl;
          std::cout << "  Last value: " << solver_control_conv.last_value() << std::endl << std::endl;
        }
        */
      }
      catch (SolverControl::NoConvergence &)
      {
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
          std::cout << "Linear solver of convective step failed to solve to given tolerance." << std::endl;
      }
      // update solution
      for(unsigned int d=0; d<dim; ++d)
      {
        solution.block(d).add(1.0, delta_velocity.block(d));
      }
      // calculate residual of nonlinear equation
      res_impl_convection(sum_alphai_ui,solution,residual_convection);
      for (unsigned int d=0; d<dim; ++d)
      {
        residual_convection.block(d) *= -1.0; // multiply by -1.0 since rhs of linearized problem is -1.0 * (residual of nonlinear equation)
      }

      norm_r = residual_convection.l2_norm();
      ++n_iter;
      // std::cout << "Norm of nonlinear residual: " << norm_r << std::endl;
    }

    if(n_iter >= MAXITER_NEWTON)
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        std::cout<<"Newton solver failed to solve nonlinear convective problem to given tolerance. Maximum number of iterations exceeded!" << std::endl;
    }

    return n_iter;
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
  rhs_convection (const parallel::distributed::BlockVector<value_type>  &src,
                  parallel::distributed::BlockVector<value_type>        &dst) const
  {
    dst = 0;

    data.loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_convection_boundary_face,
              this, dst, src);

    inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  res_impl_convection (const parallel::distributed::BlockVector<value_type> &temp,
                       const parallel::distributed::BlockVector<value_type> &src,
                       parallel::distributed::BlockVector<value_type>       &dst) const
  {
    dst = 0;

    // constant part of residual, i.e., independent of current solution of the nonlinear equation
    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_res_impl_convection_const_part,
              this, dst, temp);

    // variable part, i.e., part of residual that depends on the current solution of the nonlinear equation
    data.loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_res_impl_convection,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_res_impl_convection_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_res_impl_convection_boundary_face,
              this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_linearized_convection (const parallel::distributed::BlockVector<value_type> &src,
                               parallel::distributed::BlockVector<value_type>       &dst) const
  {
    for(unsigned int d=0;d<dim;++d)
    {
      dst.block(d)=0;
    }
    data.loop ( &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_linearized_convection,
                &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_linearized_convection_face,
                &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_linearized_convection_boundary_face,
                this, dst, src);

//  data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_linearized_convection,this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_rhs (parallel::distributed::BlockVector<value_type>  &dst) const
  {
    dst = 0;

    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_rhs,this, dst, dst);

    inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  unsigned int NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  solve_viscous (parallel::distributed::BlockVector<value_type>       &dst,
                 const parallel::distributed::BlockVector<value_type> &src)
  {
    helmholtz_operator.set_mass_matrix_coefficient(gamma0/time_step);
    // helmholtz_operator.set_constant_viscosity(viscosity);
    // helmholtz_operator.set_variable_viscosity(viscosity);
    unsigned int n_iter = helmholtz_solver.solve(dst,src);

    return n_iter;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_viscous (const parallel::distributed::BlockVector<value_type> &src,
               parallel::distributed::BlockVector<value_type>       &dst) const
  {
    dst = 0;

    data.loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous_face,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_viscous_boundary_face,
               this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection (const MatrixFree<dim,value_type>                      &data,
                        parallel::distributed::BlockVector<value_type>        &dst,
                        const parallel::distributed::BlockVector<value_type>  &src,
                        const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,fe_param,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_velocity.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,u);
        fe_eval_velocity.submit_gradient (F, q);
      }
      fe_eval_velocity.integrate (false,true);
      fe_eval_velocity.distribute_local_to_global (dst,0,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection_face (const MatrixFree<dim,value_type>                     &data,
                             parallel::distributed::BlockVector<value_type>       &dst,
                             const parallel::distributed::BlockVector<value_type> &src,
                             const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,fe_param,true,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity_neighbor(data,fe_param,false,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate(true, false);

      fe_eval_velocity_neighbor.reinit (face);
      fe_eval_velocity_neighbor.read_dof_values(src,0,dim);
      fe_eval_velocity_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);

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

        fe_eval_velocity.submit_value(-lf_flux,q);
        fe_eval_velocity_neighbor.submit_value(lf_flux,q);
      }
      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.distribute_local_to_global(dst,0,dim);
      fe_eval_velocity_neighbor.integrate(true,false);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst,0,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_convection_boundary_face (const MatrixFree<dim,value_type>                     &data,
                                      parallel::distributed::BlockVector<value_type>       &dst,
                                      const parallel::distributed::BlockVector<value_type> &src,
                                      const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,fe_param,true,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);

          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
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
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
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

          fe_eval_velocity.submit_value(-lf_flux,q);
        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;
          const VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);

          Tensor<1,dim,VectorizedArray<value_type> > jump_value;
          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = uM*uM_n;
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

          fe_eval_velocity.submit_value(-lf_flux,q);
        }
      }

      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.distribute_local_to_global(dst,0,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_res_impl_convection_const_part (const MatrixFree<dim,value_type>                       &data,
                                        parallel::distributed::BlockVector<value_type>         &dst,
                                        const parallel::distributed::BlockVector<value_type>   &src,
                                        const std::pair<unsigned int,unsigned int>             &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > sum_alphai_ui = fe_eval.get_value(q);

        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
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

        fe_eval.submit_value(sum_alphai_ui+rhs,q);
      }
      fe_eval.integrate (true,false);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_res_impl_convection (const MatrixFree<dim,value_type>                       &data,
                             parallel::distributed::BlockVector<value_type>         &dst,
                             const parallel::distributed::BlockVector<value_type>   &src,
                             const std::pair<unsigned int,unsigned int>             &cell_range) const
  {
    FEEval_Velocity_Velocity_nonlinear fe_eval(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval.get_value(q);
        Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,u);

        fe_eval.submit_gradient (F, q);
        fe_eval.submit_value(-gamma0/time_step*u,q);
      }
      fe_eval.integrate (true,true);
      fe_eval.distribute_local_to_global (dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_res_impl_convection_face (const MatrixFree<dim,value_type>                     &data,
                                  parallel::distributed::BlockVector<value_type>       &dst,
                                  const parallel::distributed::BlockVector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,fe_param,true,
       static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,fe_param,false,
       static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

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

        fe_eval.submit_value(-lf_flux,q);
        fe_eval_neighbor.submit_value(lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,false);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

   template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
   void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
   local_res_impl_convection_boundary_face (const MatrixFree<dim,value_type>                      &data,
                                            parallel::distributed::BlockVector<value_type>        &dst,
                                            const parallel::distributed::BlockVector<value_type>  &src,
                                            const std::pair<unsigned int,unsigned int>            &face_range) const
   {
     FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,fe_param,true,
         static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

     for(unsigned int face=face_range.first; face<face_range.second; face++)
     {
       fe_eval.reinit (face);
       fe_eval.read_dof_values(src);
       fe_eval.evaluate(true,false);

       for(unsigned int q=0;q<fe_eval.n_q_points;++q)
       {
         if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
         {
           // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
           Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);

           Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
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

           Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_np;
           Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
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

           fe_eval.submit_value(-lf_flux,q);
         }
         else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
         {
           // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
           Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
           Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
           const VectorizedArray<value_type> uM_n = uM*normal;

           Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = uM*uM_n;
           Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux;

           fe_eval.submit_value(-lf_flux,q);
         }
       }
       fe_eval.integrate(true,false);
       fe_eval.distribute_local_to_global(dst);
     }
   }

   template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
   void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
   local_apply_linearized_convection(const MatrixFree<dim,value_type>                 &data,
                                     parallel::distributed::BlockVector<double>       &dst,
                                     const parallel::distributed::BlockVector<double> &src,
                                     const std::pair<unsigned int,unsigned int>       &cell_range) const
   {
     FEEval_Velocity_Velocity_nonlinear fe_eval(data,fe_param,
         static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
     FEEval_Velocity_Velocity_nonlinear fe_eval_linear(data,fe_param,
         static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

     for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
     {
       fe_eval.reinit(cell);
       fe_eval.read_dof_values(src);
       fe_eval.evaluate (true,false,false);

       fe_eval_linear.reinit(cell);
       fe_eval_linear.read_dof_values(velocity_linear);
       fe_eval_linear.evaluate (true,false,false);

       for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
       {
         Tensor<1,dim,VectorizedArray<value_type> > delta_u = fe_eval.get_value(q);
         Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_linear.get_value(q);
         Tensor<2,dim,VectorizedArray<value_type> > F = outer_product(u,delta_u);
         fe_eval.submit_gradient (F+transpose(F), q);
         fe_eval.submit_value(-gamma0/time_step*delta_u,q);
       }
       fe_eval.integrate (true,true);
       fe_eval.distribute_local_to_global (dst);
     }
   }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_linearized_convection_face (const MatrixFree<dim,value_type>                 &data,
                                          parallel::distributed::BlockVector<double>       &dst,
                                          const parallel::distributed::BlockVector<double> &src,
                                          const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,fe_param,true,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_neighbor(data,fe_param,false,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linear(data,fe_param,true,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linear_neighbor(data,fe_param,false,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,false);

      fe_eval_linear.reinit(face);
      fe_eval_linear_neighbor.reinit (face);
      fe_eval_linear.read_dof_values(velocity_linear);
      fe_eval_linear.evaluate(true, false);
      fe_eval_linear_neighbor.read_dof_values(velocity_linear);
      fe_eval_linear_neighbor.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linear.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_linear_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

        const VectorizedArray<value_type> uM_n = uM*normal;
        const VectorizedArray<value_type> uP_n = uP*normal;

        // calculation of lambda according to Shahbazi et al., i.e.
        // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
        // where the maximum eigenvalue of the flux Jacobian is the
        // maximum eigenvalue of (u^T * normal) * I + u * normal^T, which is
        // abs(2*u^T*normal) (this can be verified by rank-1 matrix algebra)
        const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

        Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > delta_uP = fe_eval_neighbor.get_value(q);
        const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;
        const VectorizedArray<value_type> delta_uP_n = delta_uP*normal;
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = make_vectorized_array<value_type>(0.5)*
            (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval.submit_value(-lf_flux,q);
        fe_eval_neighbor.submit_value(lf_flux,q);
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,false);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_apply_linearized_convection_boundary_face(const MatrixFree<dim,value_type>                 &data,
                                                  parallel::distributed::BlockVector<double>       &dst,
                                                  const parallel::distributed::BlockVector<double> &src,
                                                  const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval(data,fe_param,true,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_linear(data,fe_param,true,
        static_cast <typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,false);

      fe_eval_linear.reinit (face);
      fe_eval_linear.read_dof_values(velocity_linear);
      fe_eval_linear.evaluate(true,false);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linear.get_value(q);

          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
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

          Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_np;
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;
          const VectorizedArray<value_type> uP_n = uP*normal;

          // calculation of lambda according to Shahbazi et al., i.e.
          // lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
          // where the maximum eigenvalue of the flux Jacobian is the
          // maximum eigenvalue of (u^T * normal) * I + u * normal^T, which is
          // abs(2*u^T*normal) (this can be verified by rank-1 matrix algebra)
          const VectorizedArray<value_type> lambda = 2.*std::max(std::abs(uM_n), std::abs(uP_n));

          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > delta_uP = -delta_uM;
          const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;
          const VectorizedArray<value_type> delta_uP_n = delta_uP*normal;

          Tensor<1,dim,VectorizedArray<value_type> > jump_value = delta_uM - delta_uP;
          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = make_vectorized_array<value_type>(0.5)*
              (uM*delta_uM_n + delta_uM*uM_n + uP*delta_uP_n + delta_uP*uP_n);
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

          fe_eval.submit_value(-lf_flux,q);
        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_linear.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
          const VectorizedArray<value_type> uM_n = uM*normal;

          Tensor<1,dim,VectorizedArray<value_type> > delta_uM = fe_eval.get_value(q);
          const VectorizedArray<value_type> delta_uM_n = delta_uM*normal;

          Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = (uM*delta_uM_n + delta_uM*uM_n);
          Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux;

          fe_eval.submit_value(-lf_flux,q);
        }
      }
      fe_eval.integrate(true,false);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_rhs (const MatrixFree<dim,value_type>                     &data,
                     parallel::distributed::BlockVector<value_type>       &dst,
                     const parallel::distributed::BlockVector<value_type> &,
                     const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
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
        fe_eval_velocity.submit_value (rhs, q);
      }
      fe_eval_velocity.integrate (true,false);
      fe_eval_velocity.distribute_local_to_global(dst,0,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous (const MatrixFree<dim,value_type>                      &data,
                     parallel::distributed::BlockVector<double>            &dst,
                     const parallel::distributed::BlockVector<value_type>  &src,
                     const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate (true,false,false);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_velocity.get_value(q);
        fe_eval_velocity.submit_value (make_vectorized_array<value_type>(gamma0/time_step)*u, q);
      }
      fe_eval_velocity.integrate (true,false);
      fe_eval_velocity.distribute_local_to_global (dst,0,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous_face (const MatrixFree<dim,value_type>                      &,
                          parallel::distributed::BlockVector<double>            &,
                          const parallel::distributed::BlockVector<value_type>  &,
                          const std::pair<unsigned int,unsigned int>            &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_viscous_boundary_face (const MatrixFree<dim,value_type>                     &data,
                                   parallel::distributed::BlockVector<double>           &dst,
                                   const parallel::distributed::BlockVector<value_type> &,
                                   const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,true,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);

      VectorizedArray<value_type> tau_IP = fe_eval_velocity.read_cell_data(helmholtz_operator.get_array_penalty_parameter())
                                             * helmholtz_operator.get_penalty_factor();

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        VectorizedArray<value_type> viscosity = make_vectorized_array<value_type>(helmholtz_operator.get_const_viscosity());
        if(helmholtz_operator.viscosity_is_variable())
          viscosity = helmholtz_operator.get_viscous_coefficient_face()[face][q];

        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
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

          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(2.*g_np,fe_eval_velocity.get_normal_vector(q));

          Tensor<2,dim,VectorizedArray<value_type> > average_gradient_tensor = -viscosity * tau_IP * jump_tensor;
          Tensor<1,dim,VectorizedArray<value_type> > average_gradient = average_gradient_tensor*fe_eval_velocity.get_normal_vector(q);

          if(param.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(param.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            else if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);

        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_velocity.quadrature_point(q);
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

          Tensor<2,dim,VectorizedArray<value_type> > jump_tensor
            = outer_product(jump_value,fe_eval_velocity.get_normal_vector(q));

          Tensor<1,dim,VectorizedArray<value_type> > average_gradient = -viscosity*h;

          if(param.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
          {
            if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              fe_eval_velocity.submit_gradient(fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              fe_eval_velocity.submit_gradient(-fe_eval_velocity.make_symmetric(viscosity*jump_tensor),q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else if(param.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
          {
            if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::NIPG)
              fe_eval_velocity.submit_gradient(0.5*viscosity*jump_tensor,q);
            else if(param.IP_formulation_viscous == InteriorPenaltyFormulationViscous::SIPG)
              fe_eval_velocity.submit_gradient(-0.5*viscosity*jump_tensor,q);
            else
              AssertThrow(false, ExcMessage("IP_FORMULATION_VISCOUS is not specified - possibilities are SIPG and NIPG"));
          }
          else
          {
            AssertThrow(false, ExcMessage("FORMULATION_VISCOUS_TERM is not specified - possibilities are DivergenceFormulation and LaplaceFormulation"));
          }
          fe_eval_velocity.submit_value(-average_gradient,q);
        }
      }

      fe_eval_velocity.integrate(true,true);
      fe_eval_velocity.distribute_local_to_global(dst,0,dim);
    }
  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  precompute_inverse_mass_matrix ()
//  {
//    std::vector<parallel::distributed::Vector<value_type> > dummy;
//    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_precompute_mass_matrix,
//                   this, dummy, dummy);
//  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  xwall_projection()
//  {
//
//    //make sure that this is distributed properly
//    xwall.ReturnTauWN().update_ghost_values();
//
//    std::vector<parallel::distributed::Vector<value_type> > tmp(2*dim);
//    for (unsigned int i=0;i<dim;i++)
//    {
//      tmp[i]=velocity_n[i];
//      tmp[i+dim]=velocity_n[i+dim];
//    }
//    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
//                   this, velocity_n, tmp);
//    for (unsigned int i=0;i<dim;i++)
//    {
//      tmp[i]=velocity_nm[i];
//      tmp[i+dim]=velocity_nm[i+dim];
//    }
//    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
//                   this, velocity_nm, tmp);
//    for (unsigned int i=0;i<dim;i++)
//    {
//      tmp[i]=velocity_nm2[i];
//      tmp[i+dim]=velocity_nm2[i+dim];
//    }
//    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
//                   this, velocity_nm2, tmp);
//  }

//  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_precompute_mass_matrix (const MatrixFree<dim,value_type>                              &data,
//                                std::vector<parallel::distributed::Vector<value_type> >       &,
//                                const std::vector<parallel::distributed::Vector<value_type> > &,
//                                const std::pair<unsigned int,unsigned int>                    &cell_range)
//  {
//   // initialize routine for non-enriched elements
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,fe_param.xwallstatevec[0],fe_param.xwallstatevec[1],0,3);
//
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//    //first, check if we have an enriched element
//    //if so, perform the routine for the enriched elements
//    fe_eval_xwall.reinit (cell);
//    if(fe_eval_xwall.enriched)
//    {
//      std::vector<FullMatrix<value_type> > matrix;
//      {
//        FullMatrix<value_type> onematrix(fe_eval_xwall.tensor_dofs_per_cell);
//        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//          matrix.push_back(onematrix);
//      }
//      for (unsigned int j=0; j<fe_eval_xwall.tensor_dofs_per_cell; ++j)
//      {
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
//        fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));
//
//        fe_eval_xwall.evaluate (true,false,false);
//        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
//        {
//  //        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
//          fe_eval_xwall.submit_value (fe_eval_xwall.get_value(q), q);
//        }
//        fe_eval_xwall.integrate (true,false);
//
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//            if(fe_eval_xwall.component_enriched(v))
//              (matrix[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
//            else//this is a non-enriched element
//            {
//              if(i<fe_eval_xwall.std_dofs_per_cell && j<fe_eval_xwall.std_dofs_per_cell)
//                (matrix[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
//              else if(i == j)//diagonal
//                (matrix[v])(i,j) = 1.0;
//            }
//      }
////      for (unsigned int i=0; i<10; ++i)
////        std::cout << std::endl;
////      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
////        matrix[v].print(std::cout,14,8);
//
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//      {
//        (matrix[v]).gauss_jordan();
//      }
//      matrices[cell].reinit(fe_eval_xwall.dofs_per_cell, fe_eval_xwall.dofs_per_cell);
//      //now apply vectors to inverse matrix
//      for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//        for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
//        {
//          VectorizedArray<value_type> value;
//          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//            value[v] = (matrix[v])(i,j);
//          matrices[cell](i,j) = value;
//        }
//    }
//  }
//  //
//
//
//  }

//  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_project_xwall (const MatrixFree<dim,value_type>        &data,
//      parallel::distributed::BlockVector<value_type>    &dst,
//      const std::vector<parallel::distributed::Vector<value_type> >  &src,
//               const std::pair<unsigned int,unsigned int>   &cell_range)
//  {
//    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall_n (data,fe_param.xwallstatevec[0],xwall.ReturnTauWN(),0,3);
//    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,fe_param.xwallstatevec[0],fe_param.xwallstatevec[1],0,3);
//
//    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//    {
//      //first, check if we have an enriched element
//      //if so, perform the routine for the enriched elements
//      fe_eval_xwall_n.reinit (cell);
//      fe_eval_xwall.reinit (cell);
//      if(fe_eval_xwall.enriched)
//      {
//        //now apply vectors to inverse matrix
//        for (unsigned int idim = 0; idim < dim; ++idim)
//        {
//          fe_eval_xwall_n.read_dof_values(src.at(idim),src.at(idim+dim));
//          fe_eval_xwall_n.evaluate(true,false);
//          for (unsigned int q=0; q<fe_eval_xwall.n_q_points; q++)
//            fe_eval_xwall.submit_value(fe_eval_xwall_n.get_value(q),q);
//          fe_eval_xwall.integrate(true,false);
//          AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
//          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//            for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
//              vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
//          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//            fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
//          fe_eval_xwall.set_dof_values (dst.block(idim),dst.block(idim+dim+1)); //TODO: Benjamin: check indices
//        }
//      }
//    }
//  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  apply_inverse_mass_matrix (const parallel::distributed::BlockVector<value_type>  &src,
//      parallel::distributed::BlockVector<value_type>      &dst) const
//  {
//    dst.zero_out_ghosts();
//
//    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_apply_mass_matrix,
//                   this, dst, src);
//  }

//  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_apply_mass_matrix (const MatrixFree<dim,value_type>        &data,
//                parallel::distributed::BlockVector<value_type>    &dst,
//                const parallel::distributed::BlockVector<value_type>  &src,
//                const std::pair<unsigned int,unsigned int>   &cell_range) const
//  {
//   InverseMassMatrixData<dim,fe_degree,value_type>& mass_data = mass_matrix_data->get();
//
//#ifdef XWALL
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,fe_param.xwallstatevec[0],fe_param.xwallstatevec[1],0,3);
//#endif
//
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//#ifdef XWALL
//    //first, check if we have an enriched element
//    //if so, perform the routine for the enriched elements
//    fe_eval_xwall.reinit (cell);
//    if(fe_eval_xwall.enriched)
//    {
//      //now apply vectors to inverse matrix
//      for (unsigned int idim = 0; idim < dim; ++idim)
//      {
//        fe_eval_xwall.read_dof_values(src.block(idim),src.block(idim+dim));
//        AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
//            vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
//        fe_eval_xwall.set_dof_values (dst.block(idim),dst.block(idim+dim));
//      }
//    }
//    else
//#endif
//    {
//      mass_data.fe_eval[0].reinit(cell);
//      mass_data.fe_eval[0].read_dof_values(src, 0);
//
//      mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
//      mass_data.inverse.apply(mass_data.coefficients, dim,
//                              mass_data.fe_eval[0].begin_dof_values(),
//                              mass_data.fe_eval[0].begin_dof_values());
//
//      mass_data.fe_eval[0].set_dof_values(dst,0);
//    }
//  }
//  }

//  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_apply_mass_matrix (const MatrixFree<dim,value_type>        &data,
//      std::vector<parallel::distributed::Vector<value_type> >    &dst,
//      const std::vector<parallel::distributed::Vector<value_type> >  &src,
//               const std::pair<unsigned int,unsigned int>   &cell_range) const
//  {
//    InverseMassMatrixData<dim,fe_degree,value_type>& mass_data = mass_matrix_data->get();
//    if(dst.size()>dim)
//    {
//
//#ifdef XWALL
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,fe_param.xwallstatevec[0],fe_param.xwallstatevec[1],0,3);
//#endif
//
//
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//#ifdef XWALL
//    //first, check if we have an enriched element
//    //if so, perform the routine for the enriched elements
//    fe_eval_xwall.reinit (cell);
//    if(fe_eval_xwall.enriched)
//    {
//      //now apply vectors to inverse matrix
//      for (unsigned int idim = 0; idim < dim; ++idim)
//      {
//        fe_eval_xwall.read_dof_values(src.at(idim),src.at(idim+dim));
//        AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
//            vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
//        for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//          fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
//        fe_eval_xwall.set_dof_values (dst.at(idim),dst.at(idim+dim));
//      }
//    }
//    else
//#endif
//    {
//      mass_data.fe_eval[0].reinit(cell);
//      mass_data.fe_eval[0].read_dof_values(src, 0);
//
//      mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
//      mass_data.inverse.apply(mass_data.coefficients, dim,
//                              mass_data.fe_eval[0].begin_dof_values(),
//                              mass_data.fe_eval[0].begin_dof_values());
//
//      mass_data.fe_eval[0].set_dof_values(dst,0);
//    }
//  }
//  //
//    }
//    else
//    {
//#ifdef XWALL
//     FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall (data,fe_param.xwallstatevec[0],fe_param.xwallstatevec[1],0,3);
//#endif
//
//    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//    {
//#ifdef XWALL
//      assert(false,ExcInternalError());
//      //first, check if we have an enriched element
//      //if so, perform the routine for the enriched elements
//      fe_eval_xwall.reinit (cell);
//      if(fe_eval_xwall.enriched)
//      {
//        //now apply vectors to inverse matrix
//          fe_eval_xwall.read_dof_values(src.at(0),src.at(1));
//          AlignedVector<VectorizedArray<value_type> > vector_result(fe_eval_xwall.dofs_per_cell);
//          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//            for (unsigned int j=0; j<fe_eval_xwall.dofs_per_cell; ++j)
//              vector_result[i] += matrices[cell](i,j) * fe_eval_xwall.read_cellwise_dof_value(j);
//          for (unsigned int i=0; i<fe_eval_xwall.dofs_per_cell; ++i)
//            fe_eval_xwall.write_cellwise_dof_value(i,vector_result[i]);
//          fe_eval_xwall.set_dof_values (dst.at(0),dst.at(1));
//      }
//      else
//  #endif
//      {
//        mass_data.fe_eval[0].reinit(cell);
//        mass_data.fe_eval[0].read_dof_values(src, 0);
//
//        mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
//        mass_data.inverse.apply(mass_data.coefficients, dim,
//                                mass_data.fe_eval[0].begin_dof_values(),
//                                mass_data.fe_eval[0].begin_dof_values());
//
//        mass_data.fe_eval[0].set_dof_values(dst,0);
//      }
//    }
//    }
//  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_divergence (const MatrixFree<dim,value_type>                      &data,
                            parallel::distributed::Vector<value_type>             &dst,
                            const parallel::distributed::BlockVector<value_type>  &src,
                            const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    FEEval_Velocity_scalar_Velocity_linear fe_eval_velocity_scalar(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type>
      phi(data,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity),
          static_cast<typename std::underlying_type_t<QuadratureSelector> >(QuadratureSelector::velocity));

    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity_scalar.reinit(cell);
      phi.reinit(cell);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate(false,true);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; q++)
        fe_eval_velocity_scalar.submit_value(fe_eval_velocity.get_divergence(q),q);
      fe_eval_velocity_scalar.integrate(true,false);
      for (unsigned int i=0; i<fe_eval_velocity_scalar.dofs_per_cell; i++)
        phi.begin_dof_values()[i] = fe_eval_velocity_scalar.begin_dof_values()[i];

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_vorticity (const parallel::distributed::BlockVector<value_type> &src,
                     parallel::distributed::BlockVector<value_type>       &dst) const
  {
    dst = 0;

    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_vorticity,this, dst, src);
  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_precompute_grad_div_projection(const MatrixFree<dim,value_type>                  &data,
//                std::vector<parallel::distributed::Vector<value_type> >      &,
//                const std::vector<parallel::distributed::Vector<value_type> >  &,
//                const std::pair<unsigned int,unsigned int>            &cell_range)
//  {
//  FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> phi(data,xwallstatevec[0],xwallstatevec[1],0,0);
//
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//    //first is div-div matrix, second is mass matrix
//    div_matrices[cell].resize(2);
//    //div-div matrix
//    phi.reinit(cell);
//    const unsigned int total_dofs_per_cell = phi.dofs_per_cell * dim;
//    div_matrices[cell][0].resize(data.n_components_filled(cell));
//    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//      div_matrices[cell][0][v].reinit(total_dofs_per_cell, total_dofs_per_cell);
//
//    for (unsigned int j=0; j<total_dofs_per_cell; ++j)
//    {
//      for (unsigned int i=0; i<total_dofs_per_cell; ++i)
//        phi.write_cellwise_dof_value(i,make_vectorized_array(0.));
//      phi.write_cellwise_dof_value(j,make_vectorized_array(1.));
//
//      phi.evaluate (false,true,false);
//      for (unsigned int q=0; q<phi.n_q_points; ++q)
//      {
//        const VectorizedArray<value_type> div = phi.get_divergence(q);
//        Tensor<2,dim,VectorizedArray<value_type> > test;
//        for (unsigned int d=0; d<dim; ++d)
//          test[d][d] = div;
//        phi.submit_gradient(test, q);
//      }
//      phi.integrate (false,true);
//
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
//          (div_matrices[cell][0][v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
//    }
//
//    //mass matrix
//    phi.reinit(cell);
//    div_matrices[cell][1].resize(data.n_components_filled(cell));
//    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//      div_matrices[cell][1][v].reinit(total_dofs_per_cell, total_dofs_per_cell);
//    for (unsigned int j=0; j<total_dofs_per_cell; ++j)
//    {
//      for (unsigned int i=0; i<total_dofs_per_cell; ++i)
//        phi.write_cellwise_dof_value(i,make_vectorized_array(0.));
//      phi.write_cellwise_dof_value(j,make_vectorized_array(1.));
//
//      phi.evaluate (true,false,false);
//      for (unsigned int q=0; q<phi.n_q_points; ++q)
//        phi.submit_value (phi.get_value(q), q);
//      phi.integrate (true,false);
//
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        if(phi.component_enriched(v))
//          for (unsigned int i=0; i<total_dofs_per_cell; ++i)
//            (div_matrices[cell][1][v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
//        else//this is a non-enriched element
//          {
//            if(j<phi.std_dofs_per_cell*dim)
//              for (unsigned int i=0; i<phi.std_dofs_per_cell*dim; ++i)
//                (div_matrices[cell][1][v])(i,j) = (phi.read_cellwise_dof_value(i))[v];
//            else //diagonal
//              (div_matrices[cell][1][v])(j,j) = 1.0;
//          }
//    }
//  }
//  }

//  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
//  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
//  local_compute_vorticity(const MatrixFree<dim,value_type>                  &data,
//                std::vector<parallel::distributed::Vector<value_type> >      &dst,
//                const std::vector<parallel::distributed::Vector<value_type> >  &src,
//                const std::pair<unsigned int,unsigned int>            &cell_range) const
//  {
////    //TODO Benjamin the vorticity lives only on the standard space
//////#ifdef XWALL
//////    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,3);
//////    FEEvaluation<dim,fe_degree,n_q_points_1d_xwall,number_vorticity_components,value_type> phi(data,0,3);
//////#else
//////    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
////    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,0);
//////    FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,number_vorticity_components,value_type> fe_eval_xwall_phi(data,src.at(dim),src.at(dim+1),0,0);
////    AlignedVector<VectorizedArray<value_type> > coefficients(phi.dofs_per_cell);
//
////
//#ifdef XWALL
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,number_vorticity_components,value_type> fe_eval_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
//   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> velocity_xwall(data,xwallstatevec[0],xwallstatevec[1],0,3);
//   FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
//   FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);
//#else
////   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,number_vorticity_components,value_type> fe_eval_xwall(data,src.at(2*dim+1),src.at(2*dim+2),0,0);
//   FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
//   FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);
//#endif
//  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);
//  const unsigned int dofs_per_cell = phi.dofs_per_cell;
//  AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
////    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);
//
////no XWALL but with XWALL routine
////   FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,1,value_type> fe_eval_xwall (data,src.at(dim+1),src.at(dim+2),0,0);
//
////   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_xwall (data,0,0);
//#ifdef XWALL
//  std::vector<LAPACKFullMatrix<value_type> > matrices(VectorizedArray<value_type>::n_array_elements);
//#endif
//  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//  {
//#ifdef XWALL
//    //first, check if we have an enriched element
//    //if so, perform the routine for the enriched elements
//    fe_eval_xwall.reinit (cell);
//    if(fe_eval_xwall.enriched)
//    {
//      const unsigned int total_dofs_per_cell = fe_eval_xwall.dofs_per_cell * number_vorticity_components;
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        matrices[v].reinit(total_dofs_per_cell);
//      velocity_xwall.reinit(cell);
//      velocity_xwall.read_dof_values(src,0,src,dim+1);
//      velocity_xwall.evaluate (false,true,false);
//
//      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
//      {
//        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
//          fe_eval_xwall.write_cellwise_dof_value(i,make_vectorized_array(0.));
//        fe_eval_xwall.write_cellwise_dof_value(j,make_vectorized_array(1.));
//
//        fe_eval_xwall.evaluate (true,false,false);
//        for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
//        {
//  //        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
//          fe_eval_xwall.submit_value (fe_eval_xwall.get_value(q), q);
//        }
//        fe_eval_xwall.integrate (true,false);
//
//        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
//          for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//            if(fe_eval_xwall.component_enriched(v))
//              (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
//            else//this is a non-enriched element
//            {
//              if(i<fe_eval_xwall.std_dofs_per_cell*number_vorticity_components && j<fe_eval_xwall.std_dofs_per_cell*number_vorticity_components)
//                (matrices[v])(i,j) = (fe_eval_xwall.read_cellwise_dof_value(i))[v];
//              else if(i == j)//diagonal
//                (matrices[v])(i,j) = 1.0;
//            }
//      }
////      for (unsigned int i=0; i<10; ++i)
////        std::cout << std::endl;
////      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
////        matrices[v].print(std::cout,14,8);
//
//      //initialize again to get a clean version
//      fe_eval_xwall.reinit (cell);
//      //now apply vectors to inverse matrix
//      for (unsigned int q=0; q<fe_eval_xwall.n_q_points; ++q)
//      {
//        fe_eval_xwall.submit_value (velocity_xwall.get_curl(q), q);
////        std::cout << velocity_xwall.get_curl(q)[2][0] << "   "  << velocity_xwall.get_curl(q)[2][1] << std::endl;
//      }
//      fe_eval_xwall.integrate (true,false);
//
//
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//      {
//        (matrices[v]).compute_lu_factorization();
//        Vector<value_type> vector_input(total_dofs_per_cell);
//        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
//          vector_input(j)=(fe_eval_xwall.read_cellwise_dof_value(j))[v];
//
//  //        Vector<value_type> vector_result(total_dofs_per_cell);
//        (matrices[v]).apply_lu_factorization(vector_input,false);
//  //        (matrices[v]).vmult(vector_result,vector_input);
//        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
//          fe_eval_xwall.write_cellwise_dof_value(j,vector_input(j),v);
//      }
//      fe_eval_xwall.set_dof_values (dst,0,dst,number_vorticity_components);
//
//    }
//    else
//#endif
//    {
//      phi.reinit(cell);
//      velocity.reinit(cell);
//      velocity.read_dof_values(src,0);
//      velocity.evaluate (false,true,false);
//      for (unsigned int q=0; q<phi.n_q_points; ++q)
//      {
//      Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
////      std::cout << omega[2][0] << "    " << omega[2][1] << std::endl;
//        phi.submit_value (omega, q);
//      }
//      phi.integrate (true,false);
//
//      inverse.fill_inverse_JxW_values(coefficients);
//      inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());
//
//      phi.set_dof_values(dst,0);
//    }
//  }
//
////    else
//
////    {
////      phi.read_dof_values(src,0);
////
////      inverse.fill_inverse_JxW_values(coefficients);
////      inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());
////
////      phi.set_dof_values(dst,0);
////    }
////  }
//
//  //
//  }


  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_compute_vorticity(const MatrixFree<dim,value_type>                      &data,
                          parallel::distributed::BlockVector<value_type>        &dst,
                          const parallel::distributed::BlockVector<value_type>  &src,
                          const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    //TODO: Benjamin: implement XWall
    FEEval_Velocity_Velocity_linear velocity(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEEval_Vorticity_Velocity_linear phi(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);
    const unsigned int dofs_per_cell = phi.dofs_per_cell;
    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit(cell);
      velocity.reinit(cell);
      velocity.read_dof_values(src,0);
      velocity.evaluate (false,true,false);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
        Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
        phi.submit_value (omega, q);

      }
      phi.integrate (true,false);

      inverse.fill_inverse_JxW_values(coefficients);
      inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());

      phi.set_dof_values(dst,0);
    }
  }

  template <int dim, typename FEEval>
  struct CurlCompute
  {
    static Tensor<1,dim,VectorizedArray<typename FEEval::number_type> > compute(FEEval &fe_eval,const unsigned int q_point)
    {
      return fe_eval.get_curl(q_point);
    }
  };

  template <typename FEEval>
  struct CurlCompute<2,FEEval>
  {
    static Tensor<1,2,VectorizedArray<typename FEEval::number_type> > compute(FEEval &fe_eval, const unsigned int q_point)
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
  shift_pressure (parallel::distributed::Vector<value_type>  &pressure) const
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
  unsigned int NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  solve_pressure (parallel::distributed::Vector<value_type>        &dst,
                  const parallel::distributed::Vector<value_type>  &src) const
  {
    unsigned int n_iter = pressure_poisson_solver.solve(dst,src);

    return n_iter;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  apply_nullspace_projection (parallel::distributed::Vector<value_type>  &dst) const
  {
    pressure_poisson_solver.get_matrix().apply_nullspace_projection(dst);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_pressure_divergence_term (const parallel::distributed::BlockVector<value_type>  &src,
                                parallel::distributed::Vector<value_type>             &dst) const
  {
    data.loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_divergence_term,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_divergence_term_face,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_divergence_term_boundary_face,
               this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_divergence_term (const MatrixFree<dim,value_type>                      &data,
                                      parallel::distributed::Vector<value_type>             &dst,
                                      const parallel::distributed::BlockVector<value_type>  &src,
                                      const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEEval_Pressure_Velocity_linear fe_eval_pressure(data,fe_param,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_pressure.reinit (cell);

      fe_eval_velocity.reinit (cell);
      fe_eval_velocity.read_dof_values(src,0,dim);

      if(param.divu_integrated_by_parts == true)
      {
        fe_eval_velocity.evaluate (true,false,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          fe_eval_pressure.submit_gradient (fe_eval_velocity.get_value(q)*gamma0/time_step, q);
        }
        fe_eval_pressure.integrate (false,true);
      }
      else
      {
        fe_eval_velocity.evaluate (false,true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          if(param.small_time_steps_stability == true)
            fe_eval_pressure.submit_value(-fe_eval_velocity.get_divergence(q),q);
          else
            fe_eval_pressure.submit_value (-fe_eval_velocity.get_divergence(q)*gamma0/time_step, q);
        }
        fe_eval_pressure.integrate (true,false);
      }
      fe_eval_pressure.distribute_local_to_global (dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_divergence_term_face (const MatrixFree<dim,value_type>                     &data,
                                           parallel::distributed::Vector<value_type>            &dst,
                                           const parallel::distributed::BlockVector<value_type> &src,
                                           const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    if(param.divu_integrated_by_parts == true)
    {
      if(param.small_time_steps_stability == true)
        AssertThrow(false,ExcMessage("Using small time steps stability method in combination with integration by parts of divergence term on rhs of PPE does not make sense."));

      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,fe_param,false,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure_neighbor(data,fe_param,false,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_pressure.reinit (face);
        fe_eval_pressure_neighbor.reinit (face);

        fe_eval_velocity.reinit (face);
        fe_eval_velocity_neighbor.reinit (face);
        fe_eval_velocity.read_dof_values(src,0,dim);
        fe_eval_velocity_neighbor.read_dof_values(src,0,dim);
        fe_eval_velocity.evaluate (true,false);
        fe_eval_velocity_neighbor.evaluate (true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_velocity.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > meanvel = 0.5*(fe_eval_velocity.get_value(q)+fe_eval_velocity_neighbor.get_value(q));
          VectorizedArray<value_type> submitvalue = normal * meanvel;

          fe_eval_pressure.submit_value ((-submitvalue)*gamma0/time_step, q);
          fe_eval_pressure_neighbor.submit_value (submitvalue*gamma0/time_step, q);
        }
        fe_eval_pressure.integrate (true,false);
        fe_eval_pressure_neighbor.integrate (true,false);
        fe_eval_pressure.distribute_local_to_global (dst);
        fe_eval_pressure_neighbor.distribute_local_to_global (dst);
      }
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_divergence_term_boundary_face (const MatrixFree<dim,value_type>                      &data,
                                                    parallel::distributed::Vector<value_type>             &dst,
                                                    const parallel::distributed::BlockVector<value_type>  &src,
                                                    const std::pair<unsigned int,unsigned int>            &face_range) const
  {
    if(param.divu_integrated_by_parts == true)
    {
      if(param.small_time_steps_stability == true)
        AssertThrow(false,ExcMessage("Using small time steps stability method in combination with integration by parts of divergence term on rhs of PPE does not make sense."));

      FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

      FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

      //TODO: quadrature formula
//      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,true,
//          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
//
//      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,fe_param,true,
//          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_pressure.reinit (face);

        fe_eval_velocity.reinit(face);
        fe_eval_velocity.read_dof_values(src,0,dim);
        fe_eval_velocity.evaluate (true,false);

        for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > meanvel = fe_eval_velocity.get_value(q);
          VectorizedArray<value_type> submitvalue = normal * meanvel;

          fe_eval_pressure.submit_value((-submitvalue)*gamma0/time_step,q);
        }
        fe_eval_pressure.integrate(true,false);
        fe_eval_pressure.distribute_local_to_global(dst);
      }
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_pressure_BC_term (const parallel::distributed::BlockVector<value_type>  &src,
                        parallel::distributed::Vector<value_type>             &dst) const
  {
    data.loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term_face,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_BC_term_boundary_face,
               this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_BC_term (const MatrixFree<dim,value_type>                      &,
                              parallel::distributed::Vector<value_type>             &,
                              const parallel::distributed::BlockVector<value_type>  &,
                              const std::pair<unsigned int,unsigned int>            &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_BC_term_face (const MatrixFree<dim,value_type>                     &,
                                   parallel::distributed::Vector<value_type>            &,
                                   const parallel::distributed::BlockVector<value_type> &,
                                   const std::pair<unsigned int,unsigned int>           &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_BC_term_boundary_face (const MatrixFree<dim,value_type>                      &data,
                                            parallel::distributed::Vector<value_type>             &dst,
                                            const parallel::distributed::BlockVector<value_type>  &,
                                            const std::pair<unsigned int,unsigned int>            &face_range) const
  {
    FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,fe_param,true,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

    //TODO: quadrature formula
//    FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,fe_param,true,
//        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_pressure.reinit (face);

      double factor = pressure_poisson_solver.get_matrix().get_penalty_factor();
      VectorizedArray<value_type> tau_IP = fe_eval_pressure.read_cell_data(pressure_poisson_solver.get_matrix().get_array_penalty_parameter()) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
      {
        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end())
        {
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);

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

          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);
          VectorizedArray<value_type> h;

          h = - normal * (dudt_np - rhs_np);

          fe_eval_pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
          fe_eval_pressure.submit_value(h,q);
        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end())
        {
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
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

          fe_eval_pressure.submit_normal_gradient(-g,q);
          fe_eval_pressure.submit_value(2.0 * tau_IP * g,q);
        }
      }
      fe_eval_pressure.integrate(true,true);
      fe_eval_pressure.distribute_local_to_global(dst);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_pressure_convective_term (const parallel::distributed::BlockVector<value_type>  &src,
                                parallel::distributed::Vector<value_type>             &dst) const
  {
    data.loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term_face,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_convective_term_boundary_face,
               this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_convective_term (const MatrixFree<dim,value_type>                     &,
                                      parallel::distributed::Vector<value_type>            &,
                                      const parallel::distributed::BlockVector<value_type> &,
                                      const std::pair<unsigned int,unsigned int>           &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_convective_term_face (const MatrixFree<dim,value_type>                     &,
                                           parallel::distributed::Vector<value_type>            &,
                                           const parallel::distributed::BlockVector<value_type> &,
                                           const std::pair<unsigned int,unsigned int>           &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_convective_term_boundary_face (const MatrixFree<dim,value_type>                      &data,
                                                    parallel::distributed::Vector<value_type>             &dst,
                                                    const parallel::distributed::BlockVector<value_type>  &src,
                                                    const std::pair<unsigned int,unsigned int>            &face_range) const
  {

    FEFaceEval_Velocity_Velocity_nonlinear fe_eval_velocity(data,fe_param,true,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,fe_param,true,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit(face);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate (true,true);

      fe_eval_pressure.reinit (face);

      for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
      {
        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end())
        {
          VectorizedArray<value_type> h;
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

          Tensor<1,dim,VectorizedArray<value_type> > u = fe_eval_velocity.get_value(q);
          Tensor<2,dim,VectorizedArray<value_type> > grad_u = fe_eval_velocity.get_gradient(q);
          Tensor<1,dim,VectorizedArray<value_type> > convective_term = grad_u * u + fe_eval_velocity.get_divergence(q) * u;

          h = - normal * convective_term;

          fe_eval_pressure.submit_value(h,q);
        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end())
        {
          fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
        }
      }
      fe_eval_pressure.integrate(true,false);
      fe_eval_pressure.distribute_local_to_global(dst);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_pressure_viscous_term (const parallel::distributed::BlockVector<value_type> &src,
                             parallel::distributed::Vector<value_type>            &dst) const
  {
    data.loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term_face,
               &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_pressure_viscous_term_boundary_face,
               this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_viscous_term (const MatrixFree<dim,value_type>                     &,
                                   parallel::distributed::Vector<value_type>            &,
                                   const parallel::distributed::BlockVector<value_type> &,
                                   const std::pair<unsigned int,unsigned int>           &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_viscous_term_face (const MatrixFree<dim,value_type>                      &,
                                        parallel::distributed::Vector<value_type>             &,
                                        const parallel::distributed::BlockVector<value_type>  &,
                                        const std::pair<unsigned int,unsigned int>            &) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_pressure_viscous_term_boundary_face (const MatrixFree<dim,value_type>                     &data,
                                                 parallel::distributed::Vector<value_type>            &dst,
                                                 const parallel::distributed::BlockVector<value_type> &src,
                                                 const std::pair<unsigned int,unsigned int>           &face_range) const
  {
    FEFaceEval_Vorticity_Velocity_nonlinear fe_eval_omega(data,fe_param,true,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    FEFaceEval_Pressure_Velocity_nonlinear fe_eval_pressure(data,fe_param,true,
        static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_pressure.reinit (face);

      fe_eval_omega.reinit (face);
      fe_eval_omega.read_dof_values(src,0,number_vorticity_components);
      fe_eval_omega.evaluate (false,true);

      for(unsigned int q=0;q<fe_eval_pressure.n_q_points;++q)
      {
        VectorizedArray<value_type> viscosity;
        if(helmholtz_operator.viscosity_is_variable())
          viscosity = helmholtz_operator.get_viscous_coefficient_face()[face][q];
        else
          viscosity = make_vectorized_array<value_type>(helmholtz_operator.get_const_viscosity());

        if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end())
        {
          VectorizedArray<value_type> h;
          Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval_pressure.get_normal_vector(q);

          Tensor<1,dim,VectorizedArray<value_type> > rot_omega = CurlCompute<dim,FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_nonlinear,number_vorticity_components,value_type,is_xwall> >::compute(fe_eval_omega,q);

          h = - normal * (viscosity*rot_omega);

          fe_eval_pressure.submit_value(h,q);
        }
        else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end())
        {
          fe_eval_pressure.submit_value(make_vectorized_array<value_type>(0.0),q);
        }
      }
      fe_eval_pressure.integrate(true,false);
      fe_eval_pressure.distribute_local_to_global(dst);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  unsigned int NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  solve_projection (parallel::distributed::BlockVector<value_type> &dst,
                    const parallel::distributed::BlockVector<value_type> &src,
                    const parallel::distributed::BlockVector<value_type> &velocity_n,
                    double const cfl) const
  {
    projection_solver->calculate_array_penalty_parameter(velocity_n,cfl,time_step);
    unsigned int n_iter = projection_solver->solve(dst,src);

    return n_iter;
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  rhs_projection (const parallel::distributed::BlockVector<value_type>     &src_velocity,
                  const parallel::distributed::Vector<value_type> &src_pressure,
                  parallel::distributed::BlockVector<value_type>      &dst) const
  {
    // set dst-vector to zero
    dst = 0;

    // compute mass matrix term on rhs of projection step
    data.cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_mass_term, this, dst, src_velocity);

    // do not reset dst-vector!

    //compute gradient(p) term on rhs of projection step
    data.loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_gradient_term,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_gradient_term_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_projection_gradient_term_boundary_face,
              this, dst, src_pressure);
  }

  template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  compute_divergence (const parallel::distributed::BlockVector<value_type>     &src,
                      parallel::distributed::Vector<value_type>     &dst,
                      const bool apply_inv_mass_matrix) const
  {

    parallel::distributed::BlockVector<value_type> test(src);

    if(apply_inv_mass_matrix)
    {
      inverse_mass_matrix_operator.apply_inverse_mass_matrix(src,test);
    }

    data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_compute_divergence,
                               this, dst, test);
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_projection_mass_term (const MatrixFree<dim,value_type>                     &data,
                                  parallel::distributed::BlockVector<value_type>       &dst,
                                  const parallel::distributed::BlockVector<value_type> &src,
                                  const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
        static_cast< typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate(true,false);

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        fe_eval_velocity.submit_value(fe_eval_velocity.get_value(q),q);
      }
      fe_eval_velocity.integrate (true,false);

      fe_eval_velocity.distribute_local_to_global (dst,0,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_projection_gradient_term (const MatrixFree<dim,value_type>                 &data,
                                      parallel::distributed::BlockVector<value_type>   &dst,
                                      const parallel::distributed::Vector<value_type>  &src,
                                      const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,
        static_cast< typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
    FEEval_Pressure_Velocity_linear fe_eval_pressure(data,fe_param,
        static_cast< typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

    const VectorizedArray<value_type> fac = make_vectorized_array(time_step/gamma0);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit (cell);
      fe_eval_pressure.reinit (cell);
      fe_eval_pressure.read_dof_values(src);

      if(param.gradp_integrated_by_parts == true)
      {
        fe_eval_pressure.evaluate (true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<value_type> > test;
          for (unsigned int a=0;a<dim;++a)
            test[a][a] = fac;

          test *= fe_eval_pressure.get_value(q);
          fe_eval_velocity.submit_gradient (test, q);
        }
        fe_eval_velocity.integrate (false,true);
      }
      else
      {
        fe_eval_pressure.evaluate (false,true);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          fe_eval_velocity.submit_value(-fac*fe_eval_pressure.get_gradient(q),q);
        }
        fe_eval_velocity.integrate (true,false);
      }
      fe_eval_velocity.distribute_local_to_global (dst,0,dim);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_projection_gradient_term_face (const MatrixFree<dim,value_type>                 &data,
                                           parallel::distributed::BlockVector<value_type>   &dst,
                                           const parallel::distributed::Vector<value_type>  &src,
                                           const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    if(param.gradp_integrated_by_parts==true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,fe_param,false,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));

      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure_neighbor(data,fe_param,false,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

      const VectorizedArray<value_type> fac = make_vectorized_array(time_step/gamma0);
      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_velocity.reinit (face);
        fe_eval_velocity_neighbor.reinit (face);
        fe_eval_pressure.reinit (face);
        fe_eval_pressure_neighbor.reinit (face);
        fe_eval_pressure.read_dof_values(src);
        fe_eval_pressure_neighbor.read_dof_values(src);
        fe_eval_pressure.evaluate (true,false);
        fe_eval_pressure_neighbor.evaluate (true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > normal = fac*fe_eval_pressure.get_normal_vector(q);
          VectorizedArray<value_type> meanpres = 0.5*(fe_eval_pressure.get_value(q)+fe_eval_pressure_neighbor.get_value(q));

          normal*=meanpres;

          fe_eval_velocity.submit_value (-normal, q);
          fe_eval_velocity_neighbor.submit_value (normal, q);
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity_neighbor.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst,0,dim);
        fe_eval_velocity_neighbor.distribute_local_to_global (dst,0,dim);
      }
    }
  }

  template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
  local_rhs_projection_gradient_term_boundary_face (const MatrixFree<dim,value_type>                &data,
                                                    parallel::distributed::BlockVector<value_type>  &dst,
                                                    const parallel::distributed::Vector<value_type> &src,
                                                    const std::pair<unsigned int,unsigned int>      &face_range) const
  {
    if(param.gradp_integrated_by_parts==true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::velocity));
      FEFaceEval_Pressure_Velocity_linear fe_eval_pressure(data,fe_param,true,
          static_cast<typename std::underlying_type_t<DofHandlerSelector> >(DofHandlerSelector::pressure));

      const VectorizedArray<value_type> fac = make_vectorized_array(time_step/gamma0);
      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval_velocity.reinit (face);
        fe_eval_pressure.reinit (face);
        fe_eval_pressure.read_dof_values(src);
        fe_eval_pressure.evaluate (true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          //TODO
//          if (dirichlet_boundary.find(data.get_boundary_indicator(face)) != dirichlet_boundary.end()) // Inflow and wall boundaries
//          {
//            // p+ =  p-
//            Tensor<1,dim,VectorizedArray<value_type> > normal = fac*fe_eval_pressure.get_normal_vector(q);
//            VectorizedArray<value_type> meanpres = fe_eval_pressure.get_value(q);
//            normal*=meanpres;
//            fe_eval_velocity.submit_value (-normal, q);
//          }
//          else if (neumann_boundary.find(data.get_boundary_indicator(face)) != neumann_boundary.end()) // Inflow and wall boundaries
//          {
//            // p+ = - p- + 2g
//            Point<dim,VectorizedArray<value_type> > q_points = fe_eval_pressure.quadrature_point(q);
//            VectorizedArray<value_type> g;
//            AnalyticalSolution<dim> dirichlet_boundary(dim,time+time_step);
//            value_type array [VectorizedArray<value_type>::n_array_elements];
//            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//            {
//              Point<dim> q_point;
//              for (unsigned int d=0; d<dim; ++d)
//                q_point[d] = q_points[d][n];
//              array[n] = dirichlet_boundary.value(q_point);
//            }
//            g.load(&array[0]);
//            Tensor<1,dim,VectorizedArray<value_type> > normal = fac*fe_eval_pressure.get_normal_vector(q);
//            normal *= g;
//            fe_eval_velocity.submit_value (-normal, q);
//          }
          //TODO

          //TODO
          Tensor<1,dim,VectorizedArray<value_type> > normal = fac*fe_eval_pressure.get_normal_vector(q);
          VectorizedArray<value_type> meanpres = fe_eval_pressure.get_value(q);
          normal*=meanpres;
          fe_eval_velocity.submit_value (-normal, q);
          //TODO
        }
        fe_eval_velocity.integrate (true,false);
        fe_eval_velocity.distribute_local_to_global (dst,0,dim);
      }
    }
}

#endif /* INCLUDE_DGNAVIERSTOKESDUALSPLITTING_H_ */

