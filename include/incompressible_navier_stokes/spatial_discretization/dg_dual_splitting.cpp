/*
 * dg_dual_splitting.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "dg_dual_splitting.h"

#include "../../functionalities/moving_mesh.h"

namespace IncNS
{
template<int dim, typename Number>
DGNavierStokesDualSplitting<dim, Number>::DGNavierStokesDualSplitting(
  parallel::TriangulationBase<dim> const & triangulation_in,
  std::shared_ptr<Mesh<dim>> const         mesh_in,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> const
                                                  periodic_face_pairs_in,
  std::shared_ptr<BoundaryDescriptorU<dim>> const boundary_descriptor_velocity_in,
  std::shared_ptr<BoundaryDescriptorP<dim>> const boundary_descriptor_pressure_in,
  std::shared_ptr<FieldFunctions<dim>> const      field_functions_in,
  InputParameters const &                         parameters_in,
  MPI_Comm const &                                mpi_comm_in)
  : ProjBase(triangulation_in,
             mesh_in,
             periodic_face_pairs_in,
             boundary_descriptor_velocity_in,
             boundary_descriptor_pressure_in,
             field_functions_in,
             parameters_in,
             mpi_comm_in)
{
}

template<int dim, typename Number>
DGNavierStokesDualSplitting<dim, Number>::~DGNavierStokesDualSplitting()
{
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::setup_solvers(
  double const &     scaling_factor_time_derivative_term,
  VectorType const & velocity)
{
  this->pcout << std::endl << "Setup solvers ..." << std::endl;

  ProjBase::setup_solvers(scaling_factor_time_derivative_term, velocity);

  ProjBase::setup_pressure_poisson_solver();

  ProjBase::setup_projection_solver();

  setup_helmholtz_solver();

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::setup_helmholtz_solver()
{
  initialize_helmholtz_preconditioner();

  initialize_helmholtz_solver();
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::initialize_helmholtz_preconditioner()
{
  if(this->param.preconditioner_viscous == PreconditionerViscous::None)
  {
    // do nothing, preconditioner will not be used
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
  {
    helmholtz_preconditioner.reset(new InverseMassMatrixPreconditioner<dim, dim, Number>(
      this->get_matrix_free(),
      this->param.degree_u,
      this->get_dof_index_velocity(),
      this->get_quad_index_velocity_linear()));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi)
  {
    helmholtz_preconditioner.reset(
      new JacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi)
  {
    helmholtz_preconditioner.reset(
      new BlockJacobiPreconditioner<MomentumOperator<dim, Number>>(this->momentum_operator));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
  {
    typedef MultigridPreconditioner<dim, Number, MultigridNumber> MULTIGRID;

    helmholtz_preconditioner.reset(new MULTIGRID(this->mpi_comm));

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(helmholtz_preconditioner);

    auto & dof_handler = this->get_dof_handler_u();

    parallel::TriangulationBase<dim> const * tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&dof_handler.get_triangulation());

    const FiniteElement<dim> & fe = dof_handler.get_fe();

    mg_preconditioner->initialize(this->param.multigrid_data_viscous,
                                  tria,
                                  fe,
                                  this->get_mapping(),
                                  this->momentum_operator,
                                  MultigridOperatorType::ReactionDiffusion,
                                  this->param.ale_formulation,
                                  &this->momentum_operator.get_data().bc->dirichlet_bc,
                                  &this->periodic_face_pairs);
  }
  else
  {
    AssertThrow(false, ExcMessage("Preconditioner specified for viscous step is not implemented."));
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::initialize_helmholtz_solver()
{
  if(this->param.solver_viscous == SolverViscous::CG)
  {
    // setup solver data
    CGSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_viscous.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_viscous.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_viscous.rel_tol;

    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
    {
      solver_data.use_preconditioner = true;
    }

    helmholtz_solver.reset(
      new CGSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        this->momentum_operator, *helmholtz_preconditioner, solver_data));
  }
  else if(this->param.solver_viscous == SolverViscous::GMRES)
  {
    // setup solver data
    GMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_viscous.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_viscous.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_viscous.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_viscous.max_krylov_size;
    // use default value of compute_eigenvalues

    // default value of use_preconditioner = false
    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
    {
      solver_data.use_preconditioner = true;
    }

    helmholtz_solver.reset(
      new GMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        this->momentum_operator, *helmholtz_preconditioner, solver_data, this->mpi_comm));
  }
  else if(this->param.solver_viscous == SolverViscous::FGMRES)
  {
    FGMRESSolverData solver_data;
    solver_data.max_iter             = this->param.solver_data_viscous.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_viscous.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_viscous.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_viscous.max_krylov_size;

    if(this->param.preconditioner_viscous == PreconditionerViscous::PointJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::BlockJacobi ||
       this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix ||
       this->param.preconditioner_viscous == PreconditionerViscous::Multigrid)
    {
      solver_data.use_preconditioner = true;
    }

    helmholtz_solver.reset(
      new FGMRESSolver<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>(
        this->momentum_operator, *helmholtz_preconditioner, solver_data));
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified viscous solver is not implemented."));
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::apply_velocity_divergence_term(
  VectorType &       dst,
  VectorType const & src) const
{
  this->divergence_operator.apply(dst, src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_velocity_divergence_term(
  VectorType &   dst,
  double const & evaluation_time) const
{
  this->divergence_operator.rhs(dst, evaluation_time);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_div_term_body_forces_add(VectorType &   dst,
                                                                           double const & time)
{
  this->evaluation_time = time;

  VectorType src_dummy;
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_div_term_body_forces_boundary_face,
                               this,
                               dst,
                               src_dummy);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_div_term_body_forces_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(matrix_free, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);

    BoundaryTypeU boundary_type =
      this->boundary_descriptor_velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate right-hand side
        vector rhs = evaluate_vectorial_function(this->field_functions->right_hand_side,
                                                 q_points,
                                                 this->evaluation_time);

        scalar flux_times_normal = rhs * integrator.get_normal_vector(q);
        // minus sign is introduced here which allows to call a function of type ...add()
        // and avoids a scaling of the resulting vector by the factor -1.0
        integrator.submit_value(-flux_times_normal, q);
      }
      else if(boundary_type == BoundaryTypeU::Neumann || boundary_type == BoundaryTypeU::Symmetry)
      {
        // Do nothing on Neumann and Symmetry boundaries.
        // Remark: on symmetry boundaries we prescribe g_u * n = 0, and also g_{u_hat}*n = 0 in case
        // of the dual splitting scheme. This is in contrast to Dirichlet boundaries where we
        // prescribe a consistent boundary condition for g_{u_hat} derived from the convective step
        // of the dual splitting scheme which differs from the DBC g_u. Applying this consistent DBC
        // to symmetry boundaries and using g_u*n=0 as well as exploiting symmetry, we obtain
        // g_{u_hat}*n=0 on symmetry boundaries. Hence, there are no inhomogeneous contributions for
        // g_{u_hat}*n.
        scalar zero = make_vectorized_array<Number>(0.0);
        integrator.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    integrator.integrate(true, false);
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_velocity_divergence_term_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & velocity) const
{
  this->divergence_operator.rhs_bc_from_dof_vector(dst, velocity);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_div_term_convective_term_add(
  VectorType &       dst,
  VectorType const & src) const
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_div_term_convective_term_boundary_face,
                               this,
                               dst,
                               src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_div_term_convective_term_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_dof_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_nonlinear();

  FaceIntegratorU velocity(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);

  FaceIntegratorU grid_velocity(matrix_free, true, dof_index_velocity, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    velocity.reinit(face);
    velocity.gather_evaluate(src, true, true);

    if(this->param.ale_formulation && this->param.store_previous_boundary_values)
    {
      grid_velocity.reinit(face);
      grid_velocity.gather_evaluate(this->convective_kernel->get_grid_velocity(), true, false);
    }

    pressure.reinit(face);

    BoundaryTypeU boundary_type =
      this->boundary_descriptor_velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet)
      {
        vector normal = pressure.get_normal_vector(q);

        vector u      = velocity.get_value(q);
        tensor grad_u = velocity.get_gradient(q);

        vector flux;
        if(this->param.formulation_convective_term_bc ==
           FormulationConvectiveTerm::DivergenceFormulation)
        {
          scalar div_u = velocity.get_divergence(q);
          flux         = grad_u * u + div_u * u;
        }
        else if(this->param.formulation_convective_term_bc ==
                FormulationConvectiveTerm::ConvectiveFormulation)
        {
          flux = grad_u * u;
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }

        if(this->param.ale_formulation && this->param.store_previous_boundary_values)
        {
          flux -= grad_u * grid_velocity.get_value(q);
        }

        scalar flux_times_normal = flux * normal;

        pressure.submit_value(flux_times_normal, q);
      }
      else if(boundary_type == BoundaryTypeU::Neumann || boundary_type == BoundaryTypeU::Symmetry)
      {
        // Do nothing on Neumann and Symmetry boundaries.
        // Remark: on symmetry boundaries we prescribe g_u * n = 0, and also g_{u_hat}*n = 0 in
        // case of the dual splitting scheme. This is in contrast to Dirichlet boundaries where we
        // prescribe a consistent boundary condition for g_{u_hat} derived from the convective
        // step of the dual splitting scheme which differs from the DBC g_u. Applying this
        // consistent DBC to symmetry boundaries and using g_u*n=0 as well as exploiting symmetry,
        // we obtain g_{u_hat}*n=0 on symmetry boundaries. Hence, there are no inhomogeneous
        // contributions for g_{u_hat}*n.
        scalar zero = make_vectorized_array<Number>(0.0);
        pressure.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    pressure.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_nbc_analytical_time_derivative_add(
  VectorType &   dst,
  double const & time)
{
  this->evaluation_time = time;

  VectorType src_dummy;
  this->get_matrix_free().loop(
    &This::cell_loop_empty,
    &This::face_loop_empty,
    &This::local_rhs_ppe_nbc_analytical_time_derivative_add_boundary_face,
    this,
    dst,
    src_dummy);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::
  local_rhs_ppe_nbc_analytical_time_derivative_add_boundary_face(
    MatrixFree<dim, Number> const & data,
    VectorType &                    dst,
    VectorType const &,
    Range const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(data, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);

    types::boundary_id boundary_id = data.get_boundary_id(face);
    BoundaryTypeP      boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate boundary condition
        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          this->boundary_descriptor_pressure->neumann_bc.find(boundary_id);
        vector dudt = evaluate_vectorial_function(it->second, q_points, this->evaluation_time);

        vector normal = integrator.get_normal_vector(q);

        scalar h = -normal * dudt;

        integrator.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        scalar zero = make_vectorized_array<Number>(0.0);
        integrator.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    integrator.integrate(true, false);
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_nbc_body_force_term_add(VectorType &   dst,
                                                                          double const & time)
{
  this->evaluation_time = time;

  VectorType src_dummy;
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_body_force_term_add_boundary_face,
                               this,
                               dst,
                               src_dummy);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_nbc_body_force_term_add_boundary_face(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(data, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);

    types::boundary_id boundary_id = data.get_boundary_id(face);
    BoundaryTypeP      boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate right-hand side
        vector rhs = evaluate_vectorial_function(this->field_functions->right_hand_side,
                                                 q_points,
                                                 this->evaluation_time);

        vector normal = integrator.get_normal_vector(q);

        scalar h = normal * rhs;

        integrator.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        scalar zero = make_vectorized_array<Number>(0.0);
        integrator.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    integrator.integrate(true, false);
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_nbc_numerical_time_derivative_add(
  VectorType &       dst,
  VectorType const & acceleration)
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_numerical_time_derivative_add_boundary_face,
                               this,
                               dst,
                               acceleration);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::
  local_rhs_ppe_nbc_numerical_time_derivative_add_boundary_face(
    MatrixFree<dim, Number> const & data,
    VectorType &                    dst,
    VectorType const &              acceleration,
    Range const &                   face_range) const
{
  unsigned int dof_index_velocity  = this->get_dof_index_velocity();
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_velocity = this->get_quad_index_velocity_linear();

  FaceIntegratorU integrator_velocity(data, true, dof_index_velocity, quad_index_velocity);
  FaceIntegratorP integrator_pressure(data, true, dof_index_pressure, quad_index_velocity);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_velocity.reinit(face);
    integrator_velocity.gather_evaluate(acceleration, true, false);

    integrator_pressure.reinit(face);

    types::boundary_id boundary_id = data.get_boundary_id(face);
    BoundaryTypeP      boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        vector normal = integrator_velocity.get_normal_vector(q);
        vector dudt   = integrator_velocity.get_value(q);
        scalar h      = -normal * dudt;

        integrator_pressure.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        scalar zero = make_vectorized_array<Number>(0.0);
        integrator_pressure.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }

    integrator_pressure.integrate(true, false);
    integrator_pressure.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_convective_add(VectorType &       dst,
                                                                 VectorType const & src) const
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_convective_add_boundary_face,
                               this,
                               dst,
                               src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_nbc_convective_add_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_dof_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_nonlinear();

  FaceIntegratorU velocity(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);
  FaceIntegratorU grid_velocity(matrix_free, true, dof_index_velocity, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    velocity.reinit(face);
    velocity.gather_evaluate(src, true, true);

    if(this->param.ale_formulation && this->param.store_previous_boundary_values)
    {
      grid_velocity.reinit(face);
      grid_velocity.gather_evaluate(this->convective_kernel->get_grid_velocity(), true, false);
    }

    pressure.reinit(face);

    BoundaryTypeP boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        vector normal = pressure.get_normal_vector(q);

        vector u      = velocity.get_value(q);
        tensor grad_u = velocity.get_gradient(q);

        vector flux;
        if(this->param.formulation_convective_term_bc ==
           FormulationConvectiveTerm::DivergenceFormulation)
        {
          scalar div_u = velocity.get_divergence(q);
          flux         = grad_u * u + div_u * u;
        }
        else if(this->param.formulation_convective_term_bc ==
                FormulationConvectiveTerm::ConvectiveFormulation)
        {
          flux = grad_u * u;
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }

        if(this->param.ale_formulation && this->param.store_previous_boundary_values)
        {
          flux -= grad_u * grid_velocity.get_value(q);
        }

        pressure.submit_value(-normal * flux, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        pressure.submit_value(make_vectorized_array<Number>(0.0), q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }

    pressure.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_viscous_add(VectorType &       dst,
                                                              VectorType const & src) const
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_viscous_add_boundary_face,
                               this,
                               dst,
                               src);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::local_rhs_ppe_nbc_viscous_add_boundary_face(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_quad_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_linear();

  FaceIntegratorU omega(matrix_free, true, dof_index_velocity, quad_index);

  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure.reinit(face);

    omega.reinit(face);
    omega.gather_evaluate(src, false, true);

    BoundaryTypeP boundary_type =
      this->boundary_descriptor_pressure->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      scalar viscosity = this->get_viscosity_boundary_face(face, q);

      if(boundary_type == BoundaryTypeP::Neumann)
      {
        scalar h = make_vectorized_array<Number>(0.0);

        vector normal = pressure.get_normal_vector(q);

        vector curl_omega = CurlCompute<dim, FaceIntegratorU>::compute(omega, q);

        h = -normal * (viscosity * curl_omega);

        pressure.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        pressure.submit_value(make_vectorized_array<Number>(0.0), q);
      }
      else
      {
        AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    pressure.integrate_scatter(true, false, dst);
  }
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_ppe_laplace_add(VectorType &   dst,
                                                              double const & evaluation_time) const
{
  ProjBase::do_rhs_ppe_laplace_add(dst, evaluation_time);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesDualSplitting<dim, Number>::solve_pressure(VectorType &       dst,
                                                         VectorType const & src,
                                                         bool const update_preconditioner) const
{
  return ProjBase::do_solve_pressure(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::rhs_add_viscous_term(VectorType & dst,
                                                               double const evaluation_time) const
{
  ProjBase::do_rhs_add_viscous_term(dst, evaluation_time);
}

template<int dim, typename Number>
unsigned int
DGNavierStokesDualSplitting<dim, Number>::solve_viscous(VectorType &       dst,
                                                        VectorType const & src,
                                                        bool const &       update_preconditioner,
                                                        double const &     factor)
{
  // Update operator
  this->momentum_operator.set_scaling_factor_mass_matrix(factor);

  unsigned int n_iter = helmholtz_solver->solve(dst, src, update_preconditioner);

  return n_iter;
}

template<int dim, typename Number>
void
DGNavierStokesDualSplitting<dim, Number>::apply_helmholtz_operator(VectorType &       dst,
                                                                   VectorType const & src) const
{
  this->momentum_operator.vmult(dst, src);
}

template class DGNavierStokesDualSplitting<2, float>;
template class DGNavierStokesDualSplitting<2, double>;

template class DGNavierStokesDualSplitting<3, float>;
template class DGNavierStokesDualSplitting<3, double>;

} // namespace IncNS
