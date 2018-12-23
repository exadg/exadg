#ifndef OPERATOR_WRAPPERS_INCOMP_CONV
#define OPERATOR_WRAPPERS_INCOMP_CONV

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../../include/incompressible_navier_stokes/spatial_discretization/operators/convective_operator.h"
#include "../../../../../include/incompressible_navier_stokes/spatial_discretization/operators/viscous_operator.h"
#include "icomp_wrapper.h"

template<int dim, int degree_u, int degree_p, typename Number>
class OperatorWrapperIcomConvectiveOperator
  : public OperatorWrapperIcomp<dim, degree_u, degree_p, Number>
{
  typedef OperatorWrapperIcomp<dim, degree_u, degree_p, Number> PARENT;

public:
  OperatorWrapperIcomConvectiveOperator(
    parallel::distributed::Triangulation<dim> const & triangulation)
    : OperatorWrapperIcomp<dim, degree_u, degree_p, Number>(triangulation)
  {
    IncNS::ConvectiveOperatorData<dim> convective_operator_data;

    // convective_operator_data.formulation          = param.formulation_convective_term;
    convective_operator_data.dof_index  = PARENT::dof_index_u;
    convective_operator_data.quad_index = PARENT::quad_index_u_nonlinear;
    // convective_operator_data.upwind_factor        = param.upwind_factor;
    // convective_operator_data.bc                   = boundary_descriptor_velocity;
    // convective_operator_data.use_outflow_bc       = param.use_outflow_bc_convective_term;
    // convective_operator_data.type_dirichlet_bc    = param.type_dirichlet_bc_convective;
    // convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
    convective_operator.initialize(this->data, convective_operator_data);

    // initialize vectors
    this->data.initialize_dof_vector(this->src, PARENT::dof_index_u);
    this->data.initialize_dof_vector(this->dst, PARENT::dof_index_u);
  }

  void
  run()
  {
    convective_operator.apply(this->dst, this->src, 0.0);
  }

  IncNS::ConvectiveOperator<dim, degree_u, Number> convective_operator;
};

template<int dim, int degree_u, int degree_p, typename Number>
class OperatorWrapperIcomViscousOperator
  : public OperatorWrapperIcomp<dim, degree_u, degree_p, Number>
{
  typedef OperatorWrapperIcomp<dim, degree_u, degree_p, Number> PARENT;

public:
  OperatorWrapperIcomViscousOperator(
    parallel::distributed::Triangulation<dim> const & triangulation)
    : OperatorWrapperIcomp<dim, degree_u, degree_p, Number>(triangulation)
  {
    IncNS::ViscousOperatorData<dim> viscous_operator_data;
    // viscous_operator_data.formulation_viscous_term     = param.formulation_viscous_term;
    // viscous_operator_data.penalty_term_div_formulation = param.penalty_term_div_formulation;
    // viscous_operator_data.IP_formulation               = param.IP_formulation_viscous;
    // viscous_operator_data.IP_factor                    = param.IP_factor_viscous;
    // viscous_operator_data.bc                           = boundary_descriptor_velocity;
    viscous_operator_data.dof_index  = PARENT::dof_index_u;
    viscous_operator_data.quad_index = PARENT::quad_index_u;
    // viscous_operator_data.viscosity                    = param.viscosity;
    // viscous_operator_data.use_cell_based_loops         = param.use_cell_based_face_loops;
    viscous_operator.initialize(this->mapping, this->data, viscous_operator_data);

    // initialize vectors
    this->data.initialize_dof_vector(this->src, PARENT::dof_index_u);
    this->data.initialize_dof_vector(this->dst, PARENT::dof_index_u);
  }

  void
  run()
  {
    viscous_operator.apply(this->dst, this->src);
  }

  IncNS::ViscousOperator<dim, degree_u, Number> viscous_operator;
};

#endif