#ifndef OPERATOR_WRAPPERS_INCOMP_PROJECTION
#define OPERATOR_WRAPPERS_INCOMP_PROJECTION

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../../include/incompressible_navier_stokes/spatial_discretization/projection_operator.h"

template<int dim, int degree_u, int degree_p, typename Number>
class OperatorWrapperProjection : public OperatorWrapperIcomp<dim, degree_u, degree_p, Number>
{
  typedef OperatorWrapperIcomp<dim, degree_u, degree_p, Number> PARENT;

public:
  OperatorWrapperProjection(parallel::distributed::Triangulation<dim> const & triangulation)
    : OperatorWrapperIcomp<dim, degree_u, degree_p, Number>(triangulation)
  {
    IncNS::ProjectionOperatorData laplace_additional_data;
    laplace.reset(new IncNS::ProjectionOperator<dim, degree_u, Number>(
      this->data, 0, 0, laplace_additional_data));

    // initialize vectors
    laplace->initialize_dof_vector(this->src);
    laplace->initialize_dof_vector(this->dst);
  }

  void
  run()
  {
    laplace->apply(this->dst, this->src);
  }

  std::shared_ptr<IncNS::ProjectionOperator<dim, degree_u, Number>> laplace;
};

#endif