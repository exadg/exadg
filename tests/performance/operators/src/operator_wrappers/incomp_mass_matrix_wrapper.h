#ifndef OPERATOR_WRAPPERS_INCOMP_MASS_MATRIX
#define OPERATOR_WRAPPERS_INCOMP_MASS_MATRIX

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../../include/incompressible_navier_stokes/spatial_discretization/operators/mass_matrix_operator.h"
#include "icomp_wrapper.h"

template<int dim, int degree_u, int degree_p, typename Number>
class OperatorWrapperIcompMassMatrix : public OperatorWrapperIcomp<dim, degree_u, degree_p, Number>
{
  typedef OperatorWrapperIcomp<dim, degree_u, degree_p, Number> PARENT;

public:
  OperatorWrapperIcompMassMatrix(parallel::distributed::Triangulation<dim> const & triangulation)
    : OperatorWrapperIcomp<dim, degree_u, degree_p, Number>(triangulation)
  {
    IncNS::MassMatrixOperatorData mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index  = PARENT::dof_index_u;
    mass_matrix_operator_data.quad_index = PARENT::quad_index_u;
    mass_matrix_operator.initialize(this->data, mass_matrix_operator_data);

    // initialize vectors
    this->data.initialize_dof_vector(this->src, PARENT::dof_index_u);
    this->data.initialize_dof_vector(this->dst, PARENT::dof_index_u);
  }

  void
  run()
  {
    mass_matrix_operator.apply(this->dst, this->src);
  }

  IncNS::MassMatrixOperator<dim, degree_u, Number> mass_matrix_operator;
};

#endif