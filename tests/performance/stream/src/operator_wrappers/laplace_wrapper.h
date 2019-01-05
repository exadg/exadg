#ifndef OPERATOR_WRAPPERS_LAPLACE
#define OPERATOR_WRAPPERS_LAPLACE

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "./laplace_operator.h"

#include <deal.II/distributed/tria.h>

using namespace dealii;

#include "../../../../../include/functionalities/categorization.h"

namespace Poisson
{
using namespace dealii;

template<int dim, int degree, typename Number, int n_components>
class OperatorWrapper : public OperatorWrapperBase
{
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  OperatorWrapper(parallel::distributed::Triangulation<dim> const & triangulation,
                  bool                                              do_faces,
                  bool                                              do_cell_based)
    : fe(new FESystem<dim>(FE_DGQ<dim>(degree), n_components)),
      mapping(1 /*TODO*/),
      dof_handler(triangulation),
      laplace(do_faces, do_cell_based)
  {
    this->create_dofs();
    this->initialize_matrix_free(do_cell_based);

    laplace.reinit(data);

    // initialize vectors
    laplace.initialize_dof_vector(src);
    laplace.initialize_dof_vector(dst);
  }

  void
  create_dofs()
  {
    dof_handler.distribute_dofs(*fe);
  }

  void
  initialize_matrix_free(bool do_cell_based)
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags = update_gradients | update_JxW_values;
    additional_data.mapping_update_flags_inner_faces =
      additional_data.mapping_update_flags | update_values | update_normal_vectors;
    additional_data.mapping_update_flags_boundary_faces =
      additional_data.mapping_update_flags_inner_faces | update_quadrature_points;

    QGauss<1> quadrature(degree + 1);

    if(do_cell_based)
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &dof_handler.get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data);
    }

    data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void
  run()
  {
    laplace.apply_add(dst, src);
  }

  std::shared_ptr<FESystem<dim>> fe;

  MappingQGeneric<dim>      mapping;
  AffineConstraints<double> dummy;

  DoFHandler<dim> dof_handler;

  MatrixFree<dim, Number> data;

  LaplaceOperator<dim, degree, Number, n_components> laplace;

  VectorType dst;
  VectorType src;
};

} // namespace Poisson

#endif