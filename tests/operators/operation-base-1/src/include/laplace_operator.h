#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/multigrid/mg_base.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "../../../../../include/operators/operation_base.h"

enum class OperatorType {
  full,
  homogeneous,
  inhomogeneous
};

enum class BoundaryType {
  undefined,
  dirichlet,
  neumann
};

template<int dim>
struct BoundaryDescriptor
{
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet_bc;
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > neumann_bc;
};

template <int dim> struct LaplaceOperatorData : public OperatorBaseData<dim, BoundaryType, OperatorType,
                              BoundaryDescriptor<dim>> {
public:
  LaplaceOperatorData()
      : OperatorBaseData<dim, BoundaryType, OperatorType, BoundaryDescriptor<dim>>(
              0, 0, false, true, false, false, true, false,
                              true, true, true, true, // face
                              true, true, true, true  // boundary
                              ) {}
};

template <int dim, int degree, typename Number>
class LaplaceOperator
    : public OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>> {
public:
  LaplaceOperator();

  typedef LaplaceOperator<dim, degree, Number> This;
  typedef OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>> Parent;
  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;

  void do_cell_integral(FEEvalCell &phi) const;
  void do_face_integral(FEEvalFace &p_n, FEEvalFace &p_p) const;
  void do_face_int_integral(FEEvalFace &p_n, FEEvalFace &p_p) const;
  void do_face_ext_integral(FEEvalFace &p_n, FEEvalFace &p_p) const;
  void do_boundary_integral(FEEvalFace &fe_eval,
                            OperatorType const &operator_type,
                            types::boundary_id const &boundary_id) const;
};

#endif