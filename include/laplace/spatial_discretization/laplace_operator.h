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

#include "../../../include/operators/operation_base.h"
#include "../user_interface/boundary_descriptor.h"
#include "../../operators/interior_penalty_parameter.h"

namespace Laplace
{

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

template <int dim> struct LaplaceOperatorData : public OperatorBaseData<dim, BoundaryType, OperatorType,
                              BoundaryDescriptor<dim>> {
public:
  LaplaceOperatorData()
      : OperatorBaseData<dim, BoundaryType, OperatorType, BoundaryDescriptor<dim>>(
              0, 0, false, true, false, false, true, false,
                              true, true, true, true, // face
                              true, true, true, true  // boundary
                              ), IP_factor(1.0) {}
      
      double IP_factor;
      
};

template <int dim, int degree, typename Number>
class LaplaceOperator
    : public OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>> {
public:
  LaplaceOperator();
  
  void initialize(Mapping<dim> const &mapping,
                  MatrixFree<dim, Number> const &mf_data,
                  LaplaceOperatorData<dim> const &operator_data_in) {
  ConstraintMatrix cm;
  Parent::reinit(mf_data, cm, operator_data_in);

  // calculate penalty parameters
  IP::calculate_penalty_parameter<dim, degree, Number>(
      array_penalty_parameter, *this->data, mapping, this->ad.dof_index);
}
  
void reinit(
    const DoFHandler<dim> &dof_handler, const Mapping<dim> &mapping,
    void* od, const MGConstrainedDoFs &mg_constrained_dofs, 
    const unsigned int level){
  Parent::reinit(dof_handler, mapping, od, mg_constrained_dofs, level);

  // calculate penalty parameters
  IP::calculate_penalty_parameter<dim, degree, Number>(
      array_penalty_parameter, *this->data, mapping, this->ad.dof_index);
}

  // typedefs
  typedef LaplaceOperator<dim, degree, Number> This;
  typedef OperatorBase<dim, degree, Number, LaplaceOperatorData<dim>> Parent;
  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VNumber VNumber;
  
  // static constants
  static const int DIM = Parent::DIM;

  void do_cell_integral(FEEvalCell &phi) const;
  void do_face_integral(FEEvalFace &p_n, FEEvalFace &p_p) const;
  void do_face_int_integral(FEEvalFace &p_n, FEEvalFace &p_p) const;
  void do_face_ext_integral(FEEvalFace &p_n, FEEvalFace &p_p) const;
  void do_boundary_integral(FEEvalFace &fe_eval,
                            OperatorType const &operator_type,
                            types::boundary_id const &boundary_id) const;
  
  MatrixOperatorBaseNew<dim, Number>* get_new(unsigned int deg) const{
      switch (deg) {
      case 1:
        return new LaplaceOperator<dim, 1, Number>();
//      case 2:
//        return new LaplaceOperator<dim, 2, Number>();
      case 3:
        return new LaplaceOperator<dim, 3, Number>();
//      case 4:
//        return new LaplaceOperator<dim, 4, Number>();
//      case 5:
//        return new LaplaceOperator<dim, 5, Number>();
//      case 6:
//        return new LaplaceOperator<dim, 6, Number>();
      case 7:
        return new LaplaceOperator<dim, 7, Number>();
      default:
        AssertThrow(false,
                    ExcMessage("LaplaceOperator not implemented for this degree!"));
        return new LaplaceOperator<dim, 1, Number>(); // dummy return (statement not
                                                      // reached)
      }
  }
  
private:
  AlignedVector<VectorizedArray<Number>> array_penalty_parameter;
  
};

}

#include "laplace_operator.cpp"

#endif