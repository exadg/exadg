#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/distributed/tria.h>

#include "../../../../../include/functionalities/lazy_ptr.h"

namespace Poisson
{
using namespace dealii;

template<int dim, int degree, typename Number, int n_components>
class LaplaceOperator
{
public:
  static const int DIM = dim;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef LaplaceOperator<dim, degree, Number, n_components>              This;
  typedef std::pair<unsigned int, unsigned int>                           Range;
  typedef FEEvaluation<dim, degree, degree + 1, n_components, Number>     FEEvalCell;
  typedef FEFaceEvaluation<dim, degree, degree + 1, n_components, Number> FEEvalFace;

  LaplaceOperator(bool do_eval_faces, bool do_cell_based)
    : do_eval_faces(do_eval_faces), do_cell_based(do_cell_based){};


  void
  reinit(MatrixFree<dim, Number> const & matrix_free) const
  {
    this->data.reinit(matrix_free);
  }

  bool do_eval_faces;
  bool do_cell_based;



  void
  initialize_dof_vector(VectorType & vector) const
  {
    data->initialize_dof_vector(vector);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    if(do_cell_based)
    {
      data->cell_loop(&This::cell_based_loop, this, dst, src);
    }
    else
    {
      if(do_eval_faces)
        data->loop(&This::cell_loop, &This::face_loop, &This::boundary_loop, this, dst, src);
      else
        data->cell_loop(&This::cell_loop, this, dst, src);
    }
  }


  void
  cell_based_loop(MatrixFree<dim, Number> const & data,
                  VectorType &                    dst,
                  VectorType const &              src,
                  Range const &                   range) const
  {
    FEEvalCell fe_eval(data);
    FEEvalFace fe_eval_m(data, true);
    FEEvalFace fe_eval_p(data, false);

    for(auto cell = range.first; cell < range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      fe_eval.distribute_local_to_global(dst);

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        const auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        const auto bid  = bids[0];

        // density
        fe_eval_m.reinit(cell, face);
        fe_eval_m.read_dof_values(src);

        if(bid == numbers::internal_face_boundary_id)
        {
          fe_eval_p.reinit(cell, face);
          fe_eval_p.read_dof_values(src);
        }

        fe_eval_m.distribute_local_to_global(dst);
      }
    }
  }


  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   range) const
  {
    FEEvalCell fe_eval(data);

    for(auto cell = range.first; cell < range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.distribute_local_to_global(dst);
    }
  }


  void
  face_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   range) const
  {
    FEEvalFace fe_eval_m(data, true);
    FEEvalFace fe_eval_p(data, false);

    for(auto face = range.first; face < range.second; ++face)
    {
      fe_eval_m.reinit(face);
      fe_eval_p.reinit(face);
      fe_eval_m.read_dof_values(src);
      fe_eval_p.read_dof_values(src);
      fe_eval_m.distribute_local_to_global(dst);
      fe_eval_p.distribute_local_to_global(dst);
    }
  }


  void
  boundary_loop(MatrixFree<dim, Number> const & data,
                VectorType &                    dst,
                VectorType const &              src,
                Range const &                   range) const
  {
    FEEvalFace fe_eval(data, true);

    for(unsigned int face = range.first; face < range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.read_dof_values(src);
      fe_eval.distribute_local_to_global(dst);
    }
  }


  mutable lazy_ptr<MatrixFree<dim, Number>> data;
};

} // namespace Poisson

#endif
