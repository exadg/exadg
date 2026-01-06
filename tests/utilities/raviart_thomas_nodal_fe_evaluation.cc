/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// C/C++
#include <iostream>

// deal.ii
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>

using namespace dealii;

int
main()
{
  constexpr int dim = 2;
  using Number      = double;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);

  MappingQ1<dim> const mapping;

  unsigned int constexpr degree = 5;
  FE_RaviartThomasNodal<dim> fe(degree);
  DoFHandler<dim>            dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  unsigned int constexpr n_q_points_1d = degree + 1;
  QGauss<1> quad(n_q_points_1d);

  AffineConstraints<Number> constraints;
  constraints.close();

  MatrixFree<dim, Number>                          mf;
  typename MatrixFree<dim, Number>::AdditionalData data;

  data.mapping_update_flags                = update_gradients | update_JxW_values;
  data.mapping_update_flags_boundary_faces = update_gradients | update_JxW_values;
  data.mapping_update_flags_inner_faces    = update_gradients | update_JxW_values;

  mf.reinit(mapping, dof_handler, constraints, quad, data);

  Vector<double> dst, src;
  mf.initialize_dof_vector(dst);
  mf.initialize_dof_vector(src);

  src.add(1.0);

  unsigned int constexpr n_components = dim;

  // Use the runtime degree evaluators with `degree = -1`.
  CellIntegrator<dim, n_components, Number> fe_eval(mf, 0 /*dof_index*/, 0 /*quad_index*/);
  FaceIntegrator<dim, n_components, Number> fe_face_eval(mf,
                                                         true /* is_interior_face */,
                                                         0 /*dof_index*/,
                                                         0 /*quad_index*/);

  // this is equivalent to:
  // `FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArray<Number>> fe_eval(mf);`
  // `FEFaceEvaluation<dim, -1, 0, n_components, Number, VectorizedArray<Number>> fe_face_eval(mf);`

  std::cout << "cells: fe_eval.fast_evaluation_supported(degree, n_q_points_1d) = "
            << fe_eval.fast_evaluation_supported(degree, n_q_points_1d) << "\n";

  std::cout << "faces: fe_face_eval.fast_evaluation_supported(degree, n_q_points_1d) = "
            << fe_face_eval.fast_evaluation_supported(degree, n_q_points_1d) << "\n";

  mf.loop<Vector<double>, Vector<double>>(
    /*
     * cell loop
     */
    [&](MatrixFree<dim, Number> const &               matrix_free,
        Vector<double> &                              dst,
        Vector<double> const &                        src,
        std::pair<unsigned int, unsigned int> const & range) {
      (void)matrix_free;

      if constexpr(true)
      {
        std::cout << "cells contribute"
                  << "\n";

        for(unsigned int cell = range.first; cell < range.second; ++cell)
        {
          fe_eval.reinit(cell);
          fe_eval.read_dof_values(src);
          fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            auto const & value      = fe_eval.get_value(q);
            auto const & divergence = fe_eval.get_divergence(q);
            fe_eval.submit_value(value * divergence + value * 2.0, q);
          }

          // (v_h, u_h * div(u_h))_Omega
          fe_eval.integrate_scatter(EvaluationFlags::values, dst);
        }
      }
    },
    /*
     * face loop
     */
    [&](MatrixFree<dim, Number> const &               matrix_free,
        Vector<double> &                              dst,
        Vector<double> const &                        src,
        std::pair<unsigned int, unsigned int> const & range) {
      (void)matrix_free;

      if constexpr(true)
      {
        std::cout << "inner faces contribute"
                  << "\n";

        for(unsigned int face = range.first; face < range.second; ++face)
        {
          fe_face_eval.reinit(face);
          fe_face_eval.read_dof_values(src);
          fe_face_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          for(unsigned int q = 0; q < fe_face_eval.n_q_points; ++q)
          {
            auto const & value      = fe_face_eval.get_value(q);
            auto const & divergence = fe_face_eval.get_divergence(q);
            fe_face_eval.submit_value(value * divergence + value * 2.0, q);
          }

          // (v_h, u_h * div(u_h))__Gamma
          fe_face_eval.integrate_scatter(EvaluationFlags::values, dst);
        }
      }
    },
    /*
     * boundary face loop
     */
    [&](MatrixFree<dim, Number> const &               matrix_free,
        Vector<double> &                              dst,
        Vector<double> const &                        src,
        std::pair<unsigned int, unsigned int> const & range) {
      (void)matrix_free;

      if constexpr(true)
      {
        std::cout << "boundary faces contribute"
                  << "\n";

        for(unsigned int face = range.first; face < range.second; ++face)
        {
          fe_face_eval.reinit(face);
          fe_face_eval.read_dof_values(src);
          fe_face_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          for(unsigned int q = 0; q < fe_face_eval.n_q_points; ++q)
          {
            auto const & value      = fe_face_eval.get_value(q);
            auto const & divergence = fe_face_eval.get_divergence(q);
            fe_face_eval.submit_value(value * divergence + value * 2.0, q);
          }

          // (v_h, u_h * div(u_h))__Gamma
          fe_face_eval.integrate_scatter(EvaluationFlags::values, dst);
        }
      }
    },
    dst,
    src);

  std::cout << "dst.l2_norm() = " << dst.l2_norm() << "\n";
}
