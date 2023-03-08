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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// ExaDG
#include <exadg/incompressible_navier_stokes/spatial_discretization/calculators/wall_shear_stress_calculator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
WallShearStressCalculator<dim, Number>::WallShearStressCalculator()
  : matrix_free(nullptr), dof_index(0), quad_index(0), dynamic_viscosity(0.0)
{
}

template<int dim, typename Number>
void
WallShearStressCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in,
  double const                            dynamic_viscosity_in)
{
  matrix_free       = &matrix_free_in;
  dof_index         = dof_index_in;
  quad_index        = quad_index_in;
  dynamic_viscosity = dynamic_viscosity_in;

  // identify default matching Gauss-Lobatto point indices once up
  // front since orientation of boundary faces *should be* constant
  bool is_hyper_cube =
    matrix_free->get_dof_handler(dof_index).get_fe(0).reference_cell().is_hyper_cube();
  AssertThrow(is_hyper_cube,
              dealii::ExcMessage("WallShearStressCalculator assumes hypercube reference cells."));

  unsigned int const    fe_degree = matrix_free->get_dof_handler(dof_index).get_fe(0).degree;
  dealii::FESystem<dim> fe_system(dealii::FE_DGQ<dim>(fe_degree), dim);
  dealii::QGaussLobatto<dim - 1> face_quadrature(fe_degree + 1);
  dealii::FEFaceValues<dim>      fe_face_values(fe_system,
                                           face_quadrature,
                                           dealii::update_quadrature_points);
  dealii::QGaussLobatto<dim>     quadrature(fe_degree + 1);
  dealii::FEValues<dim>          fe_values(fe_system, quadrature, dealii::update_quadrature_points);

  face_to_cell_index.clear();
  for(auto const & cell : matrix_free->get_dof_handler(dof_index).active_cell_iterators())
  {
    if(cell->is_locally_owned() == true)
    {
      fe_values.reinit(cell);
      double const tol_sqrd = std::pow((cell->diameter() * rel_tol), 2);

      face_to_cell_index.resize(cell->n_faces());

      for(auto const face : cell->face_indices())
      {
        fe_face_values.reinit(cell, face);
        std::vector<dealii::types::global_dof_index> face_cell_index_matches(
          fe_face_values.n_quadrature_points, dealii::numbers::invalid_dof_index);

        for(auto const i : fe_face_values.quadrature_point_indices())
        {
          dealii::Point<dim> const point_face = fe_face_values.quadrature_point(i);
          for(auto const j : fe_values.quadrature_point_indices())
          {
            if(point_face.distance_square(fe_values.quadrature_point(j)) < tol_sqrd)
            {
              face_cell_index_matches[i] = j;
              break;
            }
          }
        }

        face_to_cell_index[face] = face_cell_index_matches;
      }

      break;
    }
  }
}

template<int dim, typename Number>
void
WallShearStressCalculator<dim, Number>::compute_wall_shear_stress(
  VectorType &                                dst,
  VectorType const &                          src,
  std::shared_ptr<dealii::Mapping<dim> const> mapping,
  std::set<dealii::types::boundary_id> const  write_wall_shear_stress_boundary_IDs) const
{
  dst = 0;

  // if no write_wall_shear_stress_boundary_IDs is provided, the default invalid_boundary_id
  // indicates writing the wall_shear_stress on all boundaries.
  bool write_on_all_boundary_IDs = false;
  if(write_wall_shear_stress_boundary_IDs.size() == 1)
  {
    // no boundary_id(s) provided, but write_wall_shear_stress == true
    if(*write_wall_shear_stress_boundary_IDs.begin() == dealii::numbers::invalid_boundary_id)
      write_on_all_boundary_IDs = true;
  }

  bool is_hyper_cube =
    matrix_free->get_dof_handler(dof_index).get_fe(0).reference_cell().is_hyper_cube();
  AssertThrow(is_hyper_cube,
              dealii::ExcMessage("WallShearStressCalculator assumes hypercube reference cells."));

  unsigned int const    fe_degree = matrix_free->get_dof_handler(dof_index).get_fe(0).degree;
  dealii::FESystem<dim> fe_system(dealii::FE_DGQ<dim>(fe_degree), dim);
  dealii::QGaussLobatto<dim - 1> face_quadrature(fe_degree + 1);
  dealii::QGaussLobatto<dim>     quadrature(fe_degree + 1);

  unsigned int const dofs_per_component =
    matrix_free->get_dof_handler(dof_index).get_fe(0).dofs_per_cell / dim;

#ifdef DEBUG
  dealii::FEFaceValues<dim> fe_face_values(*mapping,
                                           fe_system,
                                           face_quadrature,
                                           dealii::update_quadrature_points |
                                             dealii::update_gradients |
                                             dealii::update_normal_vectors);
  dealii::FEValues<dim>     fe_values(*mapping,
                                  fe_system,
                                  quadrature,
                                  dealii::update_quadrature_points);
#else
  dealii::FEFaceValues<dim> fe_face_values(
    *mapping, fe_system, face_quadrature, dealii::update_gradients | dealii::update_normal_vectors);
#endif

  std::vector<dealii::Tensor<2, dim>> velocity_gradients(fe_face_values.n_quadrature_points);
  std::vector<dealii::types::global_dof_index> dof_indices(fe_system.dofs_per_cell);
  const dealii::FEValuesExtractors::Vector     vector(0);

  // fill vector with entries on boundary via cell-loop or prepared container.
  for(auto const & cell : matrix_free->get_dof_handler(dof_index).active_cell_iterators())
  {
    if(cell->is_locally_owned() == true && cell->at_boundary() == true)
    {
      cell->get_dof_indices(dof_indices);

#ifdef DEBUG
      fe_values.reinit(cell);
      double const tol_sqrd = std::pow((cell->diameter() * rel_tol), 2);
#endif

      for(auto const face : cell->face_indices())
      {
        if(cell->at_boundary(face) == true)
        {
          if(write_on_all_boundary_IDs ||
             write_wall_shear_stress_boundary_IDs.find(cell->face(face)->boundary_id()) !=
               write_wall_shear_stress_boundary_IDs.end())
          {
            fe_face_values.reinit(cell, face);
            fe_face_values[vector].get_function_gradients(src, velocity_gradients);

            for(auto const i : fe_face_values.quadrature_point_indices())
            {
              unsigned int matching_idx = face_to_cell_index[face][i];

#ifdef DEBUG
              // check point match in debug mode
              dealii::Point<dim> const point_face = fe_face_values.quadrature_point(i);

              if(point_face.distance_square(fe_values.quadrature_point(matching_idx)) >= tol_sqrd)
              {
                matching_idx = dealii::numbers::invalid_dof_index;
                for(auto const j : fe_values.quadrature_point_indices())
                {
                  if(point_face.distance_square(fe_values.quadrature_point(j)) < tol_sqrd)
                  {
                    matching_idx = j;
                    break;
                  }
                }
                std::cout << "Recover: "
                          << "face = " << face << "i = " << i
                          << "miss = " << face_to_cell_index[face][i] << " new = " << matching_idx
                          << "\n";
                AssertThrow(matching_idx == dealii::numbers::invalid_dof_index,
                            dealii::ExcMessage("Face to cell match recovery not succesful."));
              }
#endif

              dealii::Tensor<1, dim> const normal = fe_face_values.normal_vector(i);
              dealii::Tensor<1, dim> const S_times_n =
                dynamic_viscosity * (velocity_gradients[i] + transpose(velocity_gradients[i])) *
                normal;
              dealii::Tensor<1, dim> const wall_shear_stress =
                S_times_n - scalar_product((S_times_n), normal) * normal;
              for(unsigned int d = 0; d < dim; ++d)
              {
                if(src.get_partitioner()->in_local_range(
                     dof_indices[matching_idx + d * dofs_per_component]))
                  dst(dof_indices[matching_idx + d * dofs_per_component]) = wall_shear_stress[d];
              }
            }
          }
        }
      }
    }
  }

  dst.update_ghost_values();
}

template class WallShearStressCalculator<2, float>;
template class WallShearStressCalculator<2, double>;

template class WallShearStressCalculator<3, float>;
template class WallShearStressCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
