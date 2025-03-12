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

#ifndef INCLUDE_EXADG_OPERATORS_HYPERCUBE_RESOLUTION_PARAMETERS_H_
#define INCLUDE_EXADG_OPERATORS_HYPERCUBE_RESOLUTION_PARAMETERS_H_

// deal.II
#include <deal.II/base/parameter_handler.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/utilities/enum_patterns.h>

namespace ExaDG
{
/*
 * study throughput as a function of polynomial degree or problem size
 */
enum class RunType
{
  RefineHAndP, // run simulation for a specified range of mesh refinements and polynomial degrees
  FixedProblemSize,     // increase polynomial degree and keep problem size approximately constant
  IncreasingProblemSize // run at fixed polynomial degree
};

/*
 * Determines mesh refinement level l and number of subdivisions in 1d of hyper_cube mesh for a
 * given number of minimal and maximal number of unknowns and a given RunType.
 */
inline void
fill_resolutions_vector(
  std::vector<
    std::tuple<unsigned int /*k*/, unsigned int /*l*/, unsigned int /*subdivisions hypercube*/>> &
                                        resolutions,
  unsigned int const                    dim,
  unsigned int const                    degree,
  double const                          dofs_per_element,
  dealii::types::global_dof_index const n_dofs_min,
  dealii::types::global_dof_index const n_dofs_max,
  RunType const &                       run_type,
  ElementType const &                   element_type)
{
  unsigned int const resolutions_initial_size = resolutions.size();

  double dofs_per_cube = dofs_per_element;

  // in case of simplex elements, we create a mesh of hypercube elements that gets later subdivided
  // into simplex elements using dealii::GridTools::subdivided_hyper_cube_with_simplices()
  if(element_type == ElementType::Simplex)
  {
    unsigned int n_cells_per_cube = 1;

    // Determine the number of simplex cells per hypercube cell
    if(dim == 2)
      n_cells_per_cube = 2;
    else if(dim == 3)
      n_cells_per_cube = 5;
    else
      AssertThrow(false, dealii::ExcMessage("not implemented."));

    // From now on, we think of a hyerpcube mesh that gets later subdivided into simplex elements
    // with n_cells_per_cube simplices per hypercube cell
    dofs_per_cube *= n_cells_per_cube;
  }

  dealii::types::global_dof_index const n_cells_min =
    (n_dofs_min + dofs_per_cube - 1) / dofs_per_cube;
  dealii::types::global_dof_index const n_cells_max = n_dofs_max / dofs_per_cube;

  // From the maximum number of cells, we derive a maximum refinement level for a uniformly refined
  // mesh with one coarse-grid cells
  unsigned int const refine_level_max =
    int(std::log((double)n_cells_max) / std::log((double)dealii::Utilities::pow(2ULL, dim))) + 1;

  // we start with the coarsest possible mesh and then increase the refine level by 1
  // until we hit the maximum refinement level
  unsigned int refine_level = 0;

  // This loop is for a uniformly refined hypercube mesh. Inside the
  // loop, we test whether additional combinations of coarse-grid cells and mesh refinement levels
  // are suitable in terms of n_cells_min/n_cells_max. The reason behind is that we want to have
  // more data points than just data points differing by a factor of 2^dim in the number of cells.
  while(refine_level <= refine_level_max)
  {
    // To obtain additional data points, we test coarse grids with {3,4}^dim elements in 2D and
    // {3,4,5}^dim elements in 3D. The cases with 3 and 5 subdivisions per coordinate direction are
    // only relevant for refine_level >= 2.

    // coarse grid with 3^dim cells, and refine_level-2 uniform refinements
    if(refine_level >= 2)
    {
      unsigned int const n_subdivisions_1d = 3;
      unsigned int const l                 = refine_level - 2;

      unsigned int const n_cells =
        dealii::Utilities::pow(n_subdivisions_1d, dim) * dealii::Utilities::pow(2ULL, l * dim);

      if(n_cells >= n_cells_min and n_cells <= n_cells_max)
      {
        resolutions.push_back(
          std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));

        if(run_type == RunType::FixedProblemSize)
        {
          break;
        }
      }
    }

    // coarse grid with one cell and refine_level uniform refinements
    {
      unsigned int const n_subdivisions_1d = 1;
      unsigned int const l                 = refine_level;

      unsigned int const n_cells = dealii::Utilities::pow(2ULL, l * dim);

      if(n_cells >= n_cells_min and n_cells <= n_cells_max)
      {
        resolutions.push_back(
          std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));

        if(run_type == RunType::FixedProblemSize)
        {
          break;
        }
      }
    }

    // coarse grid with 5^dim cells, and refine_level-2 uniform refinements
    if(dim == 3 and refine_level >= 2)
    {
      unsigned int const n_subdivisions_1d = 5;
      unsigned int const l                 = refine_level - 2;

      unsigned int const n_cells =
        dealii::Utilities::pow(n_subdivisions_1d, dim) * dealii::Utilities::pow(2ULL, l * dim);

      if(n_cells >= n_cells_min and n_cells <= n_cells_max)
      {
        resolutions.push_back(
          std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));

        if(run_type == RunType::FixedProblemSize)
        {
          break;
        }
      }
    }

    // perform one global refinement
    ++refine_level;
  }

  if(run_type == RunType::FixedProblemSize)
  {
    AssertThrow(resolutions.size() > resolutions_initial_size,
                dealii::ExcMessage(
                  "No mesh found that meets the requirements regarding problem size. "
                  "Make sure that maximum number of dofs is sufficiently larger than "
                  "minimum number of dofs."));
  }
}

struct HypercubeResolutionParameters
{
  HypercubeResolutionParameters()
  {
  }

  HypercubeResolutionParameters(std::string const & input_file, unsigned int const dim) : dim(dim)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    verify_parameters();
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    prm.enter_subsection("Resolution");
    {
      prm.add_parameter(
        "RunType", run_type, "Type of throughput study.", Patterns::Enum<RunType>(), true);
      prm.add_parameter(
        "ElementType", element_type, "Type of elements.", Patterns::Enum<ElementType>(), true);
      prm.add_parameter("DegreeMin",
                        degree_min,
                        "Minimal polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1),
                        true);
      prm.add_parameter("DegreeMax",
                        degree_max,
                        "Maximal polynomial degree of shape functions.",
                        dealii::Patterns::Integer(1),
                        true);
      prm.add_parameter("RefineSpaceMin",
                        refine_space_min,
                        "Minimal number of mesh refinements.",
                        dealii::Patterns::Integer(0, 20),
                        true);
      prm.add_parameter("RefineSpaceMax",
                        refine_space_max,
                        "Maximal number of mesh refinements.",
                        dealii::Patterns::Integer(0, 20),
                        true);
      prm.add_parameter("DofsMin",
                        n_dofs_min,
                        "Minimal number of degrees of freedom.",
                        dealii::Patterns::Integer(1),
                        true);
      prm.add_parameter("DofsMax",
                        n_dofs_max,
                        "Maximal number of degrees of freedom.",
                        dealii::Patterns::Integer(1),
                        true);
    }
    prm.leave_subsection();
  }

  void
  verify_parameters()
  {
    if(run_type == RunType::RefineHAndP)
    {
      AssertThrow(degree_max >= degree_min, dealii::ExcMessage("Invalid parameters."));
      AssertThrow(refine_space_max >= refine_space_min, dealii::ExcMessage("Invalid parameters."));
    }
    else if(run_type == RunType::FixedProblemSize)
    {
      AssertThrow(degree_max >= degree_min, dealii::ExcMessage("Invalid parameters."));
      AssertThrow(n_dofs_max >= n_dofs_min, dealii::ExcMessage("Invalid parameters."));
    }
    else if(run_type == RunType::IncreasingProblemSize)
    {
      AssertThrow(
        degree_min == degree_max,
        dealii::ExcMessage(
          "Only a single polynomial degree can be considered for RunType::IncreasingProblemSize"));

      AssertThrow(n_dofs_max >= n_dofs_min, dealii::ExcMessage("Invalid parameters."));
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }
  }

  void
  fill_resolution_vector(
    std::function<double(unsigned int, unsigned int, ElementType)> const & get_dofs_per_element)
  {
    if(run_type == RunType::RefineHAndP)
    {
      unsigned int const n_cells_1d = 1;

      // k-refinement
      for(unsigned int degree = degree_min; degree <= degree_max; ++degree)
      {
        // h-refinement
        for(unsigned int refine_space = refine_space_min; refine_space <= refine_space_max;
            ++refine_space)
        {
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, refine_space, n_cells_1d));
        }
      }
    }
    else if(run_type == RunType::FixedProblemSize or run_type == RunType::IncreasingProblemSize)
    {
      for(unsigned int degree = degree_min; degree <= degree_max; ++degree)
      {
        double dofs_per_element = get_dofs_per_element(dim, degree, element_type);
        fill_resolutions_vector(resolutions,
                                dim,
                                degree,
                                dofs_per_element,
                                n_dofs_min,
                                n_dofs_max,
                                run_type,
                                element_type);
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  unsigned int dim = 2; // number of space dimensions

  RunType run_type = RunType::RefineHAndP;

  ElementType element_type = ElementType::Hypercube;

  unsigned int degree_min = 3; // minimal polynomial degree
  unsigned int degree_max = 3; // maximal polynomial degree

  unsigned int refine_space_min = 0; // minimal number of global refinements
  unsigned int refine_space_max = 0; // maximal number of global refinements

  dealii::types::global_dof_index n_dofs_min = 1e4; // minimal number of unknowns
  dealii::types::global_dof_index n_dofs_max = 3e4; // maximal number of unknowns

  // a vector storing tuples of the form (degree k, refine level l, n_subdivisions_1d)
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> resolutions;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_OPERATORS_HYPERCUBE_RESOLUTION_PARAMETERS_H_ */
