/*
 * mesh_resolution_generator_hypercube.h
 *
 *  Created on: Sep 1, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_MESH_RESOLUTION_GENERATOR_HYPERCUBE_H_
#define INCLUDE_FUNCTIONALITIES_MESH_RESOLUTION_GENERATOR_HYPERCUBE_H_

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
 * given number of minimum and maximum number of unknowns and a given RunType.
 */
void
fill_resolutions_vector(
  std::vector<
    std::tuple<unsigned int /*k*/, unsigned int /*l*/, unsigned int /*subdivisions hypercube*/>> &
                                resolutions,
  unsigned int const            degree,
  unsigned int const            dim,
  unsigned int const            dofs_per_element,
  types::global_dof_index const n_dofs_min,
  types::global_dof_index const n_dofs_max,
  RunType const &               run_type)
{
  unsigned int l = 0, n_subdivisions_1d = 1;

  double n_cells_min = n_dofs_min / dofs_per_element;
  double n_cells_max = n_dofs_max / dofs_per_element;

  int    refine_level = 0;
  double n_cells      = 1.0;

  while(n_cells <= std::pow(2, dim) * n_cells_max)
  {
    // We want to increase the problem size approximately by a factor of two, which is
    // realized by using a coarse grid with {3,4}^dim elements in 2D and {3,4,5}^dim elements
    // in 3D.

    // coarse grid with 3^dim cells, and refine_level-2 uniform refinements
    if(refine_level >= 2)
    {
      n_subdivisions_1d = 3;
      n_cells           = std::pow(n_subdivisions_1d, dim) * std::pow(2., (refine_level - 2) * dim);

      if(n_cells >= n_cells_min && n_cells <= n_cells_max)
      {
        // set mesh refinement
        l = refine_level - 2;

        if(run_type == RunType::FixedProblemSize)
          break;
        else if(run_type == RunType::IncreasingProblemSize)
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
        else
          AssertThrow(false, ExcMessage("Not implemented:"));
      }
    }

    // coarse grid with only a single cell, and refine_level uniform refinements
    {
      n_subdivisions_1d = 1;
      n_cells           = std::pow(2., refine_level * dim);

      if(n_cells >= n_cells_min && n_cells <= n_cells_max)
      {
        // set mesh refinement
        l = refine_level;

        if(run_type == RunType::FixedProblemSize)
          break;
        else if(run_type == RunType::IncreasingProblemSize)
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
        else
          AssertThrow(false, ExcMessage("Not implemented:"));
      }
    }

    // coarse grid with 5^dim cells, and refine_level-2 uniform refinements
    if(dim == 3 && refine_level >= 2)
    {
      n_subdivisions_1d = 5;
      n_cells           = std::pow(n_subdivisions_1d, dim) * std::pow(2., (refine_level - 2) * dim);

      if(n_cells >= n_cells_min && n_cells <= n_cells_max)
      {
        // set mesh refinement
        l = refine_level - 2;

        if(run_type == RunType::FixedProblemSize)
          break;
        else if(run_type == RunType::IncreasingProblemSize)
          resolutions.push_back(
            std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
        else
          AssertThrow(false, ExcMessage("Not implemented:"));
      }
    }

    // perform one global refinement
    ++refine_level;
    n_cells = std::pow(2., refine_level * dim);
  }

  if(run_type == RunType::FixedProblemSize)
  {
    AssertThrow((n_cells >= n_cells_min && n_cells <= n_cells_max),
                ExcMessage("No mesh found that meets the requirements regarding problem size. "
                           "Make sure that N_DOFS_MAX is sufficiently larger than N_DOFS_MIN."));

    resolutions.push_back(
      std::tuple<unsigned int, unsigned int, unsigned int>(degree, l, n_subdivisions_1d));
  }
  else
  {
    AssertThrow(run_type == RunType::IncreasingProblemSize, ExcMessage("Not implemented."));
  }
}



#endif /* INCLUDE_FUNCTIONALITIES_MESH_RESOLUTION_GENERATOR_HYPERCUBE_H_ */
