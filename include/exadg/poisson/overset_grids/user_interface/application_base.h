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

#ifndef INCLUDE_EXADG_POISSON_OVERSET_GRIDS_USER_INTERFACE_APPLICATION_BASE_H_
#define INCLUDE_EXADG_POISSON_OVERSET_GRIDS_USER_INTERFACE_APPLICATION_BASE_H_

// deal.II
#include <deal.II/grid/grid_tools_cache.h>

// ExaDG
#include <exadg/poisson/user_interface/application_base.h>

namespace ExaDG
{
namespace Poisson
{
namespace OversetGrids
{
/**
 * This function determines which faces of the dst triangulation are inside the src-triangulation.
 * A face is considered inside, if all vertices of the face are inside. Then, the boundary ID is
 * set to bid for all the faces of the dst-triangulation in the overlap region.
 */
template<int dim>
void
set_boundary_ids_overlap_region(dealii::Triangulation<dim> const & tria_dst,
                                dealii::types::boundary_id const & bid,
                                dealii::Mapping<dim> const &       mapping_src,
                                dealii::Triangulation<dim> const & tria_src)
{
  std::vector<dealii::Point<dim>> points;
  using CellIteratorType = typename dealii::Triangulation<dim>::cell_iterator;
  using Id               = std::tuple<CellIteratorType /* cell */, unsigned int /*face*/>;
  std::vector<std::pair<Id, unsigned int /* first_point_in_vector */>> id_to_vector_index;

  // fill vector of points for all boundary faces
  for(auto cell : tria_dst.cell_iterators())
  {
    for(auto const & f : cell->face_indices())
    {
      if(cell->face(f)->at_boundary())
      {
        for(auto const & v : cell->face(f)->vertex_indices())
        {
          if(v == 0)
          {
            Id const id = std::make_tuple(cell, f);
            id_to_vector_index.push_back({id, points.size()});
          }

          points.push_back(cell->face(f)->vertex(v));
        }
      }
    }
  }

  // create and reinit RemotePointEvaluation: find points on src-side
  std::vector<bool> marked_vertices = {};
  double const      tolerance       = 1.e-10;

  dealii::Utilities::MPI::RemotePointEvaluation<dim> rpe =
    dealii::Utilities::MPI::RemotePointEvaluation<dim>(tolerance,
                                                       false
#if DEAL_II_VERSION_GTE(9, 4, 0)
                                                       ,
                                                       0,
                                                       [marked_vertices]() {
                                                         return marked_vertices;
                                                       }
#endif
    );

  rpe.reinit(points, tria_src, mapping_src);

  // check which points have been found and whether a face on dst-side is located inside the src
  // triangulation
  for(auto iter = id_to_vector_index.begin(); iter != id_to_vector_index.end(); ++iter)
  {
    unsigned int const begin = iter->second;
    unsigned int const end =
      (iter + 1 != id_to_vector_index.end()) ? (iter + 1)->second : points.size();

    bool inside = true;
    for(unsigned int i = begin; i < end; ++i)
    {
      inside = (inside and rpe.point_found(i));
    }

    if(inside)
    {
      auto const & [cell, f] = iter->first;
      cell->face(f)->set_boundary_id(bid);
    }
  }
}

template<int dim, int n_components, typename Number>
class Domain
{
public:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef typename std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
    PeriodicFaces;

  virtual void
  add_parameters(dealii::ParameterHandler & prm, std::vector<std::string> const & subsection_names)
  {
    for(auto & name : subsection_names)
    {
      prm.enter_subsection(name);
    }

    resolution.add_parameters(prm);
    output_parameters.add_parameters(prm);

    for(auto & name : subsection_names)
    {
      (void)name;
      prm.leave_subsection();
    }
  }

  Domain(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
      parameter_file(parameter_file)
  {
    grid = std::make_shared<Grid<dim>>();
  }

  virtual ~Domain()
  {
  }

  void
  setup_pre(std::vector<std::string> const & subsection_names)
  {
    // parameters
    parse_parameters(subsection_names);

    // set resolution parameters
    param.grid.n_refine_global = this->resolution.refine_space;
    param.degree               = this->resolution.degree;

    set_parameters();
    param.check();
    param.print(pcout, "List of parameters:");

    // grid
    GridUtilities::create_mapping(mapping, param.grid.element_type, param.mapping_degree);
    create_grid();
    print_grid_info(pcout, *grid);
  }

  void
  setup_post()
  {
    // boundary conditions
    boundary_descriptor = std::make_shared<BoundaryDescriptor<rank, dim>>();
    set_boundary_descriptor();
    verify_boundary_conditions(*boundary_descriptor, *grid);

    // field functions
    field_functions = std::make_shared<FieldFunctions<dim>>();
    set_field_functions();
  }

  virtual std::shared_ptr<Poisson::PostProcessorBase<dim, n_components, Number>>
  create_postprocessor() = 0;

  Parameters const &
  get_parameters() const
  {
    return param;
  }

  std::shared_ptr<Grid<dim> const>
  get_grid() const
  {
    return grid;
  }

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const
  {
    return mapping;
  }

  std::shared_ptr<BoundaryDescriptor<rank, dim> const>
  get_boundary_descriptor() const
  {
    return boundary_descriptor;
  }

  std::shared_ptr<FieldFunctions<dim> const>
  get_field_functions() const
  {
    return field_functions;
  }

protected:
  virtual void
  parse_parameters(std::vector<std::string> const & subsection_names)
  {
    dealii::ParameterHandler prm;
    this->add_parameters(prm, subsection_names);
    prm.parse_input(parameter_file, "", true, true);
  }

  MPI_Comm const & mpi_comm;

  dealii::ConditionalOStream pcout;

  Parameters param;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<dealii::Mapping<dim>> mapping;

  std::shared_ptr<BoundaryDescriptor<rank, dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>           field_functions;

  std::string parameter_file;

  ResolutionParameters resolution;

  OutputParameters output_parameters;

private:
  virtual void
  set_parameters() = 0;

  virtual void
  create_grid() = 0;

  virtual void
  set_boundary_descriptor() = 0;

  virtual void
  set_field_functions() = 0;
};

template<int dim, int n_components, typename Number>
class ApplicationBase
{
public:
  ApplicationBase(std::string parameter_file, MPI_Comm const & comm)
    : mpi_comm(comm), parameter_file(parameter_file)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    AssertThrow(domain1.get(), dealii::ExcMessage("Domain 1 is uninitialized."));
    AssertThrow(domain2.get(), dealii::ExcMessage("Domain 2 is uninitialized."));

    domain1->add_parameters(prm, {"Domain1"});
    domain2->add_parameters(prm, {"Domain1"});
  }

  virtual ~ApplicationBase()
  {
  }

  void
  setup()
  {
    AssertThrow(domain1.get(), dealii::ExcMessage("Domain 1 is uninitialized."));
    AssertThrow(domain2.get(), dealii::ExcMessage("Domain 2 is uninitialized."));

    domain1->setup_pre({"Domain1"});
    domain2->setup_pre({"Domain2"});

    set_boundary_ids();

    domain1->setup_post();
    domain2->setup_post();
  }

  std::shared_ptr<Domain<dim, n_components, Number>> domain1, domain2;

protected:
  MPI_Comm const & mpi_comm;

  // use "-1" since max() is defined invalid by deal.II
  dealii::types::boundary_id boundary_id_overlap =
    std::numeric_limits<dealii::types::boundary_id>::max() - 1;

private:
  void
  set_boundary_ids()
  {
    // set boundary IDs for domain 1
    set_boundary_ids_overlap_region(*domain1->get_grid()->triangulation,
                                    boundary_id_overlap,
                                    *domain2->get_mapping(),
                                    *domain2->get_grid()->triangulation);

    // set boundary IDs for domain 2
    set_boundary_ids_overlap_region(*domain2->get_grid()->triangulation,
                                    boundary_id_overlap,
                                    *domain1->get_mapping(),
                                    *domain1->get_grid()->triangulation);
  }

  std::string parameter_file;
};

} // namespace OversetGrids
} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_EXADG_POISSON_OVERSET_GRIDS_USER_INTERFACE_APPLICATION_BASE_H_ */
