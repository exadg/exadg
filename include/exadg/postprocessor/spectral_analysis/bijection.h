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

#ifndef DEAL_SPECTRUM_BIJECTION
#define DEAL_SPECTRUM_BIJECTION

// std
#include <immintrin.h>
#include <stdint.h>
#include <cmath>

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

// ExaDG
#include <exadg/postprocessor/spectral_analysis/setup.h>

// define helper funtions
#ifndef MIN
#  define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
#  define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

namespace dealspectrum
{
/**
 * Class for mapping between space filling curve (at the moment only morton
 * curve) and lexicographical order. It furthermore contains the partitioning
 * information of the sfc among processes.
 *
 */
class Bijection
{
  MPI_Comm const & comm;

  // reference to DEAL.SPECTRUM setup
  Setup & s;
  // is initialized?
  bool initialized;
  // process local range along sfc
  int start;
  int end;

public:
  /**
   * Constructor
   * @param s DEAL.SPECTRUM setup
   */
  Bijection(MPI_Comm const & comm, Setup & s) : comm(comm), s(s), initialized(false)
  {
  }

  /**
   * Get process local range of cells on space filling curve
   *
   * @param start     start point
   * @param end       end point
   */
  void
  getLocalRange(int & start, int & end)
  {
    start = this->start;
    end   = this->end;
  }

  /**
   * Initialize mapping data structures in the case that no process local cell
   * count is given. We distribute cells over all processes evenly:
   * #cells / #processes. The remainder cells (#cells % #processes) are
   * assigned to the first processes.
   */
  void
  init()
  {
    int n = s.cells;

    if(s.dim == 2)
    {
      dealii::parallel::distributed::Triangulation<2> triangulation(comm);
      dealii::GridGenerator::subdivided_hyper_cube(triangulation,
                                                   1,
                                                   -dealii::numbers::PI,
                                                   dealii::numbers::PI);
      triangulation.refine_global(round(log(n) / log(2)));
      init(triangulation);
    }
    else if(s.dim == 3)
    {
      dealii::parallel::distributed::Triangulation<3> triangulation(comm);
      dealii::GridGenerator::subdivided_hyper_cube(triangulation,
                                                   1,
                                                   -dealii::numbers::PI,
                                                   dealii::numbers::PI);
      triangulation.refine_global(round(log(n) / log(2)));
      init(triangulation);
    }
  }


  /**
   * Initialize mapping data structures in the case that a deal.II-triangulation
   * is given..
   *
   * @param triangulation    triangulation
   */
  template<class Tria>
  void
  init(Tria & triangulation)
  {
    // check if already initialized
    if(this->initialized)
      return;
    this->initialized = true;

    int n = s.cells;

    {
      unsigned int const n_active_cells    = triangulation.n_global_active_cells();
      unsigned int const n_active_cells_1d = std::pow(n_active_cells, 1.0 / s.dim) + 0.49;

      std::vector<int> temp_indices;

      for(auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
      {
        if(!cell->is_locally_owned())
          continue;

        double x = 1000, y = 1000, z = 1000;
        for(int v = 0; v < int(std::pow(2, s.dim)); v++)
        {
          auto vertex = cell->vertex(v);
          x           = std::min(x, vertex[0]);
          y           = std::min(y, vertex[1]);
          if(s.dim == 3)
            z = std::min(z, vertex[2]);
        }

        // domain is between -pi and +pi
        unsigned int x_index =
          int((x + dealii::numbers::PI) / (2 * dealii::numbers::PI / n_active_cells_1d) + 0.5);
        unsigned int y_index =
          int((y + dealii::numbers::PI) / (2 * dealii::numbers::PI / n_active_cells_1d) + 0.5);
        unsigned int z_index =
          s.dim == 2 ?
            0 :
            int((z + dealii::numbers::PI) / (2 * dealii::numbers::PI / n_active_cells_1d) + 0.5);

        temp_indices.push_back(n_active_cells_1d * n_active_cells_1d * z_index +
                               n_active_cells_1d * y_index + x_index);
      }

      std::vector<int> cells(s.size);
      std::vector<int> cells_sum(s.size);
      int              cells_local = temp_indices.size();

      MPI_Allgather(&cells_local, 1, MPI_INT, &cells[0], 1, MPI_INT, comm);

      cells_sum[0] = 0;
      for(int i = 1; i < s.size; i++)
        cells_sum[i] = cells_sum[i - 1] + cells[i - 1];


      std::vector<int> _indices_temp(n_active_cells);
      MPI_Allgatherv(&temp_indices[0],
                     cells_local,
                     MPI_INT,
                     &_indices_temp[0],
                     &cells[0],
                     &cells_sum[0],
                     MPI_INT,
                     comm);

      int * Y = new int[s.dim];

      _indices      = new int[n_active_cells];
      _indices_inv  = new int[n_active_cells];
      _lbf          = new int[n_active_cells * s.dim];
      _indices_proc = new int[n_active_cells];

      for(unsigned int c = 0; c < n_active_cells; c++)
      {
        // ... determine position of dof
        int counter      = _indices_temp[c];
        int temp_counter = counter;
        for(int d = 0; d < s.dim; d++)
        {
          int r = temp_counter % n;
          temp_counter /= n;
          Y[d] = r + 0.5;
        }

        // ... save position
        for(int d = 0; d < s.dim; d++)
          _lbf[counter * s.dim + d] = Y[d];

        _indices[c]           = counter;
        _indices_inv[counter] = c;
      }

      for(int i = 0, c = 0; i < s.size; i++)
        for(int j = 0; j < cells[i]; j++, c++)
          _indices_proc[c] = i;


      // determine process local range: start index...
      this->start = 0;

      for(int i = 0; i < s.rank; i++)
        this->start += cells[i];

      // ... and end index
      this->end = this->start + cells[s.rank];

      delete[] Y;
    }
  }

  /**
   * Destructor
   */
  virtual ~Bijection()
  {
    // not initialized -> nothing to clean up
    if(!initialized)
      return;

    delete[] _indices_proc;
    delete[] _lbf;
    delete[] _indices_inv;
    delete[] _indices;
  }

  /**
   * Mapping sfc -> lexicographical ordering
   *
   * @param i     position on sfc
   * @return      position in lex. ordering
   */
  inline int
  indices(int i)
  {
    return _indices[i];
  }

  /**
   * Mapping lexicographical ordering -> sfc
   *
   * @param i     position in lex. ordering
   * @return      position on sfc
   */
  inline int
  indices_inv(int i)
  {
    return _indices_inv[i];
  }

  /**
   * Determines rank of process owning specified cell
   *
   * @param i     position on sfc
   * @return      rank of process owning cell
   */
  inline int
  indices_proc(int i)
  {
    return _indices_proc[i];
  }

  /**
   * Determines left bottom front points of cell
   *
   * @param i     index = lex-cell*dim+dir
   * @return      left bottom front point component of cell
   */
  inline int
  lbf(int i)
  {
    return _lbf[i];
  }

private:
  // mapping: sfc -> lex
  int * _indices;
  // mapping: lex -> sfc
  int * _indices_inv;
  // left bottom front point of cells
  int * _lbf;
  // process owning cell
  int * _indices_proc;
};

} // namespace dealspectrum

#endif
