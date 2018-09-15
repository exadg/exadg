/*
 * <DEAL.SPECTRUM>/includes/bijection.h
 *
 *  Created on: Mar 02, 2018
 *      Author: muench
 */

#ifndef DEAL_SPECTRUM_BIJECTION
#define DEAL_SPECTRUM_BIJECTION

// include std
#include <immintrin.h>
#include <stdint.h>
#include <cmath>

// include DEAL.SPECTRUM modules
#include "./setup.h"

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
 * Convert lexicographical order to morton order in 2D
 *
 * source:
 * https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
 *
 * @param x     x-position in lexicographical order
 * @param y     y-position in lexicographical order
 * @return      position along morton curve
 */
uint64_t
xy_to_morton(uint32_t x, uint32_t y)
{
  return _pdep_u32(x, 0x55555555) | _pdep_u32(y, 0xaaaaaaaa);
}


/**
 * Convert morton order to lexicographical order in 2D
 *
 * source:
 * https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
 *
 * @param m     position along morton curve
 * @param x     x-position in lexicographical order
 * @param y     y-position in lexicographical order
 */
void
morton_to_xy(uint64_t m, uint32_t * x, uint32_t * y)
{
  *x = _pext_u64(m, 0x5555555555555555);
  *y = _pext_u64(m, 0xaaaaaaaaaaaaaaaa);
}

/**
 * Convert lexicographical order to morton order in 3D -> Helper function
 *
 * source:
 * http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
 *
 */
// clang-format off
inline uint64_t
splitBy3(unsigned int a)
{
  uint64_t x = a & 0x1fffff;             // we only look at the first 21 bits
  x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
  x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
  x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
  x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}
// clang-format on

/**
 * Convert lexicographical order to morton order in 3D
 *
 * source:
 * http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
 *
 * @param x     x-position in lexicographical order
 * @param y     y-position in lexicographical order
 * @param z     z-position in lexicographical order
 * @return      position along morton curve
 */
inline uint64_t
mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z)
{
  uint64_t answer = 0;
  answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
  return answer;
}

/**
 * Class for mapping between space filling curve (at the moment only morton
 * curve) and lexicographical order. It furthermore contains the partitioning
 * information of the sfc among processes.
 *
 */
class Bijection
{
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
  Bijection(Setup & s) : s(s), initialized(false)
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
    int n    = pow(s.cells, s.dim);
    int rank = s.rank;
    int size = s.size;
    init((n / size) + (rank < (n % size)));
  }

  /**
   * Initialize mapping data structures in the case that process local cell
   * count is given, e.g. due prescribed domain decomposition of the given
   * application.
   *
   * @param localCells    process local number of cells
   */
  void
  init(int localCells)
  {
    // check if already initialized
    if(this->initialized)
      return;
    this->initialized = true;

    // extract info from setup and determine helper variables
    int dim  = s.dim;       // dimension
    int size = s.size;      // mpi-size
    int rank = s.rank;      // mpi-rank
    int n    = s.cells;     // nr. of cells in each direcion
    int nc   = pow(n, dim); // overall cell count

    // step 1: determine mapping sfc <-> lex
    {
      // allocate memory for data structures
      _indices     = new int[nc];
      _indices_inv = new int[nc];
      _lbf         = new int[nc * dim];
      int * Y      = new int[dim];

      // loop over all dofs ...
      for(int counter = 0; counter < nc; counter++)
      {
        // ... determine position of dof
        int temp_counter = counter;
        for(int d = 0; d < dim; d++)
        {
          int r = temp_counter % n;
          temp_counter /= n;
          Y[d] = r + 0.5;
        }

        // ... save position
        for(int d = 0; d < dim; d++)
          _lbf[counter * dim + d] = Y[d];

        // ... get position of dof on hilbert curve
        unsigned int test = 0;

        if(dim == 2)
          test = xy_to_morton(Y[0], Y[1]);
        else
          test = mortonEncode_magicbits(Y[0], Y[1], Y[2]);

        // ... save map and reverse function
        _indices[test]        = counter; // sfc -> lex
        _indices_inv[counter] = test;    // lex -> sfc
      }

      // clean up
      delete[] Y;
    }

    // step 2: determine partitioning of cells from local cell counts
    {
      // allocate memory for data structures
      _indices_proc = new int[nc];

      // collect cell count on each process
      int * global_elements = new int[size];
      MPI_Allgather(&localCells, 1, MPI_INTEGER, global_elements, 1, MPI_INTEGER, MPI_COMM_WORLD);

      // mark all cells with information on which process it can be found
      for(int i = 0, c = 0; i < size; i++)
        for(int j = 0; j < global_elements[i]; j++, c++)
          _indices_proc[c] = i;

      // determine process local range: start index...
      this->start = 0;

      for(int i = 0; i < rank; i++)
        this->start += global_elements[i];

      // ... and end index
      this->end = this->start + global_elements[rank];

      // clean up
      delete global_elements;
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
