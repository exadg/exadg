#ifndef OPERATOR_BASE_CATEGORIZATION_H
#define OPERATOR_BASE_CATEGORIZATION_H

class Categorization
{
public:
  /**
   * Adjust MatrixFree::AdditionalData such that
   *   1) cells which have the same boundary IDs for all faces are put into the
   *      same category
   *   2) cell based loops are enabled (incl. FEEvaluationBase::read_cell_data()
   *      for all neighboring cell)
   */
  template<int dim, typename AdditionalData>
  static void
  do_cell_based_loops(const parallel::TriangulationBase<dim> & tria,
                      AdditionalData &                     data,
                      const unsigned int                   level = numbers::invalid_unsigned_int)
  {
    bool is_mg = level != numbers::invalid_unsigned_int;

    // ... create list for the category of each cell
    if(is_mg)
      data.cell_vectorization_category.resize(std::distance(tria.begin(level), tria.end(level)));
    else
      data.cell_vectorization_category.resize(tria.n_active_cells());

    // ... setup scaling factor
    std::vector<unsigned int> factors(dim * 2);

    std::map<unsigned int, unsigned int> bid_map;
    for(unsigned int i = 0; i < tria.get_boundary_ids().size(); i++)
      bid_map[tria.get_boundary_ids()[i]] = i + 1;

    {
      unsigned int bids   = tria.get_boundary_ids().size() + 1;
      int          offset = 1;
      for(unsigned int i = 0; i < dim * 2; i++, offset = offset * bids)
        factors[i] = offset;
    }

    auto to_category = [&](auto & cell) {
      unsigned int c_num = 0;
      for(unsigned int i = 0; i < dim * 2; i++)
      {
        auto & face = *cell->face(i);
        if(face.at_boundary())
          c_num += factors[i] * bid_map[face.boundary_id()];
      }
      return c_num;
    };

    if(!is_mg)
    {
      for(auto cell = tria.begin_active(); cell != tria.end(); ++cell)
      {
        if(cell->is_locally_owned())
          data.cell_vectorization_category[cell->active_cell_index()] = to_category(cell);
      }
    }
    else
    {
      for(auto cell = tria.begin(level); cell != tria.end(level); ++cell)
      {
        if(cell->is_locally_owned_on_level())
          data.cell_vectorization_category[cell->index()] = to_category(cell);
      }
    }

    // ... finalize setup of matrix_free
    data.hold_all_faces_to_owned_cells        = true;
    data.cell_vectorization_categories_strict = true;
    data.mapping_update_flags_faces_by_cells =
      data.mapping_update_flags_inner_faces | data.mapping_update_flags_boundary_faces;
  }
};

#endif
