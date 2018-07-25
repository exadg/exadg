#ifndef OPERATOR_BASE_INTERPOLATE
#define OPERATOR_BASE_INTERPOLATE

namespace dealii {
namespace MGTools {

template <int dim, int spacedim, typename VectorType,
          template <int, int> class DoFHandlerType>
void interpolate(
    const DoFHandlerType<dim, spacedim> &dof,
    const Function<spacedim, typename VectorType::value_type> &function,
    VectorType &vec, unsigned level = numbers::invalid_unsigned_int) {

  if (level == numbers::invalid_unsigned_int) {
    // is not multigrid
    VectorTools::interpolate(dof, function, vec);
  } else {
    auto start = dof.begin_mg(level);
    auto end = dof.end_mg(level);
    auto &fe = dof.get_fe();

    Quadrature<dim> quadrature(fe.get_generalized_support_points());

    FEValues<dim> fe_values(dof.get_fe(), quadrature, update_quadrature_points);

    // loop over all local cells
    for (auto cell1 = start; cell1 < end; cell1++)
      if (cell1->is_locally_owned_on_level()) {
        fe_values.reinit(cell1);

        Vector<typename VectorType::value_type> coefficient_list(
            quadrature.size());
        std::vector<double> coefficient_list_(quadrature.size());
        function.value_list(fe_values.get_quadrature_points(),
                            coefficient_list_);
        for (unsigned int i = 0; i < coefficient_list_.size(); i++)
          coefficient_list[i] = coefficient_list_[i];

        std::vector<types::global_dof_index> dof_indices_dg(fe.dofs_per_cell);
        cell1->get_mg_dof_indices(dof_indices_dg);
        cell1->set_dof_indices(dof_indices_dg);
        //                        for(int o : dof_indices_dg)
        //                            std::cout << o << std::endl;

        for (unsigned int i = 0; i < dof_indices_dg.size(); i++)
          vec[dof_indices_dg[i]] = coefficient_list[i];

        //                        cell1->distribute_local_to_global(coefficient_list,
        //                        vec);
      }
  }
}
}
}

#endif