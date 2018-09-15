/*
 * InverseMassMatrix.h
 *
 *  Created on: July 14, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_INVERSEMASSMATRIXXWALL_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_INVERSEMASSMATRIXXWALL_H_

#include "../../incompressible_navier_stokes/infrastructure/fe_evaluation_wrapper.h"
#include "../../incompressible_navier_stokes/infrastructure/fe_parameters.h"
#include "operators/InverseMassMatrix.h"

// Collect all data for the inverse mass matrix operation in a struct in order to avoid allocating
// the memory repeatedly.
template<int dim,
         int fe_degree,
         int fe_degree_xwall,
         int n_q_points_1d,
         typename Number,
         int n_components = dim>
struct InverseMassMatrixXWallData
  : public InverseMassMatrixData<dim, fe_degree, Number, n_components>
{
  InverseMassMatrixXWallData(const MatrixFree<dim, Number> & data,
                             FEParameters<dim> &             fe_param,
                             const unsigned int              fe_index   = 0,
                             const unsigned int              quad_index = 0)
    : InverseMassMatrixData<dim, fe_degree, Number, n_components>(data, fe_index, quad_index),
      fe_eval_scalar(
        1,
        FEEvaluationWrapper<dim, fe_degree, fe_degree_xwall, n_q_points_1d, 1, Number, true>(
          data,
          &fe_param,
          fe_index)),
      fe_eval_components(1,
                         FEEvaluationWrapper<dim,
                                             fe_degree,
                                             fe_degree_xwall,
                                             n_q_points_1d,
                                             n_components,
                                             Number,
                                             true>(data, &fe_param, fe_index)),
      vector_result(0)
  {
  }

  // Manually implement the copy operator because CellwiseInverseMassMatrix must point to the object
  // 'fe_eval'
  InverseMassMatrixXWallData(const InverseMassMatrixXWallData & other)
    : InverseMassMatrixData<dim, fe_degree, Number, n_components>(other),
      fe_eval_scalar(other.fe_eval_scalar),
      fe_eval_components(other.fe_eval_components),
      vector_result(other.vector_result)
  {
  }

  AlignedVector<
    FEEvaluationWrapper<dim, fe_degree, fe_degree_xwall, n_q_points_1d, 1, Number, true>>
    fe_eval_scalar;
  AlignedVector<
    FEEvaluationWrapper<dim, fe_degree, fe_degree_xwall, n_q_points_1d, n_components, Number, true>>
                 fe_eval_components;
  Vector<Number> vector_result;
};

template<int dim,
         int fe_degree,
         int fe_degree_xwall,
         int n_q_points_1d,
         typename value_type,
         int n_components = dim>
class InverseMassMatrixXWallOperator
  : public InverseMassMatrixOperator<dim, fe_degree, value_type, n_components>
{
public:
  InverseMassMatrixXWallOperator()
    : InverseMassMatrixOperator<dim, fe_degree, value_type, n_components>()
  {
  }

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             FEParameters<dim> &                 fe_param,
             const unsigned int                  dof_index,
             const unsigned int                  quad_index)
  {
    InverseMassMatrixOperator<dim, fe_degree, value_type, n_components>::initialize(mf_data,
                                                                                    dof_index,
                                                                                    quad_index);
    // initialize matrices and compute them
    matrices.resize(mf_data.n_macro_cells());

    this->mass_matrix_data.reset();

    // generate initial mass matrix data to avoid allocating it over and over again
    this->mass_matrix_data_xwall.reset(
      new Threads::ThreadLocalStorage<InverseMassMatrixXWallData<dim,
                                                                 fe_degree,
                                                                 fe_degree_xwall,
                                                                 n_q_points_1d,
                                                                 value_type,
                                                                 n_components>>(
        InverseMassMatrixXWallData<dim,
                                   fe_degree,
                                   fe_degree_xwall,
                                   n_q_points_1d,
                                   value_type,
                                   n_components>(
          *(this->matrix_free_data), fe_param, dof_index, quad_index)));
  }

  void
  reinit()
  {
    parallel::distributed::Vector<value_type> dummy;
    this->matrix_free_data->cell_loop(
      &InverseMassMatrixXWallOperator<dim,
                                      fe_degree,
                                      fe_degree_xwall,
                                      n_q_points_1d,
                                      value_type,
                                      n_components>::local_precompute_mass_matrix,
      this,
      dummy,
      dummy);
  }

  void
  local_apply_inverse_mass_matrix(unsigned int                        cell,
                                  VectorizedArray<value_type> *       dst,
                                  const VectorizedArray<value_type> * src) const
  {
    InverseMassMatrixXWallData<dim,
                               fe_degree,
                               fe_degree_xwall,
                               n_q_points_1d,
                               value_type,
                               n_components> & mass_data = this->mass_matrix_data_xwall->get();
    mass_data.fe_eval_components[0].reinit(cell);
    if(not mass_data.fe_eval_components[0].enriched)
      mass_data.fe_eval[0].reinit(cell);
    // first, check if we have an enriched element
    // if so, perform the routine for the enriched elements
    if(mass_data.fe_eval_components[0].enriched)
    {
      mass_data.vector_result.reinit(mass_data.fe_eval_components[0].dofs_per_cell, true);
      for(unsigned int j = 0; j < mass_data.fe_eval_components[0].dofs_per_cell * dim; ++j)
        mass_data.fe_eval_components[0].write_cellwise_dof_value(j, src[j]);

      // now apply vectors to factorized matrix
      for(unsigned int idim = 0; idim < n_components; ++idim)
      {
        for(unsigned int v = 0; v < this->matrix_free_data->n_components_filled(cell); ++v)
        {
          mass_data.vector_result = 0;
          for(unsigned int i = 0; i < mass_data.fe_eval_components[0].dofs_per_cell; ++i)
            mass_data.vector_result[i] =
              mass_data.fe_eval_components[0].read_cellwise_dof_value(i, idim)[v];
          (matrices[cell][v]).solve(mass_data.vector_result, false);
          for(unsigned int i = 0; i < mass_data.fe_eval_components[0].dofs_per_cell; ++i)
            mass_data.fe_eval_components[0].write_cellwise_dof_value(i,
                                                                     idim,
                                                                     mass_data.vector_result[i],
                                                                     v);
        }
      }
      for(unsigned int j = 0; j < mass_data.fe_eval_components[0].dofs_per_cell * dim; ++j)
        dst[j] = mass_data.fe_eval_components[0].read_cellwise_dof_value(j);
    }
    else // perform the cheap way, if none of the elements are enriched
    {
      mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
      mass_data.inverse.apply(mass_data.coefficients, n_components, src, dst);
    }
  }

private:
  mutable std::shared_ptr<Threads::ThreadLocalStorage<InverseMassMatrixXWallData<dim,
                                                                                 fe_degree,
                                                                                 fe_degree_xwall,
                                                                                 n_q_points_1d,
                                                                                 value_type,
                                                                                 n_components>>>
                                                             mass_matrix_data_xwall;
  AlignedVector<AlignedVector<LAPACKFullMatrix<value_type>>> matrices;
  void
  local_apply_inverse_mass_matrix(const MatrixFree<dim, value_type> &,
                                  parallel::distributed::Vector<value_type> &       dst,
                                  const parallel::distributed::Vector<value_type> & src,
                                  const std::pair<unsigned int, unsigned int> & cell_range) const
  {
    InverseMassMatrixXWallData<dim,
                               fe_degree,
                               fe_degree_xwall,
                               n_q_points_1d,
                               value_type,
                               n_components> & mass_data = this->mass_matrix_data_xwall->get();

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // first, check if we have an enriched element
      // if so, perform the routine for the enriched elements
      mass_data.fe_eval_components[0].reinit(cell);
      if(mass_data.fe_eval_components[0].enriched)
      {
        mass_data.vector_result.reinit(mass_data.fe_eval_components[0].dofs_per_cell, true);
        mass_data.fe_eval_components[0].read_dof_values(src);

        // now apply vectors to factorized matrix
        for(unsigned int idim = 0; idim < n_components; ++idim)
        {
          for(unsigned int v = 0; v < this->matrix_free_data->n_components_filled(cell); ++v)
          {
            mass_data.vector_result = 0;
            for(unsigned int i = 0; i < mass_data.fe_eval_components[0].dofs_per_cell; ++i)
              mass_data.vector_result[i] =
                mass_data.fe_eval_components[0].read_cellwise_dof_value(i, idim)[v];
            (matrices[cell][v]).solve(mass_data.vector_result, false);
            for(unsigned int i = 0; i < mass_data.fe_eval_components[0].dofs_per_cell; ++i)
              mass_data.fe_eval_components[0].write_cellwise_dof_value(i,
                                                                       idim,
                                                                       mass_data.vector_result[i],
                                                                       v);
          }
        }
        mass_data.fe_eval_components[0].set_dof_values(dst);
      }
      else // perform the cheap way, if none of the elements are enriched
      {
        mass_data.fe_eval[0].reinit(cell);
        mass_data.fe_eval[0].read_dof_values(src, 0);

        mass_data.inverse.fill_inverse_JxW_values(mass_data.coefficients);
        mass_data.inverse.apply(mass_data.coefficients,
                                n_components,
                                mass_data.fe_eval[0].begin_dof_values(),
                                mass_data.fe_eval[0].begin_dof_values());

        mass_data.fe_eval[0].set_dof_values(dst, 0);
      }
    }
  }

  void
  local_precompute_mass_matrix(const MatrixFree<dim, value_type> & data,
                               parallel::distributed::Vector<value_type> &,
                               const parallel::distributed::Vector<value_type> &,
                               const std::pair<unsigned int, unsigned int> & cell_range)
  {
    InverseMassMatrixXWallData<dim,
                               fe_degree,
                               fe_degree_xwall,
                               n_q_points_1d,
                               value_type,
                               n_components> & mass_data = this->mass_matrix_data_xwall->get();

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // first, check if we have an enriched element
      // if so, perform the routine for the enriched elements
      mass_data.fe_eval_scalar[0].reinit(cell);

      if(mass_data.fe_eval_scalar[0].enriched)
      {
        if(matrices[cell].size() == 0)
          matrices[cell].resize(data.n_components_filled(cell));

        for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        {
          if(matrices[cell][v].m() != mass_data.fe_eval_scalar[0].dofs_per_cell)
            matrices[cell][v].reinit(mass_data.fe_eval_scalar[0].dofs_per_cell,
                                     mass_data.fe_eval_scalar[0].dofs_per_cell); // = onematrix;
          else
            matrices[cell][v] = 0;
        }
        for(unsigned int j = 0; j < mass_data.fe_eval_scalar[0].dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < mass_data.fe_eval_scalar[0].dofs_per_cell; ++i)
            mass_data.fe_eval_scalar[0].write_cellwise_dof_value(i, make_vectorized_array(0.));
          mass_data.fe_eval_scalar[0].write_cellwise_dof_value(j, make_vectorized_array(1.));

          mass_data.fe_eval_scalar[0].evaluate(true, false, false);
          for(unsigned int q = 0; q < mass_data.fe_eval_scalar[0].n_q_points; ++q)
          {
            mass_data.fe_eval_scalar[0].submit_value(mass_data.fe_eval_scalar[0].get_value(q), q);
          }
          mass_data.fe_eval_scalar[0].integrate(true, false);

          for(unsigned int i = 0; i < mass_data.fe_eval_scalar[0].dofs_per_cell; ++i)
            for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
              if(mass_data.fe_eval_scalar[0].component_enriched(v))
              {
                (matrices[cell][v])(i, j) =
                  (mass_data.fe_eval_scalar[0].read_cellwise_dof_value(i))[v];
              }
              else // this is a non-enriched element
              {
                if(i < mass_data.fe_eval_scalar[0].std_dofs_per_cell &&
                   j < mass_data.fe_eval_scalar[0].std_dofs_per_cell)
                  (matrices[cell][v])(i, j) =
                    (mass_data.fe_eval_scalar[0].read_cellwise_dof_value(i))[v];
                else if(i == j) // diagonal
                  (matrices[cell][v])(i, j) = 1.0;
              }
        }

        for(unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        {
          //          std::cout << "is enriched: " <<
          //          mass_data.fe_eval_scalar[0].component_enriched(v) << " vector component: " <<
          //          v << std::endl;
          //          (matrices[cell][v]).print_formatted(std::cout,3,true,0,"x",1.,0.);
          (matrices[cell][v]).compute_lu_factorization();
        }
      }
    }
  }
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_INVERSEMASSMATRIXXWALL_H_ */
