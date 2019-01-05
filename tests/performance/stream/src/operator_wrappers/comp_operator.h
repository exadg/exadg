#ifndef COMP_OPERATOR
#define COMP_OPERATOR

namespace CompNS
{
template<int dim>
struct CombinedOperatorData
{
  CombinedOperatorData() : dof_index(0), quad_index(0), use_cell_based_face_loops(false)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  bool use_cell_based_face_loops;
};

template<int dim, int fe_degree, int n_q_points_1d, typename value_type>
class CombinedOperator
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef CombinedOperator<dim, fe_degree, n_q_points_1d, value_type> This;

  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>       FEEval_scalar;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, value_type>   FEFaceEval_scalar;
  typedef FEEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type>     FEEval_vectorial;
  typedef FEFaceEvaluation<dim, fe_degree, n_q_points_1d, dim, value_type> FEFaceEval_vectorial;

  typedef VectorizedArray<value_type>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<value_type>> vector;
  typedef Tensor<2, dim, VectorizedArray<value_type>> tensor;
  typedef Point<dim, VectorizedArray<value_type>>     point;

  CombinedOperator() : data(nullptr)
  {
  }

  void
  initialize(MatrixFree<dim, value_type> const & mf_data,
             CombinedOperatorData<dim> const &   operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src) const
  {
    if(this->operator_data.use_cell_based_face_loops)
      data->cell_loop(&This::cell_based_loop, this, dst, src);
    else
      data->loop(&This::cell_loop, &This::face_loop, &This::boundary_face_loop, this, dst, src);
  }

private:
  void
  cell_based_loop(MatrixFree<dim, value_type> const &           data,
                  VectorType &                                  dst,
                  VectorType const &                            src,
                  std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_scalar    fe_eval_density(data, operator_data.dof_index, operator_data.quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, operator_data.dof_index, operator_data.quad_index, 1);
    FEEval_scalar fe_eval_energy(data, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    FEFaceEval_scalar fe_eval_m_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_scalar fe_eval_p_density(
      data, false, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_vectorial fe_eval_m_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_vectorial fe_eval_p_momentum(
      data, false, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_scalar fe_eval_m_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);
    FEFaceEval_scalar fe_eval_p_energy(
      data, false, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_density.reinit(cell);
      fe_eval_density.read_dof_values(src);

      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.read_dof_values(src);

      fe_eval_energy.reinit(cell);
      fe_eval_energy.read_dof_values(src);

      // work on quadrature points

      fe_eval_density.distribute_local_to_global(dst);
      fe_eval_momentum.distribute_local_to_global(dst);
      fe_eval_energy.distribute_local_to_global(dst);

      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        const auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        const auto bid  = bids[0];

        fe_eval_m_density.reinit(cell, face);
        fe_eval_m_density.read_dof_values(src);

        if(bid == numbers::internal_face_boundary_id)
        {
          fe_eval_p_density.reinit(cell, face);
          fe_eval_p_density.read_dof_values(src);
        }

        fe_eval_m_momentum.reinit(cell, face);
        fe_eval_m_momentum.read_dof_values(src);

        if(bid == numbers::internal_face_boundary_id)
        {
          fe_eval_p_momentum.reinit(cell, face);
          fe_eval_p_momentum.read_dof_values(src);
        }

        fe_eval_m_energy.reinit(cell, face);
        fe_eval_m_energy.read_dof_values(src);

        if(bid == numbers::internal_face_boundary_id)
        {
          fe_eval_p_energy.reinit(cell, face);
          fe_eval_p_energy.read_dof_values(src);
        }

        // work on quadrature points

        fe_eval_m_density.distribute_local_to_global(dst);
        fe_eval_m_momentum.distribute_local_to_global(dst);
        fe_eval_m_energy.distribute_local_to_global(dst);
      }
    }
  }

  void
  cell_loop(MatrixFree<dim, value_type> const &           data,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_scalar    fe_eval_density(data, operator_data.dof_index, operator_data.quad_index, 0);
    FEEval_vectorial fe_eval_momentum(data, operator_data.dof_index, operator_data.quad_index, 1);
    FEEval_scalar fe_eval_energy(data, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_density.reinit(cell);
      fe_eval_density.read_dof_values(src);

      fe_eval_momentum.reinit(cell);
      fe_eval_momentum.read_dof_values(src);

      fe_eval_energy.reinit(cell);
      fe_eval_energy.read_dof_values(src);

      // work on quadrature points

      fe_eval_density.distribute_local_to_global(dst);
      fe_eval_momentum.distribute_local_to_global(dst);
      fe_eval_energy.distribute_local_to_global(dst);
    }
  }

  void
  face_loop(MatrixFree<dim, value_type> const &           data,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_scalar fe_eval_m_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_scalar fe_eval_p_density(
      data, false, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_vectorial fe_eval_m_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_vectorial fe_eval_p_momentum(
      data, false, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_scalar fe_eval_m_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);
    FEFaceEval_scalar fe_eval_p_energy(
      data, false, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_m_density.reinit(face);
      fe_eval_m_density.read_dof_values(src);

      fe_eval_p_density.reinit(face);
      fe_eval_p_density.read_dof_values(src);

      fe_eval_m_momentum.reinit(face);
      fe_eval_m_momentum.read_dof_values(src);

      fe_eval_p_momentum.reinit(face);
      fe_eval_p_momentum.read_dof_values(src);

      fe_eval_m_energy.reinit(face);
      fe_eval_m_energy.read_dof_values(src);

      fe_eval_p_energy.reinit(face);
      fe_eval_p_energy.read_dof_values(src);

      // work on quadrature points

      fe_eval_m_density.distribute_local_to_global(dst);
      fe_eval_p_density.distribute_local_to_global(dst);

      fe_eval_m_momentum.distribute_local_to_global(dst);
      fe_eval_p_momentum.distribute_local_to_global(dst);

      fe_eval_m_energy.distribute_local_to_global(dst);
      fe_eval_p_energy.distribute_local_to_global(dst);
    }
  }

  void
  boundary_face_loop(MatrixFree<dim, value_type> const &           data,
                     VectorType &                                  dst,
                     VectorType const &                            src,
                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_scalar fe_eval_m_density(
      data, true, operator_data.dof_index, operator_data.quad_index, 0);
    FEFaceEval_vectorial fe_eval_m_momentum(
      data, true, operator_data.dof_index, operator_data.quad_index, 1);
    FEFaceEval_scalar fe_eval_m_energy(
      data, true, operator_data.dof_index, operator_data.quad_index, 1 + dim);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_m_density.reinit(face);
      fe_eval_m_density.read_dof_values(src);

      fe_eval_m_momentum.reinit(face);
      fe_eval_m_momentum.read_dof_values(src);

      fe_eval_m_energy.reinit(face);
      fe_eval_m_energy.read_dof_values(src);

      // work on quadrature points

      fe_eval_m_density.distribute_local_to_global(dst);
      fe_eval_m_momentum.distribute_local_to_global(dst);
      fe_eval_m_energy.distribute_local_to_global(dst);
    }
  }

  MatrixFree<dim, value_type> const * data;
  CombinedOperatorData<dim>           operator_data;
};
} // namespace CompNS


#endif