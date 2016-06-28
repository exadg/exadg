/*
 * FEEvaluationWrapper.h
 *
 *  Created on: May 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_FEEVALUATIONWRAPPER_H_
#define INCLUDE_FEEVALUATIONWRAPPER_H_

#include <deal.II/lac/parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "FE_Parameters.h"


/*
template <int dim, int fe_degree, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
	  int n_components_ = 1, typename Number = double, bool is_enriched = false>
  struct FEEvaluationTemplates
  {
    static const int dimension = dim;
    static const int fe_degree = fe_degree;
      ...
    typedef Number value_type;
  };
*/

//template <typename Template>
template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
	    int n_components_ = 1, typename Number = double, bool is_enriched = false>
class FEEvaluationWrapper : public FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>
{
private:
  static unsigned int find_quadrature_slot(const MatrixFree<dim,Number> & mf, const int quad_no)
  {
    unsigned int quad_index = 0;
    if(quad_no < 0)
    {
      const unsigned int n_q_points = std::pow(n_q_points_1d,dim);
      for ( ; quad_index < mf.get_mapping_info().data_cells.size(); quad_index++)
      {
        if (mf.get_mapping_info().data_cells[quad_index].n_q_points[0] == n_q_points)
          break;
      }
    }
    else
      quad_index = (unsigned int)quad_no;
    return quad_index;
  }

public:
  FEEvaluationWrapper (
  const MatrixFree<dim,Number> &matrix_free,
  const FEParameters & in_fe_param,
  const unsigned int            fe_no = 0,
  const int            quad_no = -1)
    :
    //    FEEvaluation<Templates::dimension,....,dim,fe_degree,n_q_points_1d,n_components_,Number>(matrix_free,fe_no,find_quadrature_slot(matrix_free,quad_no)),
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>(matrix_free,fe_no,find_quadrature_slot(matrix_free,quad_no)),
    fe_param(in_fe_param)
  {
    std_dofs_per_cell = FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::dofs_per_cell;
  }

  //read and write access functions
  //see definition of begin_dof_values
  VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
  {
    return FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::begin_dof_values()[j];
  }
  void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::begin_dof_values()[j][v] = value;
  }
  void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::begin_dof_values()[j] = value;
    return;
  }
  using FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values;

  void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &)
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values(src);
  }
  void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i, unsigned int )
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values(src,i);
  }
  void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i, unsigned int)
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values(src,i);
  }

  using FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global;

  void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &)
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global(dst);
  }
  void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i, unsigned int )
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global(dst,i);
  }
  void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i, unsigned int )
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global(dst,i);
  }

  using FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::set_dof_values;

  void set_dof_values (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::set_dof_values(dst);
  }
  void set_dof_values (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i, unsigned int)
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::set_dof_values(dst,i);
  }
  void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int cell)
  {
    eddyvisc.resize(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::n_q_points,make_vectorized_array(fe_param.viscosity));
  }
  Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<2,dim,VectorizedArray<Number> > symgrad;
    for (unsigned int i = 0; i<dim; i++)
      for (unsigned int j = 0; j<dim; j++)
        symgrad[i][j] =  grad[i][j] + grad[j][i];
    symgrad *= make_vectorized_array<Number>(0.5);
    return symgrad;
  }
  Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<1,dim,VectorizedArray<Number> > symgrad;
    Assert(false, ExcInternalError());
    return symgrad;
  }
  bool component_enriched(unsigned int)
  {
    return false;
  }

  AlignedVector<VectorizedArray<Number> > eddyvisc;
  unsigned int std_dofs_per_cell;
  const FEParameters & fe_param;
};

template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
            int n_components_ = 1, typename Number = double, bool is_enriched = false>
class FEFaceEvaluationWrapper : public FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>
{
private:
  static unsigned int find_quadrature_slot(const MatrixFree<dim,Number> & mf, const int quad_no)
  {
    unsigned int quad_index = 0;
    if(quad_no < 0)
    {
      const unsigned int n_q_points = std::pow(n_q_points_1d,dim);
      for ( ; quad_index < mf.get_mapping_info().data_cells.size(); quad_index++)
      {
        if (mf.get_mapping_info().data_cells[quad_index].n_q_points[0] == n_q_points)
          break;
      }
    }
    else
      quad_index = (unsigned int)quad_no;
    return quad_index;
  }
public:
  FEFaceEvaluationWrapper (
  const MatrixFree<dim,Number> &matrix_free,
  const FEParameters & in_fe_param,
  const bool                    is_left_face = true,
  const unsigned int            fe_no = 0,
  const int            quad_no = -1)
    :
  FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>(matrix_free,is_left_face,fe_no,find_quadrature_slot(matrix_free,quad_no)),
  fe_param(in_fe_param)
  {
    std_dofs_per_cell = FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::dofs_per_cell;
  }

  //read and write access functions
  //see definition of begin_dof_values
  VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
  {
    return FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::begin_dof_values()[j];
  }
  void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::begin_dof_values()[j][v] = value;
  }
  void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::begin_dof_values()[j] = value;
    return;
  }
  using FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values;

  void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &)
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values(src);
  }
  void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i, unsigned int )
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values(src,i);
  }
  void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i, unsigned int)
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::read_dof_values(src,i);
  }

  using FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global;

  void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &)
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global(dst);
  }
  void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i, unsigned int )
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global(dst,i);
  }
  void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i, unsigned int )
  {
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global(dst,i);
  }
  void evaluate_eddy_viscosity(const parallel::distributed::BlockVector<double> &velocity_n, unsigned int face, const VectorizedArray<Number> volume)
  {
    eddyvisc.resize(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::n_q_points,make_vectorized_array(fe_param.viscosity));
  }
  Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<2,dim,VectorizedArray<Number> > symgrad;
    for (unsigned int i = 0; i<dim; i++)
      for (unsigned int j = 0; j<dim; j++)
        symgrad[i][j] = grad[i][j] + grad[j][i];
    symgrad *= make_vectorized_array<Number>(0.5);
    return symgrad;
  }
  Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<1,dim,VectorizedArray<Number> > symgrad;
    // symmetric gradient is not defined in that case
    Assert(false, ExcInternalError());
    return symgrad;
  }
  bool component_enriched(unsigned int)
  {
    return false;
  }

  AlignedVector<VectorizedArray<Number> > eddyvisc;
  unsigned int std_dofs_per_cell;
  const FEParameters & fe_param;
};

#endif /* INCLUDE_FEEVALUATIONWRAPPER_H_ */
