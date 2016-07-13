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
#include "SpaldingsLaw.h"


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

  using FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global;

  using FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::set_dof_values;

  void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int cell)
  {
    eddyvisc.resize(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::n_q_points,make_vectorized_array<Number>(fe_param.viscosity));
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
    AssertThrow(false, ExcNotImplemented());
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

  using FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::distribute_local_to_global;

  void evaluate_eddy_viscosity(const parallel::distributed::BlockVector<double> &velocity_n, unsigned int face, const VectorizedArray<Number> volume)
  {
    eddyvisc.resize(FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number>::n_q_points,make_vectorized_array<Number>(fe_param.viscosity));
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
    AssertThrow(false, ExcNotImplemented());
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







template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d,
      int n_components_, typename Number>
class FEEvaluationWrapper<dim, fe_degree, fe_degree_xwall, n_q_points_1d, n_components_, Number, true>
{
private:
  typedef FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
  typedef Number                            number_type;
  typedef typename BaseClass::value_type    value_type;
  typedef typename BaseClass::gradient_type gradient_type;

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
  data(matrix_free),
  fe_param(in_fe_param),
  spalding(fe_param.viscosity),
  fe_eval_q0(matrix_free,fe_no,find_quadrature_slot(matrix_free, quad_no),0),
  fe_eval_q1(matrix_free,fe_no,find_quadrature_slot(matrix_free, quad_no),0),
  fe_eval(),
  fe_eval_xwall_q0(matrix_free,fe_no,find_quadrature_slot(matrix_free, quad_no),dim),
  fe_eval_xwall_q1(matrix_free,fe_no,find_quadrature_slot(matrix_free, quad_no),dim),
  fe_eval_xwall(),
  fe_eval_tauw_q0(matrix_free,2,find_quadrature_slot(matrix_free, quad_no)),
  fe_eval_tauw_q1(matrix_free,2,find_quadrature_slot(matrix_free, quad_no)),
  fe_eval_tauw(),
  values(),
  gradients(),
  std_dofs_per_cell(0),
  dofs_per_cell(0),
  tensor_dofs_per_cell(0),
  n_q_points(0),
  enriched(false),
  quad_type(0)
  {
    // use non-linear quadrature rule for now for all terms... small speed-up possible through better choice of standard quadrature rule
    // (via additional template argument)
    fe_eval.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,false>* >(&fe_eval_q0));
    fe_eval.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,false>* >(&fe_eval_q1));

    fe_eval_xwall.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,false>* >(&fe_eval_xwall_q0));
    fe_eval_xwall.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,false>* >(&fe_eval_xwall_q1));

    fe_eval_tauw.push_back(dynamic_cast<FEEvaluationAccess<dim,1,Number,false>* >(&fe_eval_tauw_q0));
    fe_eval_tauw.push_back(dynamic_cast<FEEvaluationAccess<dim,1,Number,false>* >(&fe_eval_tauw_q1));
  };

  void reinit(const unsigned int cell)
  {
    {
      enriched = false;
      quad_type = 0; //0: standard quadrature rule, 1: high-order quadrature rule
//        decide if we have an enriched element via the y component of the cell center
      for (unsigned int v=0; v<data.n_components_filled(cell); ++v)
      {
        typename DoFHandler<dim>::cell_iterator dcell = data.get_cell_iterator(cell, v);
//            std::cout << ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL))) << std::endl;
        if ((dcell->center()[1] > (1.0-fe_param.max_wdist_xwall)) || (dcell->center()[1] <(-1.0 + fe_param.max_wdist_xwall)))
        {
          enriched = true;
          quad_type = 1;
        }
      }
      if(quad_type == 0)
        n_q_points = fe_eval_q0.n_q_points;
      else if(quad_type == 1)
        n_q_points = fe_eval_q1.n_q_points;
      else
        AssertThrow(false,ExcMessage("only 0 or 1 allowed"));
      values.resize(n_q_points,value_type());
      gradients.resize(n_q_points,gradient_type());

      enriched_components.resize(VectorizedArray<Number>::n_array_elements);
      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        enriched_components.at(v) = false;
      if(enriched)
      {
        //store, exactly which component of the vectorized array is enriched
        for (unsigned int v=0; v<data.n_components_filled(cell); ++v)
        {
          typename DoFHandler<dim>::cell_iterator dcell = data.get_cell_iterator(cell, v);
          if ((dcell->center()[1] > (1.0-fe_param.max_wdist_xwall)) || (dcell->center()[1] <(-1.0 + fe_param.max_wdist_xwall)))
              enriched_components.at(v) = true;
        }

        //initialize the enrichment function
        {
          fe_eval_tauw[quad_type]->reinit(cell);
          //get wall distance and wss at quadrature points
          fe_eval_tauw[quad_type]->read_dof_values(fe_param.wdist);
          fe_eval_tauw_evaluate(true, true);

          AlignedVector<VectorizedArray<Number> > cell_wdist;
          AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradwdist;
          cell_wdist.resize(n_q_points);
          cell_gradwdist.resize(n_q_points);
          for(unsigned int q=0;q<n_q_points;++q)
          {
            cell_wdist[q] = fe_eval_tauw[quad_type]->get_value(q);
            cell_gradwdist[q] = fe_eval_tauw[quad_type]->get_gradient(q);
          }

          fe_eval_tauw[quad_type]->read_dof_values(fe_param.tauw);

          fe_eval_tauw_evaluate(true, true);

          AlignedVector<VectorizedArray<Number> > cell_tauw;
          AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradtauw;

          cell_tauw.resize(n_q_points);
          cell_gradtauw.resize(n_q_points);

          for(unsigned int q=0;q<n_q_points;++q)
          {
            cell_tauw[q] = fe_eval_tauw[quad_type]->get_value(q);
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            {
              if(enriched_components.at(v))
                AssertThrow( fe_eval_tauw[quad_type]->get_value(q)[v] > 1.0e-9 ,ExcMessage("Wall shear stress has to be above zero for Spalding's law"));
            }

            cell_gradtauw[q] = fe_eval_tauw[quad_type]->get_gradient(q);
          }
          spalding.reinit(cell_wdist, cell_tauw, cell_gradwdist, cell_gradtauw, n_q_points,enriched_components);
        }
      }
      fe_eval_xwall[quad_type]->reinit(cell);
    }
    fe_eval[quad_type]->reinit(cell);
    std_dofs_per_cell = fe_eval_q0.dofs_per_cell;
    if(enriched)
    {
      dofs_per_cell = fe_eval_q0.dofs_per_cell + fe_eval_xwall_q0.dofs_per_cell;
      tensor_dofs_per_cell = fe_eval_q0.tensor_dofs_per_cell + fe_eval_xwall_q0.tensor_dofs_per_cell;
    }
    else
    {
      dofs_per_cell = fe_eval_q0.dofs_per_cell;
      tensor_dofs_per_cell = fe_eval_q0.tensor_dofs_per_cell;
    }
  }

  VectorizedArray<Number> * begin_dof_values() DEAL_II_DEPRECATED
  {
    return fe_eval[quad_type]->begin_dof_values();
  }

  void read_dof_values (const parallel::distributed::Vector<Number> &src)
  {
    fe_eval[quad_type]->read_dof_values(src);
    fe_eval_xwall[quad_type]->read_dof_values(src);
  }

  void evaluate(const bool evaluate_val,
                const bool evaluate_grad,
                const bool evaluate_hess = false)
  {
    AssertThrow(evaluate_hess == false, ExcNotImplemented());
    fe_eval_evaluate(evaluate_val,evaluate_grad);
    if(enriched)
    {
      gradients.resize(n_q_points,gradient_type());
      values.resize(n_q_points,value_type());
      fe_eval_xwall_evaluate(true,evaluate_grad);
      //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
      //evaluate gradient
      if(evaluate_grad)
      {
        gradient_type submitgradient = gradient_type();
        gradient_type gradient = gradient_type();
        //there are 2 parts due to chain rule
        for(unsigned int q=0;q<n_q_points;++q)
        {
          submitgradient = gradient_type();
          gradient = fe_eval_xwall[quad_type]->get_gradient(q)*spalding.enrichment(q);
          val_enrgrad_to_grad(gradient, q);
          //delete enrichment part where not needed
          //this is essential, code won't work otherwise
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            if(enriched_components.at(v))
              add_array_component_to_gradient(submitgradient,gradient,v);
          gradients[q] = submitgradient;
        }
      }
      if(evaluate_val)
      {
        for(unsigned int q=0;q<n_q_points;++q)
        {
          value_type finalvalue = fe_eval_xwall[quad_type]->get_value(q)*spalding.enrichment(q);
          value_type submitvalue = value_type();
          //delete enrichment part where not needed
          //this is essential, code won't work otherwise
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            if(enriched_components.at(v))
              add_array_component_to_value(submitvalue,finalvalue,v);
          values[q]=submitvalue;
        }
      }
    }
  }

  void val_enrgrad_to_grad(Tensor<2,dim,VectorizedArray<Number> >& grad, unsigned int q)
  {
    for(unsigned int j=0;j<dim;++j)
    {
      for(unsigned int i=0;i<dim;++i)
      {
        grad[j][i] += fe_eval_xwall[quad_type]->get_value(q)[j]*spalding.enrichment_gradient(q)[i];
      }
    }
  }
  void val_enrgrad_to_grad(Tensor<1,dim,VectorizedArray<Number> >& grad, unsigned int q)
  {
    for(unsigned int i=0;i<dim;++i)
    {
      grad[i] += fe_eval_xwall[quad_type]->get_value(q)*spalding.enrichment_gradient(q)[i];
    }
  }

  void submit_value(const value_type val_in,
      const unsigned int q_point)
  {
    fe_eval[quad_type]->submit_value(val_in,q_point);
    if(enriched)
      values[q_point] = val_in;
    else
      values[q_point] = value_type();
  }

  void submit_value(const Tensor<1,1,VectorizedArray<Number> > val_in,
      const unsigned int q_point)
  {
    fe_eval[quad_type]->submit_value(val_in[0],q_point);
    if(enriched)
      values[q_point] = val_in[0];
    else
      values[q_point] = value_type();
  }

  void submit_gradient(const gradient_type grad_in,
      const unsigned int q_point)
  {
    fe_eval[quad_type]->submit_gradient(grad_in,q_point);
    if(enriched)
      gradients[q_point] = grad_in;
    else
      gradients[q_point] = gradient_type();
  }

  value_type get_value(const unsigned int q_point)
  {
    if(enriched)
      return values[q_point] + fe_eval[quad_type]->get_value(q_point);
    // else
    return fe_eval[quad_type]->get_value(q_point);
  }

  gradient_type get_gradient (const unsigned int q_point)
  {
    if(enriched)
      return fe_eval[quad_type]->get_gradient(q_point) + gradients[q_point];
    // else
    return fe_eval[quad_type]->get_gradient(q_point);
  }

  gradient_type get_symmetric_gradient (const unsigned int q_point)
  {
    return make_symmetric(get_gradient(q_point));
  }

  Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<2,dim,VectorizedArray<Number> > symgrad;
    for (unsigned int i = 0; i<dim; i++)
      for (unsigned int j = 0; j<dim; j++)
        symgrad[i][j] =  grad[i][j] + grad[j][i];
    return symgrad;
  }

  Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<1,dim,VectorizedArray<Number> > symgrad;
    AssertThrow(false, ExcNotImplemented());
    return symgrad;
  }

  void integrate (const bool integrate_val,
                  const bool integrate_grad)
  {
    {
      if(enriched)
      {
        AlignedVector<value_type> tmp_values(n_q_points,value_type());
        if(integrate_val)
          for(unsigned int q=0;q<n_q_points;++q)
            tmp_values[q]=values[q]*spalding.enrichment(q);
        //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
        //the scalar product of the second part of the gradient is computed directly and added to the value
        if(integrate_grad)
        {
          //first, zero out all non-enriched vectorized array components
          grad_enr_to_val(tmp_values, gradients);

          for(unsigned int q=0;q<n_q_points;++q)
            fe_eval_xwall[quad_type]->submit_gradient(gradients[q]*spalding.enrichment(q),q);
        }

        for(unsigned int q=0;q<n_q_points;++q)
          fe_eval_xwall[quad_type]->submit_value(tmp_values[q],q);
        //integrate
        fe_eval_integrate(true,integrate_grad);
      }
    }
    fe_eval_xwall_integrate(integrate_val, integrate_grad);
  }

  void distribute_local_to_global (parallel::distributed::Vector<Number> &dst)
  {
    fe_eval[quad_type]->distribute_local_to_global(dst);
    if(enriched)
      fe_eval_xwall[quad_type]->distribute_local_to_global(dst);
  }

  void set_dof_values (parallel::distributed::Vector<Number> &dst)
  {
    fe_eval[quad_type]->set_dof_values(dst);
    if(enriched)
      fe_eval_xwall[quad_type]->set_dof_values(dst);
  }

  void fill_JxW_values(AlignedVector<VectorizedArray<Number> > &JxW_values) const
  {
    fe_eval[quad_type]->fill_JxW_values(JxW_values);
  }

  Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
  {
    if(quad_type == 0)
      return fe_eval_q0.quadrature_point(q);

    // else if(quad_type == 1)
      return fe_eval_q1.quadrature_point(q);
  }

    VectorizedArray<Number> get_divergence(unsigned int q)
  {
    if(enriched)
    {
      VectorizedArray<Number> div_enr= make_vectorized_array<Number>(0.0);
      for (unsigned int i=0;i<dim;i++)
        div_enr += gradients[q][i][i];
      return fe_eval[quad_type]->get_divergence(q) + div_enr;
    }
    // else
    return fe_eval[quad_type]->get_divergence(q);
  }

  Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
  get_curl (const unsigned int q_point) const
  {
    if(enriched)
    {
      // copy from generic function into dim-specialization function
      const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
      Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
      switch (dim)
        {
        case 1:
          AssertThrow (false,
                  ExcMessage("Computing the curl in 1d is not a useful operation"));
          break;
        case 2:
          curl[0] = grad[1][0] - grad[0][1];
          break;
        case 3:
          curl[0] = grad[2][1] - grad[1][2];
          curl[1] = grad[0][2] - grad[2][0];
          curl[2] = grad[1][0] - grad[0][1];
          break;
        default:
          AssertThrow (false, ExcNotImplemented());
          break;
        }
      return fe_eval[quad_type]->get_curl(q_point) + curl;
    }
    // else
    return fe_eval[quad_type]->get_curl(q_point);
  }

  VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
  {
    if(enriched)
    {
      VectorizedArray<Number> returnvalue = make_vectorized_array<Number>(0.0);
      if(j<std_dofs_per_cell*n_components_)
        returnvalue =  fe_eval[quad_type]->begin_dof_values()[j];
      else
      {
        returnvalue = fe_eval_xwall[quad_type]->begin_dof_values()[j-std_dofs_per_cell*n_components_];
      }
      return returnvalue;
    }
    // else
    return fe_eval[quad_type]->begin_dof_values()[j];
  }
  void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
  {
    if(enriched)
    {
      if(j<std_dofs_per_cell*n_components_)
        fe_eval[quad_type]->begin_dof_values()[j][v] = value;
      else
        fe_eval_xwall[quad_type]->begin_dof_values()[j-std_dofs_per_cell*n_components_][v] = value;
    }
    else
      fe_eval[quad_type]->begin_dof_values()[j][v]=value;
    return;
  }
  void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
  {
    if(enriched)
    {
      if(j<std_dofs_per_cell*n_components_)
        fe_eval[quad_type]->begin_dof_values()[j] = value;
      else
        fe_eval_xwall[quad_type]->begin_dof_values()[j-std_dofs_per_cell*n_components_] = value;
    }
    else
      fe_eval[quad_type]->begin_dof_values()[j]=value;
    return;
  }
  bool component_enriched(unsigned int v)
  {
    if(not enriched)
      return false;
    // else
    return enriched_components.at(v);
  }

  void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int cell)
  {
    eddyvisc.resize(n_q_points);
    if(fe_param.cs > 1e-10)
    {
      const VectorizedArray<Number> Cs = make_vectorized_array<Number>(fe_param.cs);
      VectorizedArray<Number> hfac = make_vectorized_array<Number>(1.0/(Number)fe_degree);
      fe_eval_tauw[quad_type]->reinit(cell);
      {
        VectorizedArray<Number> volume = make_vectorized_array<Number>((Number)0.);
        {
          AlignedVector<VectorizedArray<Number> > JxW_values;
          JxW_values.resize(fe_eval_tauw[quad_type]->n_q_points);
          fe_eval_tauw[quad_type]->fill_JxW_values(JxW_values);
          for (unsigned int q=0; q<fe_eval_tauw[quad_type]->n_q_points; ++q)
            volume += JxW_values[q];
        }
        reinit(cell);
        read_dof_values(solution_n,0,solution_n,dim+1);
        evaluate (false,true,false);
        AlignedVector<VectorizedArray<Number> > wdist;
        wdist.resize(fe_eval_tauw[quad_type]->n_q_points);
        fe_eval_tauw[quad_type]->read_dof_values(fe_param.wdist);
        fe_eval_tauw[quad_type]->evaluate(true,false,false);
        for (unsigned int q=0; q<fe_eval_tauw[quad_type]->n_q_points; ++q)
          wdist[q] = fe_eval_tauw[quad_type]->get_value(q);
        fe_eval_tauw[quad_type]->reinit(cell);
        fe_eval_tauw[quad_type]->read_dof_values(fe_param.tauw);
        fe_eval_tauw[quad_type]->evaluate(true,false,false);

        const VectorizedArray<Number> hvol = std::pow(volume, 1./(double)dim) * hfac;

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

          VectorizedArray<Number> snorm = make_vectorized_array<Number>((Number)0.);
          for (unsigned int i = 0; i<dim ; i++)
            for (unsigned int j = 0; j<dim ; j++)
              snorm += (s[i][j])*(s[i][j]);
          snorm *= make_vectorized_array<Number>(0.5);
          //simple wall correction
          VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/fe_param.viscosity*std::sqrt(fe_eval_tauw[quad_type]->get_value(q))*0.04));
          VectorizedArray<Number> lm = Cs*hvol*fmu;
          eddyvisc[q]= make_vectorized_array<Number>(fe_param.viscosity) + lm*lm*std::sqrt(snorm);
        }
      }
      //initialize again to get a clean version
      reinit(cell);
    }
    else if (fe_param.ml>0.1)//&& enriched)
    {
      fe_eval_tauw[quad_type]->reinit(cell);
      {
        VectorizedArray<Number> h ;
        {
          VectorizedArray<Number> hfac = make_vectorized_array<Number>(1.0/(Number)fe_degree);
          VectorizedArray<Number> volume = make_vectorized_array<Number>((Number)0.);
          AlignedVector<VectorizedArray<Number> > JxW_values;
          JxW_values.resize(fe_eval_tauw[quad_type]->n_q_points);
          fe_eval_tauw[quad_type]->fill_JxW_values(JxW_values);
          for (unsigned int q=0; q<fe_eval_tauw[quad_type]->n_q_points; ++q)
            volume += JxW_values[q];
          h=std::exp(std::log(volume)/3.)*hfac;
        }
        read_dof_values(solution_n,0,solution_n,dim+1);
        evaluate (false,true,false);
        AlignedVector<VectorizedArray<Number> > wdist;
        AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > derwdist;
        wdist.resize(fe_eval_tauw[quad_type]->n_q_points);
        derwdist.resize(fe_eval_tauw[quad_type]->n_q_points);
        fe_eval_tauw[quad_type]->read_dof_values(fe_param.wdist);
        fe_eval_tauw[quad_type]->evaluate(true,true,false);
        for (unsigned int q=0; q<fe_eval_tauw[quad_type]->n_q_points; ++q)
        {
          wdist[q] = fe_eval_tauw[quad_type]->get_value(q);
          derwdist[q] = fe_eval_tauw[quad_type]->get_gradient(q);
          VectorizedArray<Number> sum = std::sqrt(derwdist[q]*derwdist[q]);//normalize to 1
          derwdist[q] /= sum;
        }
        fe_eval_tauw[quad_type]->reinit(cell);
        fe_eval_tauw[quad_type]->read_dof_values(fe_param.tauw);
        fe_eval_tauw[quad_type]->evaluate(true,false,false);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<Number> > du = get_gradient(q);
          Tensor<1,dim,VectorizedArray<Number> > dudy = du * derwdist[q];
          VectorizedArray<Number> dudynorm = std::sqrt(dudy*dudy);

          //simple wall correction
          VectorizedArray<Number> yp = wdist[q]/fe_param.viscosity*std::sqrt(fe_eval_tauw[quad_type]->get_value(q));
          const double HMIN = 0.6;
          const double KAPPA = 0.41;
          const double Ainv = 1./30.;
          VectorizedArray<Number> lssst=KAPPA*wdist[q]*(1.-std::exp(-yp*Ainv))*std::min(2.*std::exp(-9.*(0.25-wdist[q]/h*HMIN)*(0.25-wdist[q]/h*HMIN)),make_vectorized_array<Number>((Number)1.));
          VectorizedArray<Number> vt = lssst*lssst*dudynorm;
          //VectorizedArray<Number> l = KAPPA*std::min(wdist[q],HMIN*h)*(1.-std::exp(-yp*Ainv));
          //VectorizedArray<Number> vt = l*l*dudynorm;
          //VectorizedArray<Number> lssstl=KAPPA*wdist[q]*(1.-std::exp(-yp*Ainv));
          //VectorizedArray<Number> vt = lssst*lssstl*dudynorm;
          //VectorizedArray<Number> vtch = 0.41*std::sqrt(fe_eval_tauw[0].get_value(q))*std::min(wdist[q],1.5*h)*(1.-std::exp(-yp*0.05))*(1.-std::exp(-yp*0.05));
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            //vt = std::min(vt,make_vectorized_array(38.7*VISCOSITY));
           // if(enriched_components.at(v))
            //if(wdist[q][v]<MAX_ML_DIST)
            {
            //  if(yp[v]>1.)
                //vt[v] *= std::exp(-BETA*std::pow(wdist[q][v]/h[v],EXPON));
             // if(wdist[q][v] > MAX_ML_DIST)
             //   vt[v] *= 1.-(wdist[q][v]-MAX_ML_DIST)/(MAX_WDIST_XWALL-MAX_ML_DIST);
              eddyvisc[q][v]= fe_param.viscosity + vt[v];
            }
           // else
           //   eddyvisc[q][v]= VISCOSITY;
          }
        }
      }
      //initialize again to get a clean version
      reinit(cell);
  }
    else
      for (unsigned int q=0; q<n_q_points; ++q)
        eddyvisc[q]= make_vectorized_array<Number>(fe_param.viscosity);

    return;
  }
private:
  // some heler-functions
  void fe_eval_evaluate(bool evaluate_val, bool evaluate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_q0.evaluate(evaluate_val, evaluate_grad);
    }
    else
      fe_eval_q1.evaluate(evaluate_val, evaluate_grad);
  }
  void fe_eval_xwall_evaluate(bool evaluate_val, bool evaluate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_xwall_q0.evaluate(evaluate_val, evaluate_grad);
    }
    else
      fe_eval_xwall_q1.evaluate(evaluate_val, evaluate_grad);
  }
  void fe_eval_tauw_evaluate(bool evaluate_val, bool evaluate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_tauw_q0.evaluate(evaluate_val, evaluate_grad);
    }
    else
      fe_eval_tauw_q1.evaluate(evaluate_val, evaluate_grad);
  }
  void fe_eval_integrate(bool integrate_val, bool integrate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_q0.integrate(integrate_val, integrate_grad);
    }
    else
      fe_eval_q1.integrate(integrate_val, integrate_grad);
  }
  void fe_eval_xwall_integrate(bool integrate_val, bool integrate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_xwall_q0.integrate(integrate_val, integrate_grad);
    }
    else
      fe_eval_xwall_q1.integrate(integrate_val, integrate_grad);
  }

  void add_array_component_to_value(VectorizedArray<Number>& val,const VectorizedArray<Number>& toadd, unsigned int v)
  {
    val[v] += toadd[v];
  }
  void add_array_component_to_value(Tensor<1,n_components_,VectorizedArray<Number> >& val,const Tensor<1,n_components_,VectorizedArray<Number> >& toadd, unsigned int v)
  {
    for (unsigned int d = 0; d<n_components_; d++)
      val[d][v] += toadd[d][v];
  }

  void add_array_component_to_gradient(Tensor<2,dim,VectorizedArray<Number> >& grad,const Tensor<2,dim,VectorizedArray<Number> >& toadd, unsigned int v)
  {
    for (unsigned int comp = 0; comp<dim; comp++)
      for (unsigned int d = 0; d<dim; d++)
        grad[comp][d][v] += toadd[comp][d][v];
  }
  void add_array_component_to_gradient(Tensor<1,dim,VectorizedArray<Number> >& grad,const Tensor<1,dim,VectorizedArray<Number> >& toadd, unsigned int v)
  {
    for (unsigned int d = 0; d<n_components_; d++)
      grad[d][v] += toadd[d][v];
  }

  void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
  {
    for(unsigned int q=0;q<n_q_points;++q)
    {
      for(int j=0; j<dim;++j)//comp
      {
        for(int i=0; i<dim;++i)//dim
        {
          tmp_values[q][j] += gradient[q][j][i]*spalding.enrichment_gradient(q)[i];
        }
      }
    }
  }
  void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
  {
    for(unsigned int q=0;q<n_q_points;++q)
    {
      for(int i=0; i<dim;++i)//dim
      {
        tmp_values[q] += gradient[q][i]*spalding.enrichment_gradient(q)[i];
      }
    }
  }

  const MatrixFree<dim,Number> data;
  const FEParameters & fe_param;
  SpaldingsLawEvaluation<dim,n_q_points_1d, Number, VectorizedArray<Number> > spalding;
  FEEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,n_components_,Number> fe_eval_q0;
  FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval_q1;
  AlignedVector<FEEvaluationAccess<dim,n_components_,Number,false>*> fe_eval;
  FEEvaluation<dim,fe_degree_xwall,fe_degree+(fe_degree+2)/2,n_components_,Number> fe_eval_xwall_q0;
  FEEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall_q1;
  AlignedVector<FEEvaluationAccess<dim,n_components_,Number,false>* > fe_eval_xwall;
  FEEvaluation<dim,1,fe_degree+(fe_degree+2)/2,1,Number> fe_eval_tauw_q0;
  FEEvaluation<dim,1,n_q_points_1d,1,Number> fe_eval_tauw_q1;
  AlignedVector<FEEvaluationAccess<dim,1,Number,false>* > fe_eval_tauw;
  AlignedVector<value_type> values;
  AlignedVector<gradient_type> gradients;

public:
  unsigned int std_dofs_per_cell;
  unsigned int dofs_per_cell;
  unsigned int tensor_dofs_per_cell;
  unsigned int n_q_points;
  bool enriched;
  unsigned int quad_type;
  std::vector<bool> enriched_components;
  AlignedVector<VectorizedArray<Number> > eddyvisc;
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d,
            int n_components_, typename Number>
class FEFaceEvaluationWrapper<dim, fe_degree, fe_degree_xwall, n_q_points_1d, n_components_, Number, true>
{
public:
  typedef Number                            number_type;
private:
  typedef FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
  typedef typename BaseClass::value_type    value_type;
  typedef typename BaseClass::gradient_type gradient_type;

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
  FEFaceEvaluationWrapper  ( const MatrixFree<dim,Number> &matrix_free,
  const FEParameters & in_fe_param,
  const bool                    is_left_face = true,
  const unsigned int            fe_no = 0,
  const int            quad_no = -1):
    data(matrix_free),
    fe_param(in_fe_param),
    spalding(fe_param.viscosity),
    fe_eval_q0(matrix_free,is_left_face,fe_no,find_quadrature_slot(matrix_free, quad_no),0),
    fe_eval_q1(matrix_free,is_left_face,fe_no,find_quadrature_slot(matrix_free, quad_no),0),
    fe_eval(),
    fe_eval_xwall_q0(matrix_free,is_left_face,fe_no,find_quadrature_slot(matrix_free, quad_no),dim),
    fe_eval_xwall_q1(matrix_free,is_left_face,fe_no,find_quadrature_slot(matrix_free, quad_no),dim),
    fe_eval_xwall(),
    fe_eval_tauw_q0(matrix_free,is_left_face,2,find_quadrature_slot(matrix_free, quad_no)),
    fe_eval_tauw_q1(matrix_free,is_left_face,2,find_quadrature_slot(matrix_free, quad_no)),
    fe_eval_tauw(),
    is_left_face(is_left_face),
    values(),
    gradients(),
    std_dofs_per_cell(0),
    dofs_per_cell(0),
    tensor_dofs_per_cell(0),
    n_q_points(0),
    enriched(false),
    quad_type(0)
  {
    fe_eval.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,true>*>(&fe_eval_q0));
    fe_eval.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,true>*>(&fe_eval_q1));

    fe_eval_xwall.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,true>*>(&fe_eval_xwall_q0));
    fe_eval_xwall.push_back(dynamic_cast<FEEvaluationAccess<dim,n_components_,Number,true>*>(&fe_eval_xwall_q1));

    fe_eval_tauw.push_back(dynamic_cast<FEEvaluationAccess<dim,1,Number,true>*>(&fe_eval_tauw_q0));
    fe_eval_tauw.push_back(dynamic_cast<FEEvaluationAccess<dim,1,Number,true>*>(&fe_eval_tauw_q1));
  };

  void reinit(const unsigned int f)
  {


    enriched = false;
    quad_type = 0;
    if(is_left_face)
    {
//        decide if we have an enriched element via the y component of the cell center
      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
        data.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
      {
        typename DoFHandler<dim>::cell_iterator dcell =  data.get_cell_iterator(
            data.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
            data.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
            if ((dcell->center()[1] > (1.0-fe_param.max_wdist_xwall)) || (dcell->center()[1] <(-1.0 + fe_param.max_wdist_xwall)))
            {
              enriched = true;
              quad_type = 1;
            }
      }
    }
    else
    {
      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
        data.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
      {
        typename DoFHandler<dim>::cell_iterator dcell =  data.get_cell_iterator(
            data.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
            data.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
            if ((dcell->center()[1] > (1.0-fe_param.max_wdist_xwall)) || (dcell->center()[1] <(-1.0 + fe_param.max_wdist_xwall)))
            {
              enriched = true;
              quad_type = 1;
            }
      }
    }
    if(quad_type == 0)
      n_q_points = fe_eval_q0.n_q_points;
    else if(quad_type == 1)
      n_q_points = fe_eval_q1.n_q_points;
    values.resize(n_q_points,value_type());
    gradients.resize(n_q_points,gradient_type());
    enriched_components.resize(VectorizedArray<Number>::n_array_elements);
    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      enriched_components.at(v) = false;
    if(enriched)
    {
      //store, exactly which component of the vectorized array is enriched
      if(is_left_face)
      {
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
        data.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
        {
          typename DoFHandler<dim>::cell_iterator dcell =  data.get_cell_iterator(
              data.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
              data.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
              if ((dcell->center()[1] > (1.0-fe_param.max_wdist_xwall)) || (dcell->center()[1] <(-1.0 + fe_param.max_wdist_xwall)))
                enriched_components.at(v)=(true);
        }
      }
      else
      {
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
        data.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
        {
          typename DoFHandler<dim>::cell_iterator dcell =  data.get_cell_iterator(
              data.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
              data.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
              if ((dcell->center()[1] > (1.0-fe_param.max_wdist_xwall)) || (dcell->center()[1] <(-1.0 + fe_param.max_wdist_xwall)))
                enriched_components.at(v)=(true);
        }
      }

      AssertThrow(enriched_components.size()==VectorizedArray<Number>::n_array_elements,ExcInternalError());

      //initialize the enrichment function
      {
        fe_eval_tauw[quad_type]->reinit(f);
        //get wall distance and wss at quadrature points
        fe_eval_tauw[quad_type]->read_dof_values(fe_param.wdist);
        fe_eval_tauw_evaluate(true, true);

        AlignedVector<VectorizedArray<Number> > face_wdist;
        AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradwdist;
        face_wdist.resize(n_q_points);
        face_gradwdist.resize(n_q_points);
        for(unsigned int q=0;q<n_q_points;++q)
        {
          face_wdist[q] = fe_eval_tauw[quad_type]->get_value(q);
          face_gradwdist[q] = fe_eval_tauw[quad_type]->get_gradient(q);
        }

        fe_eval_tauw[quad_type]->read_dof_values(fe_param.tauw);
        fe_eval_tauw_evaluate(true, true);
        AlignedVector<VectorizedArray<Number> > face_tauw;
        AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradtauw;
        face_tauw.resize(n_q_points);
        face_gradtauw.resize(n_q_points);
        for(unsigned int q=0;q<n_q_points;++q)
        {
          face_tauw[q] = fe_eval_tauw[quad_type]->get_value(q);
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(enriched_components.at(v))
              AssertThrow( fe_eval_tauw[quad_type]->get_value(q)[v] > 1.0e-9 ,ExcInternalError());
          }

          face_gradtauw[q] = fe_eval_tauw[quad_type]->get_gradient(q);
        }
        spalding.reinit(face_wdist, face_tauw, face_gradwdist, face_gradtauw, n_q_points,enriched_components);
      }
    }
    fe_eval_xwall[quad_type]->reinit(f);

    fe_eval[quad_type]->reinit(f);
    if(enriched)
    {
      dofs_per_cell = fe_eval_q0.dofs_per_cell + fe_eval_xwall_q0.dofs_per_cell;
      tensor_dofs_per_cell = fe_eval_q0.tensor_dofs_per_cell + fe_eval_xwall_q0.tensor_dofs_per_cell;
    }
    else
    {
      dofs_per_cell = fe_eval_q0.dofs_per_cell;
      tensor_dofs_per_cell = fe_eval_q0.tensor_dofs_per_cell;
    }
  }

  void read_dof_values (const parallel::distributed::Vector<Number> &src)
  {
    fe_eval[quad_type]->read_dof_values(src);
    fe_eval_xwall[quad_type]->read_dof_values(src);
  }

  void evaluate(const bool evaluate_val,
             const bool evaluate_grad,
             const bool evaluate_hess = false)
  {
    AssertThrow(evaluate_hess == false, ExcNotImplemented());
    fe_eval_evaluate(evaluate_val,evaluate_grad);
    if(enriched)
    {
      gradients.resize(n_q_points,gradient_type());
      values.resize(n_q_points,value_type());
      fe_eval_xwall_evaluate(true,evaluate_grad);
      //evaluate gradient
      if(evaluate_grad)
      {
        //there are 2 parts due to chain rule
        gradient_type gradient = gradient_type();
        gradient_type submitgradient = gradient_type();
        for(unsigned int q=0;q<n_q_points;++q)
        {
          submitgradient = gradient_type();
          gradient = fe_eval_xwall[quad_type]->get_gradient(q)*spalding.enrichment(q);
          val_enrgrad_to_grad(gradient,q);
          //delete enrichment part where not needed
          //this is essential, code won't work otherwise
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            if(enriched_components.at(v))
              add_array_component_to_gradient(submitgradient,gradient,v);

          gradients[q] = submitgradient;
        }
      }
      if(evaluate_val)
      {
        for(unsigned int q=0;q<n_q_points;++q)
        {
          value_type finalvalue = fe_eval_xwall[quad_type]->get_value(q)*spalding.enrichment(q);
          value_type submitvalue = value_type();
          //delete enrichment part where not needed
          //this is essential, code won't work otherwise
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
            if(enriched_components.at(v))
              add_array_component_to_value(submitvalue,finalvalue,v);
          values[q]=submitvalue;
        }
      }
    }
  }

  void submit_value(const value_type val_in,
      const unsigned int q_point)
  {
    fe_eval[quad_type]->submit_value(val_in,q_point);
    if(enriched)
      values[q_point] = val_in;
    else
      values[q_point] = value_type();
  }

  void submit_gradient(const gradient_type grad_in,
      const unsigned int q_point)
  {
    fe_eval[quad_type]->submit_gradient(grad_in,q_point);
    if(enriched)
      gradients[q_point] = grad_in;
    else
      gradients[q_point] = gradient_type();
  }

  value_type get_value(const unsigned int q_point)
  {
    if(enriched)
      return fe_eval[quad_type]->get_value(q_point) + values[q_point];//fe_eval[quad_type]->get_value(q_point) + values[q_point];
    // else
    return fe_eval[quad_type]->get_value(q_point);
  }

  gradient_type get_gradient (const unsigned int q_point)
  {
    if(enriched)
      return fe_eval[quad_type]->get_gradient(q_point) + gradients[q_point];
    // else
    return fe_eval[quad_type]->get_gradient(q_point);
  }

  gradient_type get_symmetric_gradient (const unsigned int q_point)
  {
    return make_symmetric(get_gradient(q_point));
  }

  Tensor<2,dim,VectorizedArray<Number> > make_symmetric(const Tensor<2,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<2,dim,VectorizedArray<Number> > symgrad;
    for (unsigned int i = 0; i<dim; i++)
      for (unsigned int j = 0; j<dim; j++)
        symgrad[i][j] = grad[i][j] + grad[j][i];
    return symgrad;
  }

  Tensor<1,dim,VectorizedArray<Number> > make_symmetric(const Tensor<1,dim,VectorizedArray<Number> >& grad)
  {
    Tensor<1,dim,VectorizedArray<Number> > symgrad;
    // symmetric gradient is not defined in that case
    AssertThrow(false, ExcNotImplemented());
    return symgrad;
  }

  VectorizedArray<Number> get_divergence(unsigned int q)
  {
    if(enriched)
    {
      VectorizedArray<Number> div_enr= make_vectorized_array<Number>((Number)0.0);
      for (unsigned int i=0;i<dim;i++)
        div_enr += gradients[q][i][i];
      return fe_eval[quad_type]->get_divergence(q) + div_enr;
    }
    // else
    return fe_eval[quad_type]->get_divergence(q);
  }

  Tensor<1,dim,VectorizedArray<Number> > get_normal_vector(const unsigned int q_point) const
  {
    return fe_eval[quad_type]->get_normal_vector(q_point);
  }

  void integrate (const bool integrate_val,
                  const bool integrate_grad)
  {
    if(enriched)
    {
      AlignedVector<value_type> tmp_values(n_q_points,value_type());
      if(integrate_val)
        for(unsigned int q=0;q<n_q_points;++q)
          tmp_values[q]=values[q]*spalding.enrichment(q);
      //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
      //the scalar product of the second part of the gradient is computed directly and added to the value
      if(integrate_grad)
      {
        grad_enr_to_val(tmp_values,gradients);
        for(unsigned int q=0;q<n_q_points;++q)
          fe_eval_xwall[quad_type]->submit_gradient(gradients[q]*spalding.enrichment(q),q);
      }

      for(unsigned int q=0;q<n_q_points;++q)
        fe_eval_xwall[quad_type]->submit_value(tmp_values[q],q);
      //integrate
      fe_eval_xwall_integrate(true,integrate_grad);
    }
    fe_eval_integrate(integrate_val, integrate_grad);
  }

  void distribute_local_to_global (parallel::distributed::Vector<Number> &dst)
  {
    fe_eval[quad_type]->distribute_local_to_global(dst);
    if(enriched)
      fe_eval_xwall[quad_type]->distribute_local_to_global(dst);
  }

  Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
  {
    if(quad_type == 0)
      return fe_eval_q0.quadrature_point(q);
    // else if(quad_type == 1)
    return fe_eval_q1.quadrature_point(q);
  }

  VectorizedArray<Number> get_normal_volume_fraction()
  {
    return fe_eval[quad_type]->get_normal_volume_fraction();
  }

  VectorizedArray<Number> read_cell_data(const AlignedVector<VectorizedArray<Number> > &cell_data)
  {
    return fe_eval[quad_type]->read_cell_data(cell_data);
  }

  Tensor<1,n_components_,VectorizedArray<Number> > get_normal_gradient(const unsigned int q_point) const
  {
    if(enriched)
    {
      Tensor<1,n_components_,VectorizedArray<Number> > grad_out;
      for (unsigned int comp=0; comp<n_components_; comp++)
      {
        grad_out[comp] = gradients[q_point][comp][0] *
                         fe_eval[quad_type]->get_normal_vector(q_point)[0];
        for (unsigned int d=1; d<dim; ++d)
          grad_out[comp] += gradients[q_point][comp][d] *
                           fe_eval[quad_type]->get_normal_vector(q_point)[d];
      }
      return fe_eval[quad_type]->get_normal_gradient(q_point) + grad_out;
    }
    //else
    return fe_eval[quad_type]->get_normal_gradient(q_point);
  }

  VectorizedArray<Number> get_normal_gradient(const unsigned int q_point,bool test) const
  {
    if(enriched)
    {
      VectorizedArray<Number> grad_out;
        grad_out = gradients[q_point][0] *
                         fe_eval[quad_type]->get_normal_vector(q_point)[0];
        for (unsigned int d=1; d<dim; ++d)
          grad_out += gradients[q_point][d] *
                           fe_eval[quad_type]->get_normal_vector(q_point)[d];

        grad_out +=  fe_eval[quad_type]->get_normal_gradient(q_point);
      return grad_out;
    }
    // else
    return fe_eval[quad_type]->get_normal_gradient(q_point);
  }

  void submit_normal_gradient (const Tensor<1,n_components_,VectorizedArray<Number> > grad_in,
                            const unsigned int q)
  {
    fe_eval[quad_type]->submit_normal_gradient(grad_in,q);
    if(enriched)
    {
      for (unsigned int comp=0; comp<n_components_; comp++)
      {
        for (unsigned int d=0; d<dim; ++d)
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(enriched_components.at(v))
            {
              gradients[q][comp][d][v] = grad_in[comp][v] *
              fe_eval[quad_type]->get_normal_vector(q)[d][v];
            }
            else
              gradients[q][comp][d][v] = 0.0;
          }
      }
    }
    else
      gradients[q]=gradient_type();
  }
  void submit_normal_gradient (const VectorizedArray<Number> grad_in,
                            const unsigned int q)
  {
    fe_eval[quad_type]->submit_normal_gradient(grad_in,q);
    if(enriched)
    {
      for (unsigned int d=0; d<dim; ++d)
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        {
          if(enriched_components.at(v))
          {
            gradients[q][d][v] = grad_in[v] *
            fe_eval[quad_type]->get_normal_vector(q)[d][v];
          }
          else
            gradients[q][d][v] = 0.0;
        }
    }
    else
      gradients[q]=gradient_type();
  }

  Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
  get_curl (const unsigned int q_point) const
  {
    if(enriched)
    {
      // copy from generic function into dim-specialization function
      const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
      Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
      switch (dim)
        {
        case 1:
          AssertThrow (false,
                  ExcMessage("Computing the curl in 1d is not a useful operation"));
          break;
        case 2:
          curl[0] = grad[1][0] - grad[0][1];
          break;
        case 3:
          curl[0] = grad[2][1] - grad[1][2];
          curl[1] = grad[0][2] - grad[2][0];
          curl[2] = grad[1][0] - grad[0][1];
          break;
        default:
          AssertThrow (false, ExcNotImplemented());
          break;
        }
      return fe_eval[quad_type]->get_curl(q_point) + curl;
    }
    // else
    return fe_eval[quad_type]->get_curl(q_point);
  }

  VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
  {
    if(enriched)
    {
      VectorizedArray<Number> returnvalue = make_vectorized_array<Number>(0.0);
      if(j<std_dofs_per_cell*n_components_)
        returnvalue = fe_eval[quad_type]->begin_dof_values()[j];
      else
        returnvalue = fe_eval_xwall[quad_type]->begin_dof_values()[j-std_dofs_per_cell*n_components_];
      return returnvalue;
    }
    // else
    return fe_eval[quad_type]->begin_dof_values()[j];
  }
  void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
  {
    if(enriched)
    {
      if(j<std_dofs_per_cell*n_components_)
        fe_eval[quad_type]->begin_dof_values()[j][v] = value;
      else
        fe_eval_xwall[quad_type]->begin_dof_values()[j-std_dofs_per_cell*n_components_][v] = value;
    }
    else
      fe_eval[quad_type]->begin_dof_values()[j][v]=value;
    return;
  }
  void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
  {
    if(enriched)
    {
      if(j<std_dofs_per_cell*n_components_)
        fe_eval[quad_type]->begin_dof_values()[j] = value;
      else
        fe_eval_xwall[quad_type]->begin_dof_values()[j-std_dofs_per_cell*n_components_] = value;
    }
    else
      fe_eval[quad_type]->begin_dof_values()[j]=value;
    return;
  }
  void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int face, const VectorizedArray<Number> volume)
  {
    eddyvisc.resize(n_q_points);
    if(fe_param.cs > 1e-10)
    {
      const VectorizedArray<Number> Cs = make_vectorized_array<Number>(fe_param.cs);
      VectorizedArray<Number> hfac = make_vectorized_array<Number>(1.0/(Number)fe_degree);
      fe_eval_tauw[quad_type]->reinit(face);
      {
        reinit(face);
        read_dof_values(solution_n,0,solution_n,dim+1);
        evaluate (false,true,false);
        AlignedVector<VectorizedArray<Number> > wdist;
        wdist.resize(fe_eval_tauw[quad_type]->n_q_points);
        fe_eval_tauw[quad_type]->read_dof_values(fe_param.wdist);
        fe_eval_tauw[quad_type]->evaluate(true,false);
        for (unsigned int q=0; q<fe_eval_tauw[quad_type]->n_q_points; ++q)
          wdist[q] = fe_eval_tauw[quad_type]->get_value(q);
        fe_eval_tauw[quad_type]->reinit(face);
        fe_eval_tauw[quad_type]->read_dof_values(fe_param.tauw);
        fe_eval_tauw[quad_type]->evaluate(true,false);

        const VectorizedArray<Number> hvol = hfac * std::pow(volume, 1./(double)dim);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

          VectorizedArray<Number> snorm = make_vectorized_array<Number>((Number)0.);
          for (unsigned int i = 0; i<dim ; i++)
            for (unsigned int j = 0; j<dim ; j++)
              snorm += (s[i][j])*(s[i][j]);
          snorm *= make_vectorized_array<Number>(0.5);
          //simple wall correction
          VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/fe_param.viscosity*std::sqrt(fe_eval_tauw[quad_type]->get_value(q))*0.04));
          VectorizedArray<Number> lm = Cs*hvol*fmu;
          eddyvisc[q]= make_vectorized_array<Number>(fe_param.viscosity) + lm*lm*std::sqrt(snorm);
        }
      }
      //initialize again to get a clean version
      reinit(face);
    }
  else if (fe_param.ml>0.1)// && enriched)
  {
    fe_eval_tauw[quad_type]->reinit(face);
    VectorizedArray<Number> h ;
    {
      VectorizedArray<Number> hfac = make_vectorized_array<Number>(1.0/(Number)fe_degree);
      h=std::exp(std::log(volume)/3.)*hfac;
    }
    {
      read_dof_values(solution_n,0,solution_n,dim+1);
      evaluate (false,true,false);
      AlignedVector<VectorizedArray<Number> > wdist;
      AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > derwdist;
      wdist.resize(fe_eval_tauw[quad_type]->n_q_points);
      derwdist.resize(fe_eval_tauw[quad_type]->n_q_points);
      fe_eval_tauw[quad_type]->read_dof_values(fe_param.wdist);
      fe_eval_tauw[quad_type]->evaluate(true,true);
      for (unsigned int q=0; q<fe_eval_tauw[quad_type]->n_q_points; ++q)
      {
        wdist[q] = fe_eval_tauw[quad_type]->get_value(q);
        derwdist[q] = fe_eval_tauw[quad_type]->get_gradient(q);
        VectorizedArray<Number> sum = std::sqrt(derwdist[q]*derwdist[q]);//normalize to 1
        derwdist[q] /= sum;
      }
      fe_eval_tauw[quad_type]->reinit(face);
      fe_eval_tauw[quad_type]->read_dof_values(fe_param.tauw);
      fe_eval_tauw[quad_type]->evaluate(true,false);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
        Tensor<2,dim,VectorizedArray<Number> > du = get_gradient(q);
        Tensor<1,dim,VectorizedArray<Number> > dudy = du * derwdist[q];
        VectorizedArray<Number> dudynorm = std::sqrt(dudy*dudy);
        const double HMIN = 0.6;
        const double KAPPA = 0.41;
        const double Ainv = 1./30.;
        VectorizedArray<Number> yp = wdist[q]/fe_param.viscosity*std::sqrt(fe_eval_tauw[quad_type]->get_value(q));
        VectorizedArray<Number> lssst=KAPPA*wdist[q]*(1.-std::exp(-yp*Ainv))*std::min(2.*std::exp(-9.*(0.25-wdist[q]/h*HMIN)*(0.25-wdist[q]/h*HMIN)),make_vectorized_array<Number>((Number)1.));
        VectorizedArray<Number> vt = lssst*lssst*dudynorm;
        //VectorizedArray<Number> l = KAPPA*std::min(wdist[q],HMIN*h)*(1.-std::exp(-yp*Ainv));
        //VectorizedArray<Number> vt = l*l*dudynorm;
        //VectorizedArray<Number> lssstl = KAPPA*wdist[q]*(1.-std::exp(-yp*Ainv));
        //VectorizedArray<Number> vt = lssstl*lssst*dudynorm;
        //VectorizedArray<Number> vtch = 0.41*std::sqrt(fe_eval_tauw[quad_type]->get_value(q))*std::min(wdist[q],1.5*h)*(1.-std::exp(-yp*0.05))*(1.-std::exp(-yp*0.05));
       // eps *= 0.5*vt;
        //eps =std::pow(fe_eval_tauw[0].get_value(q),1.5)/(0.41*wdist[q]);
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        {
        //  if(enriched_components.at(v))
          {
            //vt = std::min(vt,make_vectorized_array(38.7*VISCOSITY));
            //if(yp[v]>1.)
              //vt[v] *= std::exp(-BETA*std::pow(wdist[q][v]/h[v],EXPON));
        //    if(wdist[q][v] > MAX_ML_DIST)
        //      vt[v] *= 1.-(wdist[q][v]-MAX_ML_DIST)/(MAX_WDIST_XWALL-MAX_ML_DIST);
            eddyvisc[q][v]= fe_param.viscosity + vt[v];
          }
        //  else
        //    eddyvisc[q][v]= VISCOSITY;
        }
      }
    }
    //initialize again to get a clean version
    reinit(face);
}
    else
      for (unsigned int q=0; q<n_q_points; ++q)
        eddyvisc[q]= make_vectorized_array<Number>(fe_param.viscosity);

    return;
  }

  void fill_JxW_values(AlignedVector<VectorizedArray<Number> > &JxW_values) const
  {
    fe_eval[quad_type]->fill_JxW_values(JxW_values);
  }

private:
  void fe_eval_evaluate(bool evaluate_val, bool evaluate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_q0.evaluate(evaluate_val, evaluate_grad);
    }
    else
      fe_eval_q1.evaluate(evaluate_val, evaluate_grad);
  }

  void fe_eval_xwall_evaluate(bool evaluate_val, bool evaluate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_xwall_q0.evaluate(evaluate_val, evaluate_grad);
    }
    else
      fe_eval_xwall_q1.evaluate(evaluate_val, evaluate_grad);
  }

  void fe_eval_tauw_evaluate(bool evaluate_val, bool evaluate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_tauw_q0.evaluate(evaluate_val, evaluate_grad);
    }
    else
      fe_eval_tauw_q1.evaluate(evaluate_val, evaluate_grad);
  }

  void fe_eval_integrate(bool integrate_val, bool integrate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_q0.integrate(integrate_val, integrate_grad);
    }
    else
      fe_eval_q1.integrate(integrate_val, integrate_grad);
  }

  void fe_eval_xwall_integrate(bool integrate_val, bool integrate_grad)
  {
    if(quad_type == 0)
    {
      fe_eval_xwall_q0.integrate(integrate_val, integrate_grad);
    }
    else
      fe_eval_xwall_q1.integrate(integrate_val, integrate_grad);
  }

  void val_enrgrad_to_grad(Tensor<2,dim,VectorizedArray<Number> >& grad, unsigned int q)
  {
    for(unsigned int j=0;j<dim;++j)
    {
      for(unsigned int i=0;i<dim;++i)
      {
        grad[j][i] += fe_eval_xwall[quad_type]->get_value(q)[j]*spalding.enrichment_gradient(q)[i];
      }
    }
  }

  void val_enrgrad_to_grad(Tensor<1,dim,VectorizedArray<Number> >& grad, unsigned int q)
  {
    for(unsigned int i=0;i<dim;++i)
    {
      grad[i] += fe_eval_xwall[quad_type]->get_value(q)*spalding.enrichment_gradient(q)[i];
    }
  }

  void add_array_component_to_value(VectorizedArray<Number>& val,const VectorizedArray<Number>& toadd, unsigned int v)
  {
    val[v] += toadd[v];
  }

  void add_array_component_to_value(Tensor<1,n_components_, VectorizedArray<Number> >& val,const Tensor<1,n_components_,VectorizedArray<Number> >& toadd, unsigned int v)
  {
    for (unsigned int d = 0; d<n_components_; d++)
      val[d][v] += toadd[d][v];
  }

  void add_array_component_to_gradient(Tensor<2,dim,VectorizedArray<Number> >& grad,const Tensor<2,dim,VectorizedArray<Number> >& toadd, unsigned int v)
  {
    for (unsigned int comp = 0; comp<dim; comp++)
      for (unsigned int d = 0; d<dim; d++)
        grad[comp][d][v] += toadd[comp][d][v];
  }
  void add_array_component_to_gradient(Tensor<1,dim,VectorizedArray<Number> >& grad,const Tensor<1,dim,VectorizedArray<Number> >& toadd, unsigned int v)
  {
    for (unsigned int d = 0; d<n_components_; d++)
      grad[d][v] += toadd[d][v];
  }

  void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
  {
    for(unsigned int q=0;q<n_q_points;++q)
    {

      for(int j=0; j<dim;++j)//comp
      {
        for(int i=0; i<dim;++i)//dim
        {
          tmp_values[q][j] += gradient[q][j][i]*spalding.enrichment_gradient(q)[i];
        }
      }
    }
  }
  void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
  {
    for(unsigned int q=0;q<n_q_points;++q)
    {
      for(int i=0; i<dim;++i)//dim
      {
        tmp_values[q] += gradient[q][i]*spalding.enrichment_gradient(q)[i];
      }
    }
  }

  const MatrixFree<dim,Number> data;
  const FEParameters & fe_param;
  SpaldingsLawEvaluation<dim,n_q_points_1d, Number, VectorizedArray<Number> > spalding;
  FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,n_components_,Number> fe_eval_q0;
  FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval_q1;
  AlignedVector<FEEvaluationAccess<dim,n_components_,Number,true>* > fe_eval;
  FEFaceEvaluation<dim,fe_degree_xwall,fe_degree+(fe_degree+2)/2,n_components_,Number> fe_eval_xwall_q0;
  FEFaceEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall_q1;
  AlignedVector<FEEvaluationAccess<dim,n_components_,Number,true>*> fe_eval_xwall;
  FEFaceEvaluation<dim,1,fe_degree+(fe_degree+2)/2,1,Number> fe_eval_tauw_q0;
  FEFaceEvaluation<dim,1,n_q_points_1d,1,Number> fe_eval_tauw_q1;
  AlignedVector<FEEvaluationAccess<dim,1,Number,true>*> fe_eval_tauw;
  bool is_left_face;
  AlignedVector<value_type> values;
  AlignedVector<gradient_type> gradients;


public:
  unsigned int std_dofs_per_cell;
  unsigned int dofs_per_cell;
  unsigned int tensor_dofs_per_cell;
  unsigned int n_q_points;
  bool enriched;
  unsigned int quad_type;
  std::vector<bool> enriched_components;
  AlignedVector<VectorizedArray<Number> > eddyvisc;
};


#endif /* INCLUDE_FEEVALUATIONWRAPPER_H_ */
