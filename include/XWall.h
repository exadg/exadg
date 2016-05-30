/*
 * XWall.h
 *
 *  Created on: May 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_XWALL_H_
#define INCLUDE_XWALL_H_

struct SimpleSpaldingsLaw
{
  static double SpaldingsLaw(double dist, double utau, double viscosity)
  {
    //watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some multiplications
    const double yplus=dist*utau/viscosity;
    double psi=0.0;


    if(yplus>11.0)//this is approximately where the intersection of log law and linear region lies
      psi=log(yplus)+5.17*0.41;
    else
      psi=yplus*0.41;

    double inc=10.0;
    double fn=10.0;
    int count=0;
    bool converged = false;
    while(not converged)
    {
      const double psiquad=psi*psi;
      const double exppsi=std::exp(psi);
      const double expmkmb=std::exp(-0.41*5.17);
             fn=-yplus + psi*(1./0.41)+(expmkmb)*(exppsi-(1.0)-psi-psiquad*(0.5) - psiquad*psi/(6.0) - psiquad*psiquad/(24.0));
             double dfn= 1/0.41+expmkmb*(exppsi-(1.0)-psi-psiquad*(0.5) - psiquad*psi/(6.0));

      inc=fn/dfn;

      psi-=inc;

      bool test=false;
      //do loop for all if one of the values is not converged
        if((std::abs(inc)>1.0E-14 && abs(fn)>1.0E-14&&1000>count++))
            test=true;

      converged = not test;
    }

    return psi;

    //Reichardt's law 1951
    // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
  }
};

template <int dim, int n_q_points_1d, typename Number>
class EvaluationXWall
{
public:
  EvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                      const parallel::distributed::Vector<double>& wdist,
                      const parallel::distributed::Vector<double>& tauw,
                      double viscosity):
                        mydata(matrix_free),
                        wdist(wdist),
                        tauw(tauw),
                        viscosity(viscosity),
                        evaluate_value(true),
                        evaluate_gradient(true),
                        evaluate_hessian(false),
                        k(0.41),
                        km1(1.0/k),
                        B(5.17),
                        expmkmb(exp(-k*B))
    {};

  void reinit(AlignedVector<VectorizedArray<Number> > qp_wdist,
      AlignedVector<VectorizedArray<Number> > qp_tauw,
      AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_gradwdist,
      AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_gradtauw,
      unsigned int n_q_points,
      std::vector<bool> enriched_components)
  {

    qp_enrichment.resize(n_q_points);
    qp_grad_enrichment.resize(n_q_points);
    for(unsigned int q=0;q<n_q_points;++q)
    {
      qp_enrichment[q] =  EnrichmentShapeDer(qp_wdist[q], qp_tauw[q],
          qp_gradwdist[q], qp_gradtauw[q],&(qp_grad_enrichment[q]), enriched_components);

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        if(not enriched_components.at(v))
        {
          qp_enrichment[q][v] = 0.0;
          for (unsigned int d = 0; d<dim; d++)
            qp_grad_enrichment[q][d][v] = 0.0;
        }

      }
    }

  };

  void evaluate(const bool evaluate_val,
             const bool evaluate_grad,
             const bool evaluate_hess = false)
  {
    evaluate_value = evaluate_val;
    evaluate_gradient = evaluate_grad;
    //second derivative not implemented yet
    evaluate_hessian = evaluate_hess;
    Assert(not evaluate_hessian,ExcInternalError());
  }
  VectorizedArray<Number> enrichment(unsigned int q){return qp_enrichment[q];}
  Tensor<1,dim,VectorizedArray<Number> > enrichment_gradient(unsigned int q){return qp_grad_enrichment[q];}
  protected:
  VectorizedArray<Number> EnrichmentShapeDer(VectorizedArray<Number> wdist, VectorizedArray<Number> tauw,
      Tensor<1,dim,VectorizedArray<Number> > gradwdist, Tensor<1,dim,VectorizedArray<Number> > gradtauw,
      Tensor<1,dim,VectorizedArray<Number> >* gradpsi, std::vector<bool> enriched_components)
    {
         VectorizedArray<Number> density = make_vectorized_array(1.0);
//        //calculate transformation ---------------------------------------

       Tensor<1,dim,VectorizedArray<Number> > gradtrans;

       const VectorizedArray<Number> utau=std::sqrt(tauw*make_vectorized_array(1.0)/density);
       const VectorizedArray<Number> fac=make_vectorized_array(0.5)/std::sqrt(density*tauw);
       const VectorizedArray<Number> wdistfac=wdist*fac;
//
       for(unsigned int sdm=0;sdm < dim;++sdm)
         gradtrans[sdm]=(utau*gradwdist[sdm]+wdistfac*gradtauw[sdm])*make_vectorized_array(1.0/viscosity);

       //get enrichment function and scalar derivatives
         VectorizedArray<Number> psigp = SpaldingsLaw(wdist, utau, enriched_components)*make_vectorized_array(1.0);
         VectorizedArray<Number> derpsigpsc=DerSpaldingsLaw(psigp)*make_vectorized_array(1.0);
//         //calculate final derivatives
       Tensor<1,dim,VectorizedArray<Number> > gradpsiq;
       for(int sdm=0;sdm < dim;++sdm)
       {
         gradpsiq[sdm]=derpsigpsc*gradtrans[sdm];
       }

       (*gradpsi)=gradpsiq;

      return psigp;
    }

    const MatrixFree<dim,Number> &mydata;

  const parallel::distributed::Vector<double>& wdist;
  const parallel::distributed::Vector<double>& tauw;
  const double viscosity;

  private:

  bool evaluate_value;
  bool evaluate_gradient;
  bool evaluate_hessian;

  const Number k;
  const Number km1;
  const Number B;
  const Number expmkmb;

  AlignedVector<VectorizedArray<Number> > qp_enrichment;
  AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > qp_grad_enrichment;


    VectorizedArray<Number> SpaldingsLaw(VectorizedArray<Number> dist, VectorizedArray<Number> utau, std::vector<bool> enriched_components)
    {
      //watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some multiplications
      const VectorizedArray<Number> yplus=dist*utau*make_vectorized_array(1.0/viscosity);
      VectorizedArray<Number> psi=make_vectorized_array(0.0);

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        if(enriched_components.at(v))
        {
          if(yplus[v]>11.0)//this is approximately where the intersection of log law and linear region lies
            psi[v]=log(yplus[v])+B*k;
          else
            psi[v]=yplus[v]*k;
        }
        else
          psi[v] = 0.0;
      }

      VectorizedArray<Number> inc=make_vectorized_array(10.0);
      VectorizedArray<Number> fn=make_vectorized_array(10.0);
      int count=0;
      bool converged = false;
      while(not converged)
      {
        VectorizedArray<Number> psiquad=psi*psi;
        VectorizedArray<Number> exppsi=std::exp(psi);
               fn=-yplus + psi*make_vectorized_array(km1)+make_vectorized_array(expmkmb)*(exppsi-make_vectorized_array(1.0)-psi-psiquad*make_vectorized_array(0.5) - psiquad*psi/make_vectorized_array(6.0) - psiquad*psiquad/make_vectorized_array(24.0));
               VectorizedArray<Number> dfn= km1+expmkmb*(exppsi-make_vectorized_array(1.0)-psi-psiquad*make_vectorized_array(0.5) - psiquad*psi/make_vectorized_array(6.0));

        inc=fn/dfn;

        psi-=inc;

        bool test=false;
        //do loop for all if one of the values is not converged
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        {
          if(enriched_components.at(v))
            if((std::abs(inc[v])>1.0E-14 && abs(fn[v])>1.0E-14&&1000>count++))
              test=true;
        }
        converged = not test;
      }

      return psi;

      //Reichardt's law 1951
      // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
    }

    VectorizedArray<Number> DerSpaldingsLaw(VectorizedArray<Number> psi)
    {
      //derivative with respect to y+!
      //spaldings law according to paper (derivative)
      return make_vectorized_array(1.0)/(make_vectorized_array(1.0/k)+make_vectorized_array(expmkmb)*(std::exp(psi)-make_vectorized_array(1.0)-psi-psi*psi*make_vectorized_array(0.5)-psi*psi*psi/make_vectorized_array(6.0)));

    // Reichardt's law
    //  double yplus=dist*utau*viscinv_;
    //  return (0.4/(k_*(1.0+0.4*yplus))+7.8*(1.0/11.0*exp(-yplus/11.0)-1.0/11.0*exp(-yplus/3.0)+yplus/33.0*exp(-yplus/3.0)))*k_;
    }

    Number Der2SpaldingsLaw(Number psi,Number derpsi)
    {
      //derivative with respect to y+!
      //spaldings law according to paper (2nd derivative)
      return -make_vectorized_array(expmkmb)*(exp(psi)-make_vectorized_array(1.)-psi-psi*psi*make_vectorized_array(0.5))*derpsi*derpsi*derpsi;

      // Reichardt's law
    //  double yplus=dist*utau*viscinv_;
    //  return (-0.4*0.4/(k_*(1.0+0.4*yplus)*(1.0+0.4*yplus))+7.8*(-1.0/121.0*exp(-yplus/11.0)+(2.0/33.0-yplus/99.0)*exp(-yplus/3.0)))*k_;
    }
  };

template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
            int n_components_ = 1, typename Number = double >
  class FEEvaluationXWall : public EvaluationXWall<dim,n_q_points_1d, Number>
  {
    typedef FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
    typedef Number                            number_type;
    typedef typename BaseClass::value_type    value_type;
    typedef typename BaseClass::gradient_type gradient_type;

public:
  FEEvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                      FEParameters<Number> const & in_fe_params,
                      const unsigned int            fe_no = 0,
                      const unsigned int            quad_no = 0):
                        EvaluationXWall<dim,n_q_points_1d, Number>::EvaluationXWall(matrix_free, in_fe_params.xwallstatevec[0], in_fe_params.xwallstatevec[1],in_fe_params.viscosity),
                        fe_params(in_fe_params),
                        fe_eval(),
                        fe_eval_xwall(),
                        fe_eval_tauw(),
                        values(),
                        gradients(),
                        std_dofs_per_cell(0),
                        dofs_per_cell(0),
                        tensor_dofs_per_cell(0),
                        n_q_points(0),
                        enriched(false)
    {
      {
        FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval_tmp(matrix_free,fe_no,quad_no);
        fe_eval.resize(1,fe_eval_tmp);
      }
#ifdef XWALL
      {
        FEEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall_tmp(matrix_free,3,quad_no);
        fe_eval_xwall.resize(1,fe_eval_xwall_tmp);
      }
#endif
      {
        FEEvaluation<dim,1,n_q_points_1d,1,double> fe_eval_tauw_tmp(matrix_free,2,quad_no);
        fe_eval_tauw.resize(1,fe_eval_tauw_tmp);
      }
      values.resize(fe_eval[0].n_q_points,value_type());
      gradients.resize(fe_eval[0].n_q_points,gradient_type());
      n_q_points = fe_eval[0].n_q_points;
    };

    void reinit(const unsigned int cell)
    {
#ifdef XWALL
      {
        enriched = false;
        values.resize(fe_eval[0].n_q_points,value_type());
        gradients.resize(fe_eval[0].n_q_points,gradient_type());
//        decide if we have an enriched element via the y component of the cell center
        for (unsigned int v=0; v<EvaluationXWall<dim,n_q_points_1d, Number>::mydata.n_components_filled(cell); ++v)
        {
          typename DoFHandler<dim>::cell_iterator dcell = EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(cell, v);
//            std::cout << ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL))) << std::endl;
          if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
            enriched = true;
        }
        enriched_components.resize(VectorizedArray<Number>::n_array_elements);
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          enriched_components.at(v) = false;
        if(enriched)
        {
          //store, exactly which component of the vectorized array is enriched
          for (unsigned int v=0; v<EvaluationXWall<dim,n_q_points_1d, Number>::mydata.n_components_filled(cell); ++v)
          {
            typename DoFHandler<dim>::cell_iterator dcell = EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(cell, v);
            if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                enriched_components.at(v) = true;
          }

          //initialize the enrichment function
          {
            fe_eval_tauw[0].reinit(cell);
            //get wall distance and wss at quadrature points
            fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
            fe_eval_tauw[0].evaluate(true, true);

            AlignedVector<VectorizedArray<Number> > cell_wdist;
            AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradwdist;
            cell_wdist.resize(fe_eval_tauw[0].n_q_points);
            cell_gradwdist.resize(fe_eval_tauw[0].n_q_points);
            for(unsigned int q=0;q<fe_eval_tauw[0].n_q_points;++q)
            {
              cell_wdist[q] = fe_eval_tauw[0].get_value(q);
              cell_gradwdist[q] = fe_eval_tauw[0].get_gradient(q);
            }

            fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);

            fe_eval_tauw[0].evaluate(true, true);

            AlignedVector<VectorizedArray<Number> > cell_tauw;
            AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > cell_gradtauw;

            cell_tauw.resize(fe_eval_tauw[0].n_q_points);
            cell_gradtauw.resize(fe_eval_tauw[0].n_q_points);

            for(unsigned int q=0;q<fe_eval_tauw[0].n_q_points;++q)
            {
              cell_tauw[q] = fe_eval_tauw[0].get_value(q);
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
              {
                if(enriched_components.at(v))
                  Assert( fe_eval_tauw[0].get_value(q)[v] > 1.0e-9 ,ExcInternalError());
              }

              cell_gradtauw[q] = fe_eval_tauw[0].get_gradient(q);
            }
            EvaluationXWall<dim,n_q_points_1d, Number>::reinit(cell_wdist, cell_tauw, cell_gradwdist, cell_gradtauw, fe_eval_tauw[0].n_q_points,enriched_components);
          }
        }
        fe_eval_xwall[0].reinit(cell);
      }
#endif
      fe_eval[0].reinit(cell);
      std_dofs_per_cell = fe_eval[0].dofs_per_cell;
#ifdef XWALL
      if(enriched)
      {
        dofs_per_cell = fe_eval[0].dofs_per_cell + fe_eval_xwall[0].dofs_per_cell;
        tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell + fe_eval_xwall[0].tensor_dofs_per_cell;
      }
      else
      {
        dofs_per_cell = fe_eval[0].dofs_per_cell;
        tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell;
      }
#else
      dofs_per_cell = fe_eval[0].dofs_per_cell;
      tensor_dofs_per_cell = fe_eval[0].tensor_dofs_per_cell;
#endif
    }

    VectorizedArray<double> * begin_dof_values()
    {
      return fe_eval[0].begin_dof_values();
    }

    void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &src_xwall)
    {
      fe_eval[0].read_dof_values(src);
#ifdef XWALL
      fe_eval_xwall[0].read_dof_values(src_xwall);
#endif
    }

    void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i,const std::vector<parallel::distributed::Vector<double> > &src_xwall, unsigned int j)
    {
      fe_eval[0].read_dof_values(src,i);
#ifdef XWALL
      fe_eval_xwall[0].read_dof_values(src_xwall,j);
#endif
    }
    void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i,const parallel::distributed::BlockVector<double> &src_xwall, unsigned int j)
    {
      fe_eval[0].read_dof_values(src,i);
#ifdef XWALL
      fe_eval_xwall[0].read_dof_values(src_xwall,j);
#endif
    }

    void evaluate(const bool evaluate_val,
               const bool evaluate_grad,
               const bool evaluate_hess = false)
    {
      fe_eval[0].evaluate(evaluate_val,evaluate_grad,evaluate_hess);
#ifdef XWALL
        if(enriched)
        {
          gradients.resize(fe_eval[0].n_q_points,gradient_type());
          values.resize(fe_eval[0].n_q_points,value_type());
          fe_eval_xwall[0].evaluate(true,evaluate_grad);
          //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
          EvaluationXWall<dim,n_q_points_1d,Number>::evaluate(evaluate_val,evaluate_grad,evaluate_hess);
          //evaluate gradient
          if(evaluate_grad)
          {
            gradient_type submitgradient = gradient_type();
            gradient_type gradient = gradient_type();
            //there are 2 parts due to chain rule
            for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
            {
              submitgradient = gradient_type();
              gradient = fe_eval_xwall[0].get_gradient(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
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
            for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
            {
              value_type finalvalue = fe_eval_xwall[0].get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
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
#endif
    }

    void val_enrgrad_to_grad(Tensor<2,dim,VectorizedArray<Number> >& grad, unsigned int q)
    {
      for(unsigned int j=0;j<dim;++j)
      {
        for(unsigned int i=0;i<dim;++i)
        {
          grad[j][i] += fe_eval_xwall[0].get_value(q)[j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
        }
      }
    }
    void val_enrgrad_to_grad(Tensor<1,dim,VectorizedArray<Number> >& grad, unsigned int q)
    {
      for(unsigned int i=0;i<dim;++i)
      {
        grad[i] += fe_eval_xwall[0].get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
      }
    }


    void submit_value(const value_type val_in,
        const unsigned int q_point)
    {
      fe_eval[0].submit_value(val_in,q_point);
#ifdef XWALL
      values[q_point] = value_type();
        if(enriched)
          values[q_point] = val_in;
#endif
    }
    void submit_value(const Tensor<1,1,VectorizedArray<Number> > val_in,
        const unsigned int q_point)
    {
      fe_eval[0].submit_value(val_in[0],q_point);
#ifdef XWALL
      values[q_point] = value_type();
        if(enriched)
          values[q_point] = val_in[0];
#endif
    }

    void submit_gradient(const gradient_type grad_in,
        const unsigned int q_point)
    {
      fe_eval[0].submit_gradient(grad_in,q_point);
#ifdef XWALL
      gradients[q_point] = gradient_type();
      if(enriched)
        gradients[q_point] = grad_in;
#endif
    }

    void value_type_unit(VectorizedArray<Number>* test)
      {
        *test = make_vectorized_array(1.);
      }

    void value_type_unit(Tensor<1,n_components_,VectorizedArray<Number> >* test)
      {
        for(unsigned int i = 0; i< n_components_; i++)
          (*test)[i] = make_vectorized_array(1.);
      }

    void print_value_type_unit(VectorizedArray<Number> test)
      {
        std::cout << test[0] << std::endl;
      }

    void print_value_type_unit(Tensor<1,n_components_,VectorizedArray<Number> > test)
      {
        for(unsigned int i = 0; i< n_components_; i++)
          std::cout << test[i][0] << "  ";
        std::cout << std::endl;
      }

    value_type get_value(const unsigned int q_point)
    {
#ifdef XWALL
      if(enriched)
        return values[q_point] + fe_eval[0].get_value(q_point);
#endif
        return fe_eval[0].get_value(q_point);
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


    gradient_type get_gradient (const unsigned int q_point)
    {
#ifdef XWALL
        if(enriched)
          return fe_eval[0].get_gradient(q_point) + gradients[q_point];
#endif
      return fe_eval[0].get_gradient(q_point);
    }

    gradient_type get_symmetric_gradient (const unsigned int q_point)
    {
      return make_symmetric(get_gradient(q_point)); //TODO Benjamin, now make_symmetric defined with 1/2(grad + gradT)
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
        Assert(false, ExcInternalError());
        return symgrad;
    }

    void integrate (const bool integrate_val,
                    const bool integrate_grad)
    {
#ifdef XWALL
      {
        if(enriched)
        {
          AlignedVector<value_type> tmp_values(fe_eval[0].n_q_points,value_type());
          if(integrate_val)
            for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              tmp_values[q]=values[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
          //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
          //the scalar product of the second part of the gradient is computed directly and added to the value
          if(integrate_grad)
          {
            //first, zero out all non-enriched vectorized array components
            grad_enr_to_val(tmp_values, gradients);

            for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
              fe_eval_xwall[0].submit_gradient(gradients[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q),q);
          }

          for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
            fe_eval_xwall[0].submit_value(tmp_values[q],q);
          //integrate
          fe_eval_xwall[0].integrate(true,integrate_grad);
        }
      }
#endif
      fe_eval[0].integrate(integrate_val, integrate_grad);
    }

    void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
    {
      for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
      {
        for(int j=0; j<dim;++j)//comp
        {
          for(int i=0; i<dim;++i)//dim
          {
            tmp_values[q][j] += gradient[q][j][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }
    }
    void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
    {
      for(unsigned int q=0;q<fe_eval[0].n_q_points;++q)
      {
        for(int i=0; i<dim;++i)//dim
        {
          tmp_values[q] += gradient[q][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
        }
      }
    }

    void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
    {
      fe_eval[0].distribute_local_to_global(dst);
#ifdef XWALL
      if(enriched)
        fe_eval_xwall[0].distribute_local_to_global(dst_xwall);
#endif
    }

    void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
    {
      fe_eval[0].distribute_local_to_global(dst,i);
#ifdef XWALL
      if(enriched)
        fe_eval_xwall[0].distribute_local_to_global(dst_xwall,j);
#endif
    }

    void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i,parallel::distributed::BlockVector<double> &dst_xwall, unsigned int j)
    {
      fe_eval[0].distribute_local_to_global(dst,i);
#ifdef XWALL
      if(enriched)
        fe_eval_xwall[0].distribute_local_to_global(dst_xwall,j);
#endif
    }

    void set_dof_values (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
    {
      fe_eval[0].set_dof_values(dst);
#ifdef XWALL
      if(enriched)
        fe_eval_xwall[0].set_dof_values(dst_xwall);
#endif
    }

    void set_dof_values (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
    {
      fe_eval[0].set_dof_values(dst,i);
#ifdef XWALL
      if(enriched)
        fe_eval_xwall[0].set_dof_values(dst_xwall,j);
#endif
    }

    void fill_JxW_values(AlignedVector<VectorizedArray<Number> > &JxW_values) const
    {
      fe_eval[0].fill_JxW_values(JxW_values);
    }

    Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
    {
      return fe_eval[0].quadrature_point(q);
    }

    VectorizedArray<Number> get_divergence(unsigned int q)
  {
#ifdef XWALL
      if(enriched)
      {
        VectorizedArray<Number> div_enr= make_vectorized_array(0.0);
        for (unsigned int i=0;i<dim;i++)
          div_enr += gradients[q][i][i];
        return fe_eval[0].get_divergence(q) + div_enr;
      }
#endif
      return fe_eval[0].get_divergence(q);
  }

  Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
  get_curl (const unsigned int q_point) const
   {
#ifdef XWALL
    if(enriched)
    {
      // copy from generic function into dim-specialization function
      const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
      Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
      switch (dim)
        {
        case 1:
          Assert (false,
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
          Assert (false, ExcNotImplemented());
          break;
        }
      return fe_eval[0].get_curl(q_point) + curl;
    }
#endif
    return fe_eval[0].get_curl(q_point);
   }
  VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
  {
#ifdef XWALL
    if(enriched)
    {
      VectorizedArray<Number> returnvalue = make_vectorized_array(0.0);
      if(j<fe_eval[0].dofs_per_cell*n_components_)
        returnvalue =  fe_eval[0].begin_dof_values()[j];
      else
      {
        returnvalue = fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_];
      }
      return returnvalue;
    }
    else
      return fe_eval[0].begin_dof_values()[j];
#else

    return fe_eval[0].begin_dof_values()[j];
#endif
  }
  void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
  {
#ifdef XWALL
    if(enriched)
    {
      if(j<fe_eval[0].dofs_per_cell*n_components_)
        fe_eval[0].begin_dof_values()[j][v] = value;
      else
        fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_][v] = value;
    }
    else
      fe_eval[0].begin_dof_values()[j][v]=value;
    return;
#else
    fe_eval[0].begin_dof_values()[j][v]=value;
    return;
#endif
  }
  void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
  {
#ifdef XWALL
    if(enriched)
    {
      if(j<fe_eval[0].dofs_per_cell*n_components_)
        fe_eval[0].begin_dof_values()[j] = value;
      else
        fe_eval_xwall[0].begin_dof_values()[j-fe_eval[0].dofs_per_cell*n_components_] = value;
    }
    else
      fe_eval[0].begin_dof_values()[j]=value;
    return;
#else
    fe_eval[0].begin_dof_values()[j]=value;
    return;
#endif
  }
  bool component_enriched(unsigned int v)
  {
    if(not enriched)
      return false;
    else
      return enriched_components.at(v);
  }

  void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int cell)
  {
    eddyvisc.resize(n_q_points);
    if(fe_params.cs > 1e-10)
    {
      const VectorizedArray<Number> Cs = make_vectorized_array(fe_params.cs);
      VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
      fe_eval_tauw[0].reinit(cell);
      {
        VectorizedArray<Number> volume = make_vectorized_array(0.);
        {
          AlignedVector<VectorizedArray<Number> > JxW_values;
          JxW_values.resize(fe_eval_tauw[0].n_q_points);
          fe_eval_tauw[0].fill_JxW_values(JxW_values);
          for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
            volume += JxW_values[q];
        }
        reinit(cell);
        read_dof_values(solution_n,0,solution_n,dim+1);
        evaluate (false,true,false);
        AlignedVector<VectorizedArray<Number> > wdist;
        wdist.resize(fe_eval_tauw[0].n_q_points);
        fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
        fe_eval_tauw[0].evaluate(true,false,false);
        for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
          wdist[q] = fe_eval_tauw[0].get_value(q);
        fe_eval_tauw[0].reinit(cell);
        fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
        fe_eval_tauw[0].evaluate(true,false,false);

        const VectorizedArray<Number> hvol = std::pow(volume, 1./(double)dim) * hfac;

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

          VectorizedArray<Number> snorm = make_vectorized_array(0.);
          for (unsigned int i = 0; i<dim ; i++)
            for (unsigned int j = 0; j<dim ; j++)
              snorm += (s[i][j])*(s[i][j]);
          snorm *= make_vectorized_array<Number>(0.5);
          //simple wall correction
          VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/fe_params.viscosity*std::sqrt(fe_eval_tauw[0].get_value(q))*0.04));
          VectorizedArray<Number> lm = Cs*hvol*fmu;
          eddyvisc[q]= make_vectorized_array(fe_params.viscosity) + lm*lm*std::sqrt(snorm);
        }
      }
      //initialize again to get a clean version
      reinit(cell);
    }
#ifdef XWALL
    else if (fe_params.ml>0.1 && enriched)
    {
      fe_eval_tauw[0].reinit(cell);
      {
        read_dof_values(solution_n,0,solution_n,dim+1);
        evaluate (false,true,false);
        AlignedVector<VectorizedArray<Number> > wdist;
        wdist.resize(fe_eval_tauw[0].n_q_points);
        fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
        fe_eval_tauw[0].evaluate(true,false,false);
        for (unsigned int q=0; q<fe_eval_tauw[0].n_q_points; ++q)
          wdist[q] = fe_eval_tauw[0].get_value(q);
        fe_eval_tauw[0].reinit(cell);
        fe_eval_tauw[0].read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
        fe_eval_tauw[0].evaluate(true,false,false);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<Number> > s = get_gradient(q);
          Tensor<2,dim,VectorizedArray<Number> > om;
          for (unsigned int i=0; i<dim;i++)
            for (unsigned int j=0;j<dim;j++)
              om[i][j]=0.5*(s[i][j]-s[j][i]);

          VectorizedArray<Number> osum = make_vectorized_array(0.);
          for (unsigned int i=0; i<dim;i++)
            for (unsigned int j=0;j<dim;j++)
              osum += om[i][j]*om[i][j];
          VectorizedArray<Number> onorm = std::sqrt(2.*osum);

          //simple wall correction
          VectorizedArray<Number> l = 0.41*wdist[q]*(1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw[0].get_value(q))*0.04));
          VectorizedArray<Number> vt = l*l*onorm;
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(enriched_components.at(v))
            {
              eddyvisc[q][v]= VISCOSITY + vt[v];
            }
            else
              eddyvisc[q][v]= VISCOSITY;
          }
        }
      }
      //initialize again to get a clean version
      reinit(cell);
  }
#endif
    else
      for (unsigned int q=0; q<n_q_points; ++q)
        eddyvisc[q]= make_vectorized_array(fe_params.viscosity);

    return;
  }
  private:
    FEParameters<Number> const & fe_params;
    AlignedVector<FEEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> > fe_eval;
    AlignedVector<FEEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> > fe_eval_xwall;
    AlignedVector<FEEvaluation<dim,1,n_q_points_1d,1,double> > fe_eval_tauw;
    AlignedVector<value_type> values;
    AlignedVector<gradient_type> gradients;

  public:
    unsigned int std_dofs_per_cell;
    unsigned int dofs_per_cell;
    unsigned int tensor_dofs_per_cell;
    unsigned int n_q_points;
    bool enriched;
    std::vector<bool> enriched_components;
    AlignedVector<VectorizedArray<Number> > eddyvisc;

  };


template <int dim, int fe_degree = 1, int fe_degree_xwall = 1, int n_q_points_1d = fe_degree+1,
            int n_components_ = 1, typename Number = double >
  class FEFaceEvaluationXWall : public EvaluationXWall<dim,n_q_points_1d, Number>
  {
  public:
    typedef FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> BaseClass;
    typedef Number                            number_type;
    typedef typename BaseClass::value_type    value_type;
    typedef typename BaseClass::gradient_type gradient_type;

    FEFaceEvaluationXWall (const MatrixFree<dim,Number> &matrix_free,
                      FEParameters<Number> const & in_fe_params,
                      const bool                    is_left_face = true,
                      const unsigned int            fe_no = 0,
                      const unsigned int            quad_no = 0,
                      const bool                    no_gradients_on_faces = false):
                        EvaluationXWall<dim,n_q_points_1d, Number>::EvaluationXWall(matrix_free, in_fe_params.xwallstatevec[0], in_fe_params.xwallstatevec[1],in_fe_params.viscosity),
                        fe_params(in_fe_params),
                        fe_eval(matrix_free,is_left_face,fe_no,quad_no,no_gradients_on_faces),
                        fe_eval_xwall(matrix_free,is_left_face,3,quad_no,no_gradients_on_faces),
                        fe_eval_tauw(matrix_free,is_left_face,2,quad_no,no_gradients_on_faces),
                        is_left_face(is_left_face),
                        values(fe_eval.n_q_points),
                        gradients(fe_eval.n_q_points),
                        dofs_per_cell(0),
                        tensor_dofs_per_cell(0),
                        n_q_points(fe_eval.n_q_points),
                        enriched(false)
    {
    };

    void reinit(const unsigned int f)
    {
#ifdef XWALL
      {
        enriched = false;
        values.resize(fe_eval.n_q_points,value_type());
        gradients.resize(fe_eval.n_q_points,gradient_type());
        if(is_left_face)
        {
//        decide if we have an enriched element via the y component of the cell center
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
            EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
          {
            typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
                EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
                if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                  enriched = true;
          }
        }
        else
        {
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements &&
            EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
          {
            typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
                EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
                if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                  enriched = true;
          }
        }
        enriched_components.resize(VectorizedArray<Number>::n_array_elements);
        for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          enriched_components.at(v) = false;
        if(enriched)
        {
          //store, exactly which component of the vectorized array is enriched
          if(is_left_face)
          {
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
            EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] != numbers::invalid_unsigned_int; ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] / VectorizedArray<Number>::n_array_elements,
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).left_cell[v] % VectorizedArray<Number>::n_array_elements);
                  if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                    enriched_components.at(v)=(true);
            }
          }
          else
          {
            for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements&&
            EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] != numbers::invalid_unsigned_int; ++v)
            {
              typename DoFHandler<dim>::cell_iterator dcell =  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.get_cell_iterator(
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] / VectorizedArray<Number>::n_array_elements,
                  EvaluationXWall<dim,n_q_points_1d, Number>::mydata.faces.at(f).right_cell[v] % VectorizedArray<Number>::n_array_elements);
                  if ((dcell->center()[1] > (1.0-MAX_WDIST_XWALL)) || (dcell->center()[1] <(-1.0 + MAX_WDIST_XWALL)))
                    enriched_components.at(v)=(true);
            }
          }

          Assert(enriched_components.size()==VectorizedArray<Number>::n_array_elements,ExcInternalError());

          //initialize the enrichment function
          {
            fe_eval_tauw.reinit(f);
            //get wall distance and wss at quadrature points
            fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
            fe_eval_tauw.evaluate(true, true);

            AlignedVector<VectorizedArray<Number> > face_wdist;
            AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradwdist;
            face_wdist.resize(fe_eval_tauw.n_q_points);
            face_gradwdist.resize(fe_eval_tauw.n_q_points);
            for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
            {
              face_wdist[q] = fe_eval_tauw.get_value(q);
              face_gradwdist[q] = fe_eval_tauw.get_gradient(q);
            }

            fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
            fe_eval_tauw.evaluate(true, true);
            AlignedVector<VectorizedArray<Number> > face_tauw;
            AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > face_gradtauw;
            face_tauw.resize(fe_eval_tauw.n_q_points);
            face_gradtauw.resize(fe_eval_tauw.n_q_points);
            for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
            {
              face_tauw[q] = fe_eval_tauw.get_value(q);
              for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
              {
                if(enriched_components.at(v))
                  Assert( fe_eval_tauw.get_value(q)[v] > 1.0e-9 ,ExcInternalError());
              }

              face_gradtauw[q] = fe_eval_tauw.get_gradient(q);
            }
            EvaluationXWall<dim,n_q_points_1d, Number>::reinit(face_wdist, face_tauw, face_gradwdist, face_gradtauw, fe_eval_tauw.n_q_points,enriched_components);
          }
        }
        fe_eval_xwall.reinit(f);
      }
#endif
      fe_eval.reinit(f);
#ifdef XWALL
      if(enriched)
      {
        dofs_per_cell = fe_eval.dofs_per_cell + fe_eval_xwall.dofs_per_cell;
        tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell + fe_eval_xwall.tensor_dofs_per_cell;
      }
      else
      {
        dofs_per_cell = fe_eval.dofs_per_cell;
        tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell;
      }
#else
      dofs_per_cell = fe_eval.dofs_per_cell;
      tensor_dofs_per_cell = fe_eval.tensor_dofs_per_cell;
#endif
    }

    void read_dof_values (const parallel::distributed::Vector<double> &src, const parallel::distributed::Vector<double> &src_xwall)
    {
      fe_eval.read_dof_values(src);
#ifdef XWALL
      fe_eval_xwall.read_dof_values(src_xwall);
#endif
    }

    void read_dof_values (const std::vector<parallel::distributed::Vector<double> > &src, unsigned int i,const std::vector<parallel::distributed::Vector<double> > &src_xwall, unsigned int j)
    {
      fe_eval.read_dof_values(src,i);
#ifdef XWALL
      fe_eval_xwall.read_dof_values(src_xwall,j);
#endif
    }

    void read_dof_values (const parallel::distributed::BlockVector<double> &src, unsigned int i,const parallel::distributed::BlockVector<double> &src_xwall, unsigned int j)
    {
      fe_eval.read_dof_values(src,i);
#ifdef XWALL
      fe_eval_xwall.read_dof_values(src_xwall,j);
#endif
    }

    void evaluate(const bool evaluate_val,
               const bool evaluate_grad,
               const bool evaluate_hess = false)
    {
      AssertThrow(evaluate_hess == false, ExcNotImplemented());
      fe_eval.evaluate(evaluate_val,evaluate_grad);
#ifdef XWALL
        if(enriched)
        {
          gradients.resize(fe_eval.n_q_points,gradient_type());
          values.resize(fe_eval.n_q_points,value_type());
          fe_eval_xwall.evaluate(true,evaluate_grad);
          //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
          EvaluationXWall<dim,n_q_points_1d,Number>::evaluate(evaluate_val,evaluate_grad,evaluate_hess);
          //evaluate gradient
          if(evaluate_grad)
          {
            //there are 2 parts due to chain rule
            gradient_type gradient = gradient_type();
            gradient_type submitgradient = gradient_type();
            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
            {
              submitgradient = gradient_type();
              gradient = fe_eval_xwall.get_gradient(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
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
            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
            {
              value_type finalvalue = fe_eval_xwall.get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
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
#endif
    }
    void val_enrgrad_to_grad(Tensor<2,dim,VectorizedArray<Number> >& grad, unsigned int q)
    {
      for(unsigned int j=0;j<dim;++j)
      {
        for(unsigned int i=0;i<dim;++i)
        {
          grad[j][i] += fe_eval_xwall.get_value(q)[j]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
        }
      }
    }
    void val_enrgrad_to_grad(Tensor<1,dim,VectorizedArray<Number> >& grad, unsigned int q)
    {
      for(unsigned int i=0;i<dim;++i)
      {
        grad[i] += fe_eval_xwall.get_value(q)*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
      }
    }

    void submit_value(const value_type val_in,
        const unsigned int q_point)
    {
      fe_eval.submit_value(val_in,q_point);
#ifdef XWALL
      values[q_point] = value_type();
      if(enriched)
        values[q_point] = val_in;
#endif
    }

    void submit_gradient(const gradient_type grad_in,
        const unsigned int q_point)
    {
      fe_eval.submit_gradient(grad_in,q_point);
#ifdef XWALL
      gradients[q_point] = gradient_type();
      if(enriched)
        gradients[q_point] = grad_in;
#endif
    }

    value_type get_value(const unsigned int q_point)
    {
#ifdef XWALL
      {
        if(enriched)
          return fe_eval.get_value(q_point) + values[q_point];//fe_eval.get_value(q_point) + values[q_point];
      }
#endif
        return fe_eval.get_value(q_point);
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

    gradient_type get_gradient (const unsigned int q_point)
    {
#ifdef XWALL
      if(enriched)
        return fe_eval.get_gradient(q_point) + gradients[q_point];
#endif
      return fe_eval.get_gradient(q_point);
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
      Assert(false, ExcInternalError());
      return symgrad;
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

    VectorizedArray<Number> get_divergence(unsigned int q)
  {
#ifdef XWALL
      if(enriched)
      {
        VectorizedArray<Number> div_enr= make_vectorized_array(0.0);
        for (unsigned int i=0;i<dim;i++)
          div_enr += gradients[q][i][i];
        return fe_eval.get_divergence(q) + div_enr;
      }
#endif
      return fe_eval.get_divergence(q);
  }

    Tensor<1,dim,VectorizedArray<Number> > get_normal_vector(const unsigned int q_point) const
    {
      return fe_eval.get_normal_vector(q_point);
    }

    void integrate (const bool integrate_val,
                    const bool integrate_grad)
    {
#ifdef XWALL
      {
        if(enriched)
        {
          AlignedVector<value_type> tmp_values(fe_eval.n_q_points,value_type());
          if(integrate_val)
            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              tmp_values[q]=values[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q);
          //this function is quite nasty because deal.ii doesn't seem to be made for enrichments
          //the scalar product of the second part of the gradient is computed directly and added to the value
          if(integrate_grad)
          {
            grad_enr_to_val(tmp_values,gradients);
            for(unsigned int q=0;q<fe_eval.n_q_points;++q)
              fe_eval_xwall.submit_gradient(gradients[q]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment(q),q);
          }

          for(unsigned int q=0;q<fe_eval.n_q_points;++q)
            fe_eval_xwall.submit_value(tmp_values[q],q);
          //integrate
          fe_eval_xwall.integrate(true,integrate_grad);
        }
      }
#endif
      fe_eval.integrate(integrate_val, integrate_grad);
    }

    void grad_enr_to_val(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& tmp_values, AlignedVector<Tensor<2,dim,VectorizedArray<Number> > >& gradient)
    {
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {

        for(int j=0; j<dim;++j)//comp
        {
          for(int i=0; i<dim;++i)//dim
          {
            tmp_values[q][j] += gradient[q][j][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
          }
        }
      }
    }
    void grad_enr_to_val(AlignedVector<VectorizedArray<Number> >& tmp_values, AlignedVector<Tensor<1,dim,VectorizedArray<Number> > >& gradient)
    {
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        for(int i=0; i<dim;++i)//dim
        {
          tmp_values[q] += gradient[q][i]*EvaluationXWall<dim,n_q_points_1d,Number>::enrichment_gradient(q)[i];
        }
      }
    }

    void distribute_local_to_global (parallel::distributed::Vector<double> &dst, parallel::distributed::Vector<double> &dst_xwall)
    {
      fe_eval.distribute_local_to_global(dst);
#ifdef XWALL
        if(enriched)
          fe_eval_xwall.distribute_local_to_global(dst_xwall);
#endif
    }

    void distribute_local_to_global (std::vector<parallel::distributed::Vector<double> > &dst, unsigned int i,std::vector<parallel::distributed::Vector<double> > &dst_xwall, unsigned int j)
    {
      fe_eval.distribute_local_to_global(dst,i);
#ifdef XWALL
      if(enriched)
        fe_eval_xwall.distribute_local_to_global(dst_xwall,j);
#endif
    }


    void distribute_local_to_global (parallel::distributed::BlockVector<double> &dst, unsigned int i,parallel::distributed::BlockVector<double> &dst_xwall, unsigned int j)
    {
      fe_eval.distribute_local_to_global(dst,i);
#ifdef XWALL
      if(enriched)
        fe_eval_xwall.distribute_local_to_global(dst_xwall,j);
#endif
    }

    Point<dim,VectorizedArray<Number> > quadrature_point(unsigned int q)
    {
      return fe_eval.quadrature_point(q);
    }

    VectorizedArray<Number> get_normal_volume_fraction()
    {
      return fe_eval.get_normal_volume_fraction();
    }

    VectorizedArray<Number> read_cell_data(const AlignedVector<VectorizedArray<Number> > &cell_data)
    {
      return fe_eval.read_cell_data(cell_data);
    }

    Tensor<1,n_components_,VectorizedArray<Number> > get_normal_gradient(const unsigned int q_point) const
    {
#ifdef XWALL
    {
      if(enriched)
      {
        Tensor<1,n_components_,VectorizedArray<Number> > grad_out;
        for (unsigned int comp=0; comp<n_components_; comp++)
        {
          grad_out[comp] = gradients[q_point][comp][0] *
                           fe_eval.get_normal_vector(q_point)[0];
          for (unsigned int d=1; d<dim; ++d)
            grad_out[comp] += gradients[q_point][comp][d] *
                             fe_eval.get_normal_vector(q_point)[d];
        }
        return fe_eval.get_normal_gradient(q_point) + grad_out;
      }
    }
#endif
      return fe_eval.get_normal_gradient(q_point);
    }
    VectorizedArray<Number> get_normal_gradient(const unsigned int q_point,bool test) const
    {
#ifdef XWALL
    if(enriched)
    {
      VectorizedArray<Number> grad_out;
        grad_out = gradients[q_point][0] *
                         fe_eval.get_normal_vector(q_point)[0];
        for (unsigned int d=1; d<dim; ++d)
          grad_out += gradients[q_point][d] *
                           fe_eval.get_normal_vector(q_point)[d];

        grad_out +=  fe_eval.get_normal_gradient(q_point);
      return grad_out;
    }
#endif
      return fe_eval.get_normal_gradient(q_point);
    }

    void submit_normal_gradient (const Tensor<1,n_components_,VectorizedArray<Number> > grad_in,
                              const unsigned int q)
    {
      fe_eval.submit_normal_gradient(grad_in,q);
#ifdef XWALL
      gradients[q]=gradient_type();
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
                fe_eval.get_normal_vector(q)[d][v];
              }
              else
                gradients[q][comp][d][v] = 0.0;
            }
        }
    }
#endif
    }
    void submit_normal_gradient (const VectorizedArray<Number> grad_in,
                              const unsigned int q)
    {
      fe_eval.submit_normal_gradient(grad_in,q);
#ifdef XWALL
      gradients[q]=gradient_type();
      if(enriched)
      {
        for (unsigned int d=0; d<dim; ++d)
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(enriched_components.at(v))
            {
              gradients[q][d][v] = grad_in[v] *
              fe_eval.get_normal_vector(q)[d][v];
            }
            else
              gradients[q][d][v] = 0.0;
          }
      }
#endif
    }
    Tensor<1,dim==2?1:dim,VectorizedArray<Number> >
    get_curl (const unsigned int q_point) const
     {
#ifdef XWALL
      if(enriched)
      {
        // copy from generic function into dim-specialization function
        const Tensor<2,dim,VectorizedArray<Number> > grad = gradients[q_point];
        Tensor<1,dim==2?1:dim,VectorizedArray<Number> > curl;
        switch (dim)
          {
          case 1:
            Assert (false,
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
            Assert (false, ExcNotImplemented());
            break;
          }
        return fe_eval.get_curl(q_point) + curl;
      }
#endif
      return fe_eval.get_curl(q_point);
     }

    VectorizedArray<Number> read_cellwise_dof_value (unsigned int j)
    {
#ifdef XWALL
      if(enriched)
      {
        VectorizedArray<Number> returnvalue = make_vectorized_array(0.0);
        if(j<fe_eval.dofs_per_cell*n_components_)
          returnvalue = fe_eval.begin_dof_values()[j];
        else
          returnvalue = fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_];
        return returnvalue;
      }
      else
        return fe_eval.begin_dof_values()[j];
#else

      return fe_eval.begin_dof_values()[j];
#endif
    }
    void write_cellwise_dof_value (unsigned int j, Number value, unsigned int v)
    {
#ifdef XWALL
      if(enriched)
      {
        if(j<fe_eval.dofs_per_cell*n_components_)
          fe_eval.begin_dof_values()[j][v] = value;
        else
          fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_][v] = value;
      }
      else
        fe_eval.begin_dof_values()[j][v]=value;
      return;
#else
      fe_eval.begin_dof_values()[j][v]=value;
      return;
#endif
    }
    void write_cellwise_dof_value (unsigned int j, VectorizedArray<Number> value)
    {
#ifdef XWALL
      if(enriched)
      {
        if(j<fe_eval.dofs_per_cell*n_components_)
          fe_eval.begin_dof_values()[j] = value;
        else
          fe_eval_xwall.begin_dof_values()[j-fe_eval.dofs_per_cell*n_components_] = value;
      }
      else
        fe_eval.begin_dof_values()[j]=value;
      return;
#else
      fe_eval.begin_dof_values()[j]=value;
      return;
#endif
    }
    void evaluate_eddy_viscosity(const std::vector<parallel::distributed::Vector<double> > &solution_n, unsigned int face, const VectorizedArray<Number> volume)
    {
      eddyvisc.resize(n_q_points);
      if(fe_params.cs > 1e-10)
      {
        const VectorizedArray<Number> Cs = make_vectorized_array(fe_params.cs);
        VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
        fe_eval_tauw.reinit(face);
        {
          reinit(face);
          read_dof_values(solution_n,0,solution_n,dim+1);
          evaluate (false,true,false);
          AlignedVector<VectorizedArray<Number> > wdist;
          wdist.resize(fe_eval_tauw.n_q_points);
          fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
          fe_eval_tauw.evaluate(true,false);
          for (unsigned int q=0; q<fe_eval_tauw.n_q_points; ++q)
            wdist[q] = fe_eval_tauw.get_value(q);
          fe_eval_tauw.reinit(face);
          fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
          fe_eval_tauw.evaluate(true,false);

          const VectorizedArray<Number> hvol = hfac * std::pow(volume, 1./(double)dim);

          for (unsigned int q=0; q<n_q_points; ++q)
          {
            Tensor<2,dim,VectorizedArray<Number> > s = get_symmetric_gradient(q);

            VectorizedArray<Number> snorm = make_vectorized_array(0.);
            for (unsigned int i = 0; i<dim ; i++)
              for (unsigned int j = 0; j<dim ; j++)
                snorm += (s[i][j])*(s[i][j]);
            snorm *= make_vectorized_array<Number>(0.5);
            //simple wall correction
            VectorizedArray<Number> fmu = (1.-std::exp(-wdist[q]/fe_params.viscosity*std::sqrt(fe_eval_tauw.get_value(q))*0.04));
            VectorizedArray<Number> lm = Cs*hvol*fmu;
            eddyvisc[q]= make_vectorized_array(fe_params.viscosity) + lm*lm*std::sqrt(snorm);
          }
        }
        //initialize again to get a clean version
        reinit(face);
      }
#ifdef XWALL
    else if (fe_params.ml>0.1 && enriched)
    {
      VectorizedArray<Number> hfac = make_vectorized_array(1.0/(double)fe_degree);
      fe_eval_tauw.reinit(face);
      {
        read_dof_values(solution_n,0,solution_n,dim+1);
        evaluate (false,true,false);
        AlignedVector<VectorizedArray<Number> > wdist;
        wdist.resize(fe_eval_tauw.n_q_points);
        fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::wdist);
        fe_eval_tauw.evaluate(true,false);
        for (unsigned int q=0; q<fe_eval_tauw.n_q_points; ++q)
          wdist[q] = fe_eval_tauw.get_value(q);
        fe_eval_tauw.reinit(face);
        fe_eval_tauw.read_dof_values(EvaluationXWall<dim,n_q_points_1d, Number>::tauw);
        fe_eval_tauw.evaluate(true,false);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          Tensor<2,dim,VectorizedArray<Number> > s = get_gradient(q);
          Tensor<2,dim,VectorizedArray<Number> > om;
          for (unsigned int i=0; i<dim;i++)
            for (unsigned int j=0;j<dim;j++)
              om[i][j]=0.5*(s[i][j]-s[j][i]);

          VectorizedArray<Number> osum = make_vectorized_array(0.);
          for (unsigned int i=0; i<dim;i++)
            for (unsigned int j=0;j<dim;j++)
              osum += om[i][j]*om[i][j];
          VectorizedArray<Number> onorm = std::sqrt(2.*osum);

          //simple wall correction
          VectorizedArray<Number> l = 0.41*wdist[q]*(1.-std::exp(-wdist[q]/VISCOSITY*std::sqrt(fe_eval_tauw.get_value(q))*0.04));
          VectorizedArray<Number> vt = l*l*onorm;
          for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
          {
            if(enriched_components.at(v))
            {
              eddyvisc[q][v]= VISCOSITY + vt[v];
            }
            else
              eddyvisc[q][v]= VISCOSITY;
          }
        }
      }
      //initialize again to get a clean version
      reinit(face);
  }
#endif
      else
        for (unsigned int q=0; q<n_q_points; ++q)
          eddyvisc[q]= make_vectorized_array(fe_params.viscosity);

      return;
    }

    void fill_JxW_values(AlignedVector<VectorizedArray<Number> > &JxW_values) const
    {
      fe_eval.fill_JxW_values(JxW_values);
    }

  private:
    FEParameters<Number> const & fe_params;
    FEFaceEvaluation<dim,fe_degree,n_q_points_1d,n_components_,Number> fe_eval;
    FEFaceEvaluation<dim,fe_degree_xwall,n_q_points_1d,n_components_,Number> fe_eval_xwall;
    FEFaceEvaluation<dim,1,n_q_points_1d,1,Number> fe_eval_tauw;
    bool is_left_face;
    AlignedVector<value_type> values;
    AlignedVector<gradient_type> gradients;


  public:
    unsigned int dofs_per_cell;
    unsigned int tensor_dofs_per_cell;
    const unsigned int n_q_points;
    bool enriched;
    std::vector<bool> enriched_components;
    AlignedVector<VectorizedArray<Number> > eddyvisc;
  };



template<int dim, int fe_degree, int fe_degree_xwall>
class XWall
{
//time-integration-level routines for xwall
public:
  XWall(const DoFHandler<dim> &dof_handler,
      MatrixFree<dim,double>* data,
      AlignedVector<VectorizedArray<double> > &element_volume,
      FEParameters<double> & fe_params);

  //initialize everything, e.g.
  //setup of wall distance
  //setup of communication of tauw to off-wall nodes
  //setup quadrature rules
  //possibly setup new matrixfree data object only including the xwall elements
  void initialize()
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "\nXWall Initialization:" << std::endl;

    //initialize wall distance and closest wall-node connectivity
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Initialize wall distance:...";
    InitWDist();
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << " done!" << std::endl;

    //initialize some vectors
    (*mydata).initialize_dof_vector(fe_params.xwallstatevec[1], 2);
    fe_params.xwallstatevec[1] = 1.0;
    tauw_n=fe_params.xwallstatevec[1];
  }

  //Update wall shear stress at the beginning of every time step
  void UpdateTauW(std::vector<parallel::distributed::Vector<double> > &solution_np);

  DoFHandler<dim> const & ReturnDofHandlerWallDistance() const {return dof_handler_wall_distance;}
//  const parallel::distributed::Vector<double> & ReturnWDist() const
//      {return wall_distance;}
//  const parallel::distributed::Vector<double> & ReturnTauW() const
//      {return tauw;}
  const parallel::distributed::Vector<double> & ReturnTauWN() const
      {return tauw_n;}

  const ConstraintMatrix & ReturnConstraintMatrix() const
      {return constraint_periodic;}

  const FE_Q<dim> & ReturnFE() const
      {return fe_wall_distance;}

  // fill the periodicity constraints given a level 0 periodicity structure
  void initialize_constraints(const std::vector< GridTools::PeriodicFacePair< typename Triangulation<dim>::cell_iterator > > &periodic_face_pair);
private:

  void InitWDist();

  //calculate wall shear stress based on current solution
  void CalculateWallShearStress(const std::vector<parallel::distributed::Vector<double> >   &src,
      parallel::distributed::Vector<double>      &dst);

  //element-level routines
  void local_rhs_dummy (const MatrixFree<dim,double>                &,
                        parallel::distributed::Vector<double>      &,
                        const std::vector<parallel::distributed::Vector<double> >    &,
                        const std::pair<unsigned int,unsigned int>          &) const;

  void local_rhs_wss_boundary_face(const MatrixFree<dim,double>              &data,
                    parallel::distributed::Vector<double>      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_dummy_face (const MatrixFree<dim,double>              &,
                parallel::distributed::Vector<double>      &,
                const std::vector<parallel::distributed::Vector<double> >  &,
                const std::pair<unsigned int,unsigned int>          &) const;

  void local_rhs_normalization_boundary_face(const MatrixFree<dim,double>              &data,
                    parallel::distributed::Vector<double>      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  //continuous vectors with linear interpolation
  FE_Q<dim> fe_wall_distance;
  DoFHandler<dim> dof_handler_wall_distance;
  FEParameters<double> & fe_params;
  parallel::distributed::Vector<double> tauw_boundary;
  std::vector<unsigned int> vector_to_tauw_boundary;
  parallel::distributed::Vector<double> tauw_n;
  MatrixFree<dim,double>* mydata;
//    parallel::distributed::Vector<double> &eddy_viscosity;
  AlignedVector<VectorizedArray<double> >& element_volume;
  ConstraintMatrix constraint_periodic;

public:

};

template<int dim, int fe_degree, int fe_degree_xwall>
XWall<dim,fe_degree,fe_degree_xwall>::XWall(const DoFHandler<dim> &dof_handler,
    MatrixFree<dim,double>* data,
    AlignedVector<VectorizedArray<double> > &element_volume,
    FEParameters<double> & fe_params)
:fe_wall_distance(QGaussLobatto<1>(1+1)),
 fe_params(fe_params),
 dof_handler_wall_distance(dof_handler.get_triangulation()),
 mydata(data),
 element_volume(element_volume)
{
//    dof_handler_wall_distance.distribute_dofs(fe_wall_distance);
//    dof_handler_wall_distance.distribute_mg_dofs(fe_wall_distance);
}

template<int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim,fe_degree,fe_degree_xwall>::InitWDist()
{
  // layout of aux_vector: 0-dim: normal, dim: distance, dim+1: nearest dof
  // index, dim+2: touch count (for computing weighted normals); normals not
  // currently used
  std::vector<parallel::distributed::Vector<double> > aux_vectors(dim+3);

  // store integer indices in a double. In order not to get overflow, we
  // need to make sure the global index fits into a double -> this limits
  // the maximum size in the dof indices to 2^53 (approx 10^15)
#ifdef DEAL_II_WITH_64BIT_INTEGERS
  AssertThrow(dof_handler_wall_distance.n_dofs() <
              (types::global_dof_index(1ull) << 53),
              ExcMessage("Sizes larger than 2^53 currently not supported"));
#endif

  IndexSet locally_relevant_set;
  DoFTools::extract_locally_relevant_dofs(dof_handler_wall_distance,
                                          locally_relevant_set);
  aux_vectors[0].reinit(dof_handler_wall_distance.locally_owned_dofs(),
                        locally_relevant_set, MPI_COMM_WORLD);
  for (unsigned int d=1; d<aux_vectors.size(); ++d)
    aux_vectors[d].reinit(aux_vectors[0]);

  // assign distance to close to infinity (we would like to use inf here but
  // there are checks in deal.II whether numbers are finite so we must use a
  // finite number here)
  const double unreached = 1e305;
  aux_vectors[dim] = unreached;

  // TODO: get the actual set of wall (Dirichlet) boundaries as input
  // arguments. Currently, this is matched with what is set in the outer
  // problem type.
  std::set<types::boundary_id> wall_boundaries;
  wall_boundaries.insert(0);

  // set the initial distance for the wall to zero and initialize the normal
  // directions
  {
    QGauss<dim-1> face_quadrature(1);
    FEFaceValues<dim> fe_face_values(fe_wall_distance, face_quadrature,
                                     update_normal_vectors);
    std::vector<types::global_dof_index> dof_indices(fe_wall_distance.dofs_per_face);
    int found = 0;
    for (typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_wall_distance.begin_active(); cell != dof_handler_wall_distance.end(); ++cell)
      if (cell->is_locally_owned())
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->at_boundary(f) &&
              wall_boundaries.find(cell->face(f)->boundary_id()) !=
              wall_boundaries.end())
            {
              found = 1;
              cell->face(f)->get_dof_indices(dof_indices);
              // get normal vector on face
              fe_face_values.reinit(cell, f);
              const Tensor<1,dim> normal = fe_face_values.normal_vector(0);
              for (unsigned int i=0; i<dof_indices.size(); ++i)
                {
                  for (unsigned int d=0; d<dim; ++d)
                    aux_vectors[d](dof_indices[i]) += normal[d];
                  aux_vectors[dim](dof_indices[i]) = 0.;
                  if(constraint_periodic.is_constrained(dof_indices[i]))
                    aux_vectors[dim+1](dof_indices[i]) = (*constraint_periodic.get_constraint_entries(dof_indices[i]))[0].first;
                  else
                    aux_vectors[dim+1](dof_indices[i]) = dof_indices[i];
                  aux_vectors[dim+2](dof_indices[i]) += 1.;
                }
            }
    int found_global = Utilities::MPI::sum(found,MPI_COMM_WORLD);
    //at least one processor has to have walls
    AssertThrow(found_global>0, ExcMessage("Could not find any wall. Aborting."));
    for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
      if (aux_vectors[dim+2].local_element(i) != 0)
        for (unsigned int d=0; d<dim; ++d)
          aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
  }

  // this algorithm finds the closest point on the interface by simply
  // searching locally on each element. This algorithm is only correct for
  // simple meshes (as it searches purely locally and can result in zig-zag
  // paths that are nowhere near optimal on general meshes) but it works in
  // parallel when the mesh can be arbitrarily decomposed among
  // processors. A generic class of algorithms to find the closest point on
  // the wall (not necessarily on a node of the mesh) is by some interface
  // evolution method similar to finding signed distance functions to a
  // given interface (see e.g. Sethian, Level Set Methods and Fast Marching
  // Methods, 2000, Chapter 6). But I do not know how to keep track of the
  // point of origin in those algorithms which is essential here, so skip
  // that for the moment. -- MK, Dec 2015

  // loop as long as we have untracked degrees of freedom. this loop should
  // terminate after a number of steps that is approximately half the width
  // of the mesh in elements
  while (aux_vectors[dim].linfty_norm() == unreached)
    {
      aux_vectors[dim+2] = 0.;
      for (unsigned int d=0; d<dim+2; ++d)
        aux_vectors[d].update_ghost_values();

      // get a pristine vector with the content of the distances at the
      // beginning of the step to distinguish which degrees of freedom were
      // already touched before the current loop and which are in the
      // process of being updated
      parallel::distributed::Vector<double> distances_step(aux_vectors[dim]);
      distances_step.update_ghost_values();

      AssertThrow(fe_wall_distance.dofs_per_cell ==
                  GeometryInfo<dim>::vertices_per_cell, ExcNotImplemented());
      Quadrature<dim> quadrature(fe_wall_distance.get_unit_support_points());
      FEValues<dim> fe_values(fe_wall_distance, quadrature, update_quadrature_points);
      std::vector<types::global_dof_index> dof_indices(fe_wall_distance.dofs_per_cell);

      // go through all locally owned and ghosted cells and compute the
      // nearest point from within the element. Since we have both ghosted
      // and owned cells, we can be sure that the locally owned vector
      // elements get the closest point from the neighborhood
      for (typename DoFHandler<dim>::active_cell_iterator cell =
             dof_handler_wall_distance.begin_active();
           cell != dof_handler_wall_distance.end(); ++cell)
        if (!cell->is_artificial())
          {
            bool cell_is_initialized = false;
            cell->get_dof_indices(dof_indices);

            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
              // point is unreached -> find the closest point within cell
              // that is already reached
              if (distances_step(dof_indices[v]) == unreached)
                {
                  for (unsigned int w=0; w<GeometryInfo<dim>::vertices_per_cell; ++w)
                    if (distances_step(dof_indices[w]) < unreached)
                      {
                        if (! cell_is_initialized)
                          {
                            fe_values.reinit(cell);
                            cell_is_initialized = true;
                          }

                        // here are the normal vectors in case they should
                        // be necessary in a refined version of the
                        // algorithm
                        /*
                        Tensor<1,dim> normal;
                        for (unsigned int d=0; d<dim; ++d)
                          normal[d] = aux_vectors[d](dof_indices[w]);
                        */
                        const Tensor<1,dim> distance_vec =
                          fe_values.quadrature_point(v) - fe_values.quadrature_point(w);
                        if (distances_step(dof_indices[w]) + distance_vec.norm() <
                            aux_vectors[dim](dof_indices[v]))
                          {
                            aux_vectors[dim](dof_indices[v]) =
                              distances_step(dof_indices[w]) + distance_vec.norm();
                            aux_vectors[dim+1](dof_indices[v]) =
                              aux_vectors[dim+1](dof_indices[w]);
                            for (unsigned int d=0; d<dim; ++d)
                              aux_vectors[d](dof_indices[v]) +=
                                aux_vectors[d](dof_indices[w]);
                            aux_vectors[dim+2](dof_indices[v]) += 1;
                          }
                      }
                }
          }
      for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
        if (aux_vectors[dim+2].local_element(i) != 0)
          for (unsigned int d=0; d<dim; ++d)
            aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
    }
  aux_vectors[dim+1].update_ghost_values();

  // at this point we could do a search for closer points in the
  // neighborhood of the points identified before (but it is probably quite
  // difficult to do and one needs to search in layers around a given point
  // to have all data available locally; I currently do not have a good idea
  // to sort out this mess and I am not sure whether we really need
  // something better than the local search above). -- MK, Dec 2015

  // copy the aux vector with extended ghosting into a vector that fits the
  // matrix-free partitioner
  (*mydata).initialize_dof_vector(fe_params.xwallstatevec[0], 2);
  AssertThrow(fe_params.xwallstatevec[0].local_size() == aux_vectors[dim].local_size(),
              ExcMessage("Vector sizes do not match, cannot import wall distances"));
  fe_params.xwallstatevec[0] = aux_vectors[dim];
  fe_params.xwallstatevec[0].update_ghost_values();

  IndexSet accessed_indices(aux_vectors[dim+1].size());
  {
    // copy the accumulated indices into an index vector
    std::vector<types::global_dof_index> my_indices;
    my_indices.reserve(aux_vectors[dim+1].local_size());
    for (unsigned int i=0; i<aux_vectors[dim+1].local_size(); ++i)
      my_indices.push_back(static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));
    // sort and compress out duplicates
    std::sort(my_indices.begin(), my_indices.end());
    my_indices.erase(std::unique(my_indices.begin(), my_indices.end()),
                     my_indices.end());
    accessed_indices.add_indices(my_indices.begin(),
                                 my_indices.end());
  }

  // create partitioner for exchange of ghost data (after having computed
  // the vector of wall shear stresses)
  std_cxx11::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner
    (new Utilities::MPI::Partitioner(dof_handler_wall_distance.locally_owned_dofs(),
                                     accessed_indices, MPI_COMM_WORLD));
  tauw_boundary.reinit(vector_partitioner);

  vector_to_tauw_boundary.resize(fe_params.xwallstatevec[0].local_size());
  for (unsigned int i=0; i<fe_params.xwallstatevec[0].local_size(); ++i)
    vector_to_tauw_boundary[i] = vector_partitioner->global_to_local
      (static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));

}

template<int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim,fe_degree,fe_degree_xwall>::UpdateTauW(std::vector<parallel::distributed::Vector<double> > &solution_np)
{
  //store old wall shear stress
  tauw_n.swap(fe_params.xwallstatevec[1]);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "\nCompute new tauw: ";
  CalculateWallShearStress(solution_np,fe_params.xwallstatevec[1]);
  //mean does not work currently because of all off-wall nodes in the vector
//    double tauwmean = tauw.mean_value();
//    std::cout << "mean = " << tauwmean << " ";

  double tauwmax = fe_params.xwallstatevec[1].linfty_norm();
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "max = " << tauwmax << " ";

  double minloc = 1e9;
  for(unsigned int i = 0; i < fe_params.xwallstatevec[1].local_size(); ++i)
  {
    if(fe_params.xwallstatevec[1].local_element(i)>0.0)
    {
      if(minloc > fe_params.xwallstatevec[1].local_element(i))
        minloc = fe_params.xwallstatevec[1].local_element(i);
    }
  }
  const double minglob = Utilities::MPI::min(minloc, MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "min = " << minglob << " ";
  if(not fe_params.variabletauw)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "(manually set to 1.0) ";
    fe_params.xwallstatevec[1] = 1.0;
  }
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl;
  fe_params.xwallstatevec[1].update_ghost_values();
}

template<int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim, fe_degree,fe_degree_xwall>::
CalculateWallShearStress (const std::vector<parallel::distributed::Vector<double> >   &src,
          parallel::distributed::Vector<double>      &dst)
{
  parallel::distributed::Vector<double> normalization;
  (*mydata).initialize_dof_vector(normalization, 2);
  parallel::distributed::Vector<double> force;
  (*mydata).initialize_dof_vector(force, 2);

  // initialize
  force = 0.0;
  normalization = 0.0;

  // run loop to compute the local integrals
  (*mydata).loop (&XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy,
      &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy_face,
      &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_wss_boundary_face,
            this, force, src);

  (*mydata).loop (&XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy,
      &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_dummy_face,
      &XWall<dim, fe_degree, fe_degree_xwall>::local_rhs_normalization_boundary_face,
            this, normalization, src);

  // run normalization
  double mean = 0.0;
  unsigned int count = 0;
  for(unsigned int i = 0; i < force.local_size(); ++i)
  {
    if(normalization.local_element(i)>0.0)
    {
      tauw_boundary.local_element(i) = force.local_element(i) / normalization.local_element(i);
      mean += tauw_boundary.local_element(i);
      count++;
    }
  }
  mean = Utilities::MPI::sum(mean,MPI_COMM_WORLD);
  count = Utilities::MPI::sum(count,MPI_COMM_WORLD);
  mean /= (double)count;
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "mean = " << mean << " ";

  // communicate the boundary values for the shear stress to the calling
  // processor and access the data according to the vector_to_tauw_boundary
  // field
  tauw_boundary.update_ghost_values();

  for (unsigned int i=0; i<fe_params.xwallstatevec[1].local_size(); ++i)
    dst.local_element(i) = (1.-fe_params.dtauw)*tauw_n.local_element(i)+fe_params.dtauw*tauw_boundary.local_element(vector_to_tauw_boundary[i]);
  dst.update_ghost_values();
}

template <int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim,fe_degree,fe_degree_xwall>::
local_rhs_dummy (const MatrixFree<dim,double>                &,
            parallel::distributed::Vector<double>      &,
            const std::vector<parallel::distributed::Vector<double> >  &,
            const std::pair<unsigned int,unsigned int>           &) const
{

}

template <int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim,fe_degree,fe_degree_xwall>::
local_rhs_wss_boundary_face (const MatrixFree<dim,double>             &data,
                       parallel::distributed::Vector<double>    &dst,
                       const std::vector<parallel::distributed::Vector<double> >  &src,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
{
#ifdef XWALL
  FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,double> fe_eval_xwall(data,wall_distance,tauw,true,0,3);
  FEFaceEvaluation<dim,1,n_q_points_1d_xwall,1,double> fe_eval_tauw(data,true,2,3);
#else
  FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,double> fe_eval_xwall(data,fe_params.xwallstatevec[0],fe_params.xwallstatevec[1],true,0,0);
  FEFaceEvaluation<dim,1,fe_degree+1,1,double> fe_eval_tauw(data,true,2,0);
#endif
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
    {
      fe_eval_xwall.reinit (face);
      fe_eval_xwall.evaluate_eddy_viscosity(src,face,fe_eval_xwall.read_cell_data(element_volume));
      fe_eval_tauw.reinit (face);

      fe_eval_xwall.read_dof_values(src,0,src,dim+1);
      fe_eval_xwall.evaluate(false,true);
      if(fe_eval_xwall.n_q_points != fe_eval_tauw.n_q_points)
        std::cerr << "\nwrong number of quadrature points" << std::endl;

      for(unsigned int q=0;q<fe_eval_xwall.n_q_points;++q)
      {
        Tensor<1, dim, VectorizedArray<double> > average_gradient = fe_eval_xwall.get_normal_gradient(q);

        VectorizedArray<double> tauwsc = make_vectorized_array<double>(0.0);
        tauwsc = average_gradient.norm();
        tauwsc *= fe_eval_xwall.eddyvisc[q];
        fe_eval_tauw.submit_value(tauwsc,q);
      }
      fe_eval_tauw.integrate(true,false);
      fe_eval_tauw.distribute_local_to_global(dst);
    }
  }
}

template <int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim,fe_degree,fe_degree_xwall>::
local_rhs_normalization_boundary_face (const MatrixFree<dim,double>             &data,
                       parallel::distributed::Vector<double>    &dst,
                       const std::vector<parallel::distributed::Vector<double> >  &,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
{
  FEFaceEvaluation<dim,1,fe_degree+1,1,double> fe_eval_tauw(data,true,2,0);
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
    {
      fe_eval_tauw.reinit (face);

      for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
        fe_eval_tauw.submit_value(make_vectorized_array<double>(1.0),q);

      fe_eval_tauw.integrate(true,false);
      fe_eval_tauw.distribute_local_to_global(dst);
    }
  }
}

template <int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim,fe_degree,fe_degree_xwall>::
local_rhs_dummy_face (const MatrixFree<dim,double>                 &,
              parallel::distributed::Vector<double>      &,
              const std::vector<parallel::distributed::Vector<double> >  &,
              const std::pair<unsigned int,unsigned int>          &) const
{

}

template <int dim, int fe_degree, int fe_degree_xwall>
void XWall<dim,fe_degree,fe_degree_xwall>::
initialize_constraints(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs)
{
  dof_handler_wall_distance.distribute_dofs(fe_wall_distance);
  dof_handler_wall_distance.distribute_mg_dofs(fe_wall_distance);

  IndexSet xwall_relevant_set;
  DoFTools::extract_locally_relevant_dofs(dof_handler_wall_distance,
                                          xwall_relevant_set);
  constraint_periodic.clear();
  constraint_periodic.reinit(xwall_relevant_set);
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> >
    periodic_face_pairs_dh(periodic_face_pairs.size());
  for (unsigned int i=0; i<periodic_face_pairs.size(); ++i)
    {
      GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> pair;
      pair.cell[0] = typename DoFHandler<dim>::cell_iterator
        (&periodic_face_pairs[i].cell[0]->get_triangulation(),
         periodic_face_pairs[i].cell[0]->level(),
         periodic_face_pairs[i].cell[0]->index(),
         &dof_handler_wall_distance);
      pair.cell[1] = typename DoFHandler<dim>::cell_iterator
        (&periodic_face_pairs[i].cell[1]->get_triangulation(),
         periodic_face_pairs[i].cell[1]->level(),
         periodic_face_pairs[i].cell[1]->index(),
         &dof_handler_wall_distance);
      pair.face_idx[0] = periodic_face_pairs[i].face_idx[0];
      pair.face_idx[1] = periodic_face_pairs[i].face_idx[1];
      pair.orientation = periodic_face_pairs[i].orientation;
      pair.matrix = periodic_face_pairs[i].matrix;
      periodic_face_pairs_dh[i] = pair;
    }
  DoFTools::make_periodicity_constraints<DoFHandler<dim> >(periodic_face_pairs_dh, constraint_periodic);
  DoFTools::make_hanging_node_constraints(dof_handler_wall_distance,
                                          constraint_periodic);

  constraint_periodic.close();
}

#endif /* INCLUDE_XWALL_H_ */
