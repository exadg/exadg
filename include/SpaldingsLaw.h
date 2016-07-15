/*
 * SpaldingsLaw.h
 *
 *  Created on: Jul 8, 2016
 *      Author: krank
 */

#ifndef INCLUDE_SPALDINGSLAW_H_
#define INCLUDE_SPALDINGSLAW_H_

#include <deal.II/base/exceptions.h>


//define a number of helper-functions to allow vectorized and non-vectorized V
namespace internalSpalding
{
  template <typename Number, typename V>
  V vectorize_or_not(Number val)
  {
    return make_vectorized_array<Number>(val);
  }
  //case when Number==V
  template <typename Number, typename V>
  V vectorize_or_not(V val)
  {
    return val;
  }

  template <int dim, typename Number>
  void zero_out(AlignedVector<Number> &, std::vector<bool> & , const unsigned int )
  {
    ;
  }
  template <int dim, typename Number>
  void zero_out(AlignedVector<Tensor<1,dim,Number> > &, std::vector<bool> & , const unsigned int )
  {
    ;
  }
  template <int dim, typename Number>
  void zero_out(AlignedVector<VectorizedArray<Number> > & qp_enrichment, std::vector<bool> & enriched_components, const unsigned int q)
  {
    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(not enriched_components.at(v))
      {
        qp_enrichment[q][v] = 0.0;
      }
    }
  }
  template <int dim, typename Number>
  void zero_out(AlignedVector<Tensor<1,dim,VectorizedArray<Number> > > & qp_grad_enrichment, std::vector<bool> & enriched_components, const unsigned int q)
  {
    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(not enriched_components.at(v))
      {
        for (unsigned int d = 0; d<dim; d++)
          qp_grad_enrichment[q][d][v] = 0.0;
      }
    }
  }

  template <typename Number>
  bool convergence_check (VectorizedArray<Number> & inc, VectorizedArray<Number> & fn, const int count, std::vector<bool> & enriched_components)
  {
    bool test = false;
    //do loop for all if one of the values is not converged
    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(enriched_components.at(v))
        if((std::abs(inc[v])>1.0E-14 && abs(fn[v])>1.0E-14&&1000>count))
          test=true;
    }
    return test;
  }
  template <typename Number>
  bool convergence_check (Number & inc, Number & fn, const int count, std::vector<bool> &)
  {
    bool test = false;
    if((std::abs(inc)>1.0E-14 && abs(fn)>1.0E-14&&1000>count))
      test=true;
    return test;
  }
}

template <int dim, typename Number, typename V >
class SpaldingsLawEvaluation
{
public:
  SpaldingsLawEvaluation (Number viscosity):
                viscosity(viscosity),
                k(0.41),
                km1(1.0/k),
                B(5.17),
                expmkmb(exp(-k*B)),
                density(internalSpalding::vectorize_or_not<Number,V>((Number)1.))
    {
      AssertThrow((not std::is_same<Number,float>::value),ExcMessage("If you are using float, the tolerances would probalby have to be adjusted"));
    };

  void reinit(AlignedVector<V > qp_wdist,
      AlignedVector<V > qp_tauw,
      AlignedVector<Tensor<1,dim,V > > qp_gradwdist,
      AlignedVector<Tensor<1,dim,V > > qp_gradtauw,
      unsigned int n_q_points,
      std::vector<bool> enriched_components)
  {
    qp_enrichment.resize(n_q_points);
    qp_grad_enrichment.resize(n_q_points);
    for(unsigned int q=0;q<n_q_points;++q)
    {
      qp_enrichment[q] =  EnrichmentShapeDer(qp_wdist[q], qp_tauw[q],
          qp_gradwdist[q], qp_gradtauw[q],&(qp_grad_enrichment[q]), enriched_components);

      internalSpalding::zero_out<dim, Number>(qp_enrichment, enriched_components, q);
      internalSpalding::zero_out<dim, Number>(qp_grad_enrichment, enriched_components, q);
//      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//      {
//        if(not enriched_components.at(v))
//        {
//          qp_enrichment[q][v] = 0.0;
//          for (unsigned int d = 0; d<dim; d++)
//            qp_grad_enrichment[q][d][v] = 0.0;
//        }
//
//      }
    }

  };

  void reinit(AlignedVector<V> qp_wdist,
      AlignedVector<V> qp_tauw,
      unsigned int n_q_points)
  {
    qp_enrichment.resize(n_q_points);
    std::vector<bool> enriched_components;
    enriched_components.resize(1,true);
    for(unsigned int q=0;q<n_q_points;++q)
    {
      const V utau=std::sqrt(qp_tauw[q]*internalSpalding::vectorize_or_not<Number,V>((Number)1.0)/density);
      qp_enrichment[q] = SpaldingsLaw(qp_wdist[q], utau, enriched_components);
    }
  };

  V enrichment(unsigned int q){return qp_enrichment[q];}
  Tensor<1,dim,V > enrichment_gradient(unsigned int q){return qp_grad_enrichment[q];}

private:

  const Number viscosity;
  const Number k;
  const Number km1;
  const Number B;
  const Number expmkmb;
  const VectorizedArray<Number> density;

  AlignedVector<V> qp_enrichment;
  AlignedVector<Tensor<1,dim,V> > qp_grad_enrichment;

  V EnrichmentShapeDer(
        V wdist, V tauw,
        Tensor<1,dim,V > gradwdist, Tensor<1,dim,V > gradtauw,
        Tensor<1,dim,V >* gradpsi, std::vector<bool> enriched_components)
  {
   //calculate transformation

   Tensor<1,dim,V > gradtrans;

   const V utau=std::sqrt(tauw*internalSpalding::vectorize_or_not<Number,V>((Number)1.0)/density);
   const V fac=internalSpalding::vectorize_or_not<Number,V>((Number)0.5)/std::sqrt(density*tauw);
   const V wdistfac=wdist*fac;

   for(unsigned int sdm=0;sdm < dim;++sdm)
     gradtrans[sdm]=(utau*gradwdist[sdm]+wdistfac*gradtauw[sdm])*internalSpalding::vectorize_or_not<Number,V>((Number)1.0/viscosity);

   //get enrichment function and scalar derivatives
     V psigp = SpaldingsLaw(wdist, utau, enriched_components)*internalSpalding::vectorize_or_not<Number,V>((Number)1.0);
     V derpsigpsc=DerSpaldingsLaw(psigp)*internalSpalding::vectorize_or_not<Number,V>((Number)1.0);
   //calculate final derivatives
   Tensor<1,dim,V > gradpsiq;
   for(int sdm=0;sdm < dim;++sdm)
   {
     gradpsiq[sdm]=derpsigpsc*gradtrans[sdm];
   }

   (*gradpsi)=gradpsiq;

    return psigp;
  }

  void initial_value(VectorizedArray<Number> &psi,const VectorizedArray<Number> &yplus, const std::vector<bool> & enriched_components)
  {
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
  }

  void initial_value(Number &psi,const Number &yplus, const std::vector<bool> & enriched_components)
  {
    if(yplus>11.0)//this is approximately where the intersection of log law and linear region lies
      psi=log(yplus)+B*k;
    else
      psi=yplus*k;
  }

  V SpaldingsLaw(V dist, V utau, std::vector<bool> enriched_components)
  {
    //watch out, this is not exactly Spalding's law but psi=u_+*k, which saves quite some multiplications
    const V yplus=dist*utau*internalSpalding::vectorize_or_not<Number,V>((Number)1.0/viscosity);
    V psi=internalSpalding::vectorize_or_not<Number,V>((Number)0.0);

    initial_value(psi, yplus, enriched_components);

    V inc=internalSpalding::vectorize_or_not<Number,V>((Number)10.0);
    V fn=internalSpalding::vectorize_or_not<Number,V>((Number)10.0);
    int count=0;
    bool converged = false;
    while(not converged)
    {
      V psiquad=psi*psi;
      V exppsi=std::exp(psi);
             fn=-yplus + psi*internalSpalding::vectorize_or_not<Number,V>(km1)
                       + internalSpalding::vectorize_or_not<Number,V>(expmkmb)
                       *(  exppsi
                           - internalSpalding::vectorize_or_not<Number,V>((Number)1.0)
                           - psi-psiquad*internalSpalding::vectorize_or_not<Number,V>((Number)0.5)
                           - psiquad*psi/internalSpalding::vectorize_or_not<Number,V>((Number)6.0)
                           - psiquad*psiquad/internalSpalding::vectorize_or_not<Number,V>((Number)24.0)
                        );
             V dfn= internalSpalding::vectorize_or_not<Number,V>(km1)
                       + internalSpalding::vectorize_or_not<Number,V>(expmkmb)
                       *(exppsi
                           - internalSpalding::vectorize_or_not<Number,V>((Number)1.0)
                           - psi-psiquad*internalSpalding::vectorize_or_not<Number,V>((Number)0.5)
                           - psiquad*psi/internalSpalding::vectorize_or_not<Number,V>((Number)6.0)
                        );

      inc=fn/dfn;

      psi-=inc;

//      bool test=false;
//      //do loop for all if one of the values is not converged
//      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
//      {
//        if(enriched_components.at(v))
//          if((std::abs(inc[v])>1.0E-14 && abs(fn[v])>1.0E-14&&1000>count++))
//            test=true;
//      }
      converged = not internalSpalding::convergence_check<Number> (inc,fn,count,enriched_components);
      count++;
    }

    return psi;

    //Reichardt's law 1951
    // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
  }

  VectorizedArray<Number> DerSpaldingsLaw(VectorizedArray<Number> psi)
  {
    //derivative with respect to y+!
    //spaldings law according to paper (derivative)
    return internalSpalding::vectorize_or_not<Number,V>((Number)1.0)/(internalSpalding::vectorize_or_not<Number,V>(km1)
        + internalSpalding::vectorize_or_not<Number,V>(expmkmb)*(std::exp(psi)
        - internalSpalding::vectorize_or_not<Number,V>((Number)1.0)
        - psi-psi*psi*internalSpalding::vectorize_or_not<Number,V>((Number)0.5)
        - psi*psi*psi/internalSpalding::vectorize_or_not<Number,V>((Number)6.0)));

  // Reichardt's law
  //  double yplus=dist*utau*viscinv_;
  //  return (0.4/(k_*(1.0+0.4*yplus))+7.8*(1.0/11.0*exp(-yplus/11.0)-1.0/11.0*exp(-yplus/3.0)+yplus/33.0*exp(-yplus/3.0)))*k_;
  }

};


#endif /* INCLUDE_SPALDINGSLAW_H_ */
