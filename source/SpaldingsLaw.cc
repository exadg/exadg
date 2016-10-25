/*
 * SpaldingsLaw.cc
 *
 *  Created on: Aug 22, 2016
 *      Author: krank
 */

#include <SpaldingsLaw.h>

//define a number of helper-functions to allow vectorized and non-vectorized V
namespace internalSpalding
{
  template <typename Number, typename V>
  V vectorize_or_not(Number val)
  {
    return make_vectorized_array<Number>(val);
  }
  template<>
  double vectorize_or_not<double, double>(double val)
  {
    return val;
  }

  template <int dim, typename Number>
  void zero_out(Number &, const std::vector<bool> &)
  {
    ;
  }
  template <int dim, typename Number>
  void zero_out(Tensor<1,dim,Number> &, const std::vector<bool> &)
  {
    ;
  }
  template <int dim, typename Number>
  void zero_out(VectorizedArray<Number> & qp_enrichment, const std::vector<bool> & enriched_components)
  {
    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(not enriched_components[v])
      {
        qp_enrichment[v] = 0.0;
      }
    }
  }
  template <int dim, typename Number>
  void zero_out(Tensor<1,dim,VectorizedArray<Number> > & qp_grad_enrichment, const std::vector<bool> & enriched_components)
  {
    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(not enriched_components[v])
      {
        for (unsigned int d = 0; d<dim; d++)
          qp_grad_enrichment[d][v] = 0.0;
      }
    }
  }

  template <typename Number>
  bool convergence_check (VectorizedArray<Number> & inc, VectorizedArray<Number> & fn, const int count, const std::vector<bool> & enriched_components)
  {
    bool test = false;
    //do loop for all if one of the values is not converged
    for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(enriched_components[v])
        if((std::abs(inc[v])>1.0E-14 && abs(fn[v])>1.0E-14&&1000>count))
          test=true;
    }
    return test;
  }
  template <typename Number>
  bool convergence_check (Number & inc, Number & fn, const int count, const std::vector<bool> &)
  {
    bool test = false;
    if((std::abs(inc)>1.0E-14 && abs(fn)>1.0E-14&&1000>count))
      test=true;
    return test;
  }
}

template <int dim, typename Number, typename V >
void SpaldingsLawEvaluation<dim,Number,V>::reinit_zero(const unsigned int n_q_points)
{
  qp_enrichment.resize(n_q_points);
  qp_grad_enrichment.resize(n_q_points);
  std::vector<bool> enriched_components(VectorizedArray<Number>::n_array_elements);
  for (unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; v++)
    enriched_components[v] = false;
  for(unsigned int q=0;q<n_q_points;++q)
  {
    internalSpalding::zero_out<dim, Number>(qp_enrichment[q], enriched_components);
    internalSpalding::zero_out<dim, Number>(qp_grad_enrichment[q], enriched_components);
  }
}

template <int dim, typename Number, typename V >
void SpaldingsLawEvaluation<dim,Number,V>::reinit(const AlignedVector<V > & qp_wdist,
                                                  const AlignedVector<V > & qp_tauw,
                                                  const AlignedVector<Tensor<1,dim,V > > & qp_gradwdist,
                                                  const AlignedVector<Tensor<1,dim,V > > & qp_gradtauw,
                                                  const unsigned int n_q_points,
                                                  const std::vector<bool> & enriched_components)
{
  qp_enrichment.resize(n_q_points);
  qp_grad_enrichment.resize(n_q_points);
  for(unsigned int q=0;q<n_q_points;++q)
  {
    qp_enrichment[q] =  EnrichmentShapeDer(qp_wdist[q], qp_tauw[q],
        qp_gradwdist[q], qp_gradtauw[q],qp_grad_enrichment[q], enriched_components);

    internalSpalding::zero_out<dim, Number>(qp_enrichment[q], enriched_components);
    internalSpalding::zero_out<dim, Number>(qp_grad_enrichment[q], enriched_components);
  }
}

template <int dim, typename Number, typename V >
void SpaldingsLawEvaluation<dim,Number,V>::reinit(const AlignedVector<V> & qp_wdist,
                                                  const AlignedVector<V> & qp_tauw,
                                                  const unsigned int n_q_points)
{
  qp_enrichment.resize(n_q_points);
  std::vector<bool> enriched_components;
  enriched_components.resize(VectorizedArray<Number>::n_array_elements,true);
  for(unsigned int q=0;q<n_q_points;++q)
  {
    const V utau=std::sqrt(qp_tauw[q]*internalSpalding::vectorize_or_not<Number,V>((Number)1.0)/density);
    qp_enrichment[q] = SpaldingsLaw(qp_wdist[q], utau, enriched_components);
  }
}
template <int dim, typename Number, typename V >
void SpaldingsLawEvaluation<dim,Number,V>::reinit(const AlignedVector<VectorizedArray<double> > & ,
                                                  const AlignedVector<Tensor<1,dim,VectorizedArray<double> > > & ,
                                                  const unsigned int )
{
  //covers all the template-cases that are not used
  AssertThrow(false,ExcInternalError());
}

template <int dim, typename Number, typename V >
V SpaldingsLawEvaluation<dim,Number,V>::EnrichmentShapeDer(
        const V & wdist, const V & tauw,
        const Tensor<1,dim,V > & gradwdist, const Tensor<1,dim,V > & gradtauw,
        Tensor<1,dim,V > & gradpsi, const std::vector<bool> & enriched_components)
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
 for(int sdm=0;sdm < dim;++sdm)
 {
   gradpsi[sdm]=derpsigpsc*gradtrans[sdm];
 }

  return psigp;
}

template <int dim, typename Number, typename V >
void SpaldingsLawEvaluation<dim,Number,V>::initial_value(VectorizedArray<Number> &psi,const VectorizedArray<Number> &yplus, const std::vector<bool> & enriched_components)
{
  for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
  {
    if(enriched_components[v])
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

template <int dim, typename Number, typename V >
void SpaldingsLawEvaluation<dim,Number,V>::initial_value(Number &psi,const Number &yplus, const std::vector<bool> &)
{
  if(yplus>11.0)//this is approximately where the intersection of log law and linear region lies
    psi=log(yplus)+B*k;
  else
    psi=yplus*k;
}

template <int dim, typename Number, typename V >
V SpaldingsLawEvaluation<dim,Number,V>::SpaldingsLaw(const V dist, const V utau, const std::vector<bool> & enriched_components)
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
                  - psi-psiquad*num2m1
                  - psiquad*psi*num6m1
                  - psiquad*psiquad/internalSpalding::vectorize_or_not<Number,V>((Number)24.0)
               );
    V dfn= internalSpalding::vectorize_or_not<Number,V>(km1)
              + internalSpalding::vectorize_or_not<Number,V>(expmkmb)
              *(exppsi
                  - internalSpalding::vectorize_or_not<Number,V>((Number)1.0)
                  - psi-psiquad*num2m1
                  - psiquad*psi*num6m1
               );

    inc=fn/dfn;

    psi-=inc;

    converged = not internalSpalding::convergence_check<Number> (inc,fn,count,enriched_components);
    count++;
  }

  return psi;

  //Reichardt's law 1951
  // return (1.0/k_*log(1.0+0.4*yplus)+7.8*(1.0-exp(-yplus/11.0)-(yplus/11.0)*exp(-yplus/3.0)))*k_;
}

template <int dim, typename Number, typename V >
V SpaldingsLawEvaluation<dim,Number,V>::DerSpaldingsLaw(const V psi)
{
  //derivative with respect to y+!
  //spaldings law according to paper (derivative)
  return internalSpalding::vectorize_or_not<Number,V>((Number)1.0)/(internalSpalding::vectorize_or_not<Number,V>(km1)
      + internalSpalding::vectorize_or_not<Number,V>(expmkmb)*(std::exp(psi)
      - internalSpalding::vectorize_or_not<Number,V>((Number)1.0)
      - psi-psi*psi*num2m1
      - psi*psi*psi*num6m1));

// Reichardt's law
//  double yplus=dist*utau*viscinv_;
//  return (0.4/(k_*(1.0+0.4*yplus))+7.8*(1.0/11.0*exp(-yplus/11.0)-1.0/11.0*exp(-yplus/3.0)+yplus/33.0*exp(-yplus/3.0)))*k_;
}

template <>
void SpaldingsLawEvaluation<3,double,VectorizedArray<double> >::
reinit(const AlignedVector<VectorizedArray<double> > & enrichment_in,
              const AlignedVector<Tensor<1,3,VectorizedArray<double>  > > & enrichment_gradient_in,
              const unsigned int n_q_points)
{
  AssertThrow(enrichment_in.size()==n_q_points,ExcInternalError());
  AssertThrow(enrichment_gradient_in.size()==n_q_points,ExcInternalError());
  qp_enrichment.resize(n_q_points);
  qp_grad_enrichment.resize(n_q_points);
  for(unsigned int q = 0; q < n_q_points; q++)
  {
    qp_enrichment[q] = enrichment_in[q];
    qp_grad_enrichment[q] = enrichment_gradient_in[q];
  }
}

template <>
void SpaldingsLawEvaluation<2,double,VectorizedArray<double> >::
reinit(const AlignedVector<VectorizedArray<double> > & enrichment_in,
              const AlignedVector<Tensor<1,2,VectorizedArray<double>  > > & enrichment_gradient_in,
              const unsigned int n_q_points)
{
  AssertThrow(enrichment_in.size()==n_q_points,ExcInternalError());
  AssertThrow(enrichment_gradient_in.size()==n_q_points,ExcInternalError());
  qp_enrichment.resize(n_q_points);
  qp_grad_enrichment.resize(n_q_points);
  for(unsigned int q = 0; q < n_q_points; q++)
  {
    qp_enrichment[q] = enrichment_in[q];
    qp_grad_enrichment[q] = enrichment_gradient_in[q];
  }
}

// explicit instantiation
template VectorizedArray<double> internalSpalding::vectorize_or_not<double,VectorizedArray<double> > (double val);
template double internalSpalding::vectorize_or_not<double,double> (double val);
template VectorizedArray<float> internalSpalding::vectorize_or_not<float,VectorizedArray<float> >(float val);
template class SpaldingsLawEvaluation<2,double,VectorizedArray<double> >;
template class SpaldingsLawEvaluation<2,float,VectorizedArray<float> >;
template class SpaldingsLawEvaluation<2,double,double>;
template class SpaldingsLawEvaluation<3,double,VectorizedArray<double> >;
template class SpaldingsLawEvaluation<3,float,VectorizedArray<float> >;
template class SpaldingsLawEvaluation<3,double,double>;
