/*
 * SpaldingsLaw.h
 *
 *  Created on: Jul 8, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_SPALDINGSLAW_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_SPALDINGSLAW_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

using namespace dealii;
namespace internalSpalding
{
template<typename Number, typename V>
V
vectorize_or_not(Number val);
}

template<int dim, typename Number, typename V>
class SpaldingsLawEvaluation
{
public:
  SpaldingsLawEvaluation(const Number viscosity)
    : viscosity(viscosity),
      k(0.41),
      km1(1.0 / k),
      B(5.17),
      expmkmb(exp(-k * B)),
      num2m1(internalSpalding::vectorize_or_not<Number, V>((Number)0.5)),
      num6m1(internalSpalding::vectorize_or_not<Number, V>((Number)1. / (Number)6.0)),
      density(internalSpalding::vectorize_or_not<Number, V>((Number)1.))
  {
    AssertThrow((not std::is_same<Number, float>::value),
                ExcMessage(
                  "If you are using float, the tolerances would probalby have to be adjusted"));
  };

  void
  reinit_zero(const unsigned int n_q_points);

  void
  reinit(const AlignedVector<V> &                 qp_wdist,
         const AlignedVector<V> &                 qp_tauw,
         const AlignedVector<Tensor<1, dim, V>> & qp_gradwdist,
         const AlignedVector<Tensor<1, dim, V>> & qp_gradtauw,
         const unsigned int                       n_q_points,
         const std::vector<bool> &                enriched_components);

  void
  reinit(const AlignedVector<V> & qp_wdist,
         const AlignedVector<V> & qp_tauw,
         const unsigned int       n_q_points);

  void
  reinit(const AlignedVector<VectorizedArray<double>> &,
         const AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &,
         const unsigned int);

  V
  enrichment(unsigned int q)
  {
    return qp_enrichment[q];
  }
  Tensor<1, dim, V>
  enrichment_gradient(unsigned int q)
  {
    return qp_grad_enrichment[q];
  }

private:
  const Number viscosity;
  const Number k;
  const Number km1;
  const Number B;
  const Number expmkmb;
  const V      num2m1;
  const V      num6m1;
  const V      density;

  AlignedVector<V>                 qp_enrichment;
  AlignedVector<Tensor<1, dim, V>> qp_grad_enrichment;

  V
  EnrichmentShapeDer(const V &                 wdist,
                     const V &                 tauw,
                     const Tensor<1, dim, V> & gradwdist,
                     const Tensor<1, dim, V> & gradtauw,
                     Tensor<1, dim, V> &       gradpsi,
                     const std::vector<bool> & enriched_components);

  void
  initial_value(VectorizedArray<Number> &       psi,
                const VectorizedArray<Number> & yplus,
                const std::vector<bool> &       enriched_components);

  void
  initial_value(Number & psi, const Number & yplus, const std::vector<bool> &);

  V
  SpaldingsLaw(const V dist, const V utau, const std::vector<bool> & enriched_components);

  V
  DerSpaldingsLaw(const V psi);
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_XWALL_SPALDINGSLAW_H_ */
