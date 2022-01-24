/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_EXPLICIT_RUNGE_KUTTA_H_
#define INCLUDE_CONVECTION_DIFFUSION_EXPLICIT_RUNGE_KUTTA_H_

namespace ExaDG
{
using namespace dealii;

template<typename Operator, typename VectorType>
class ExplicitTimeIntegrator
{
public:
  ExplicitTimeIntegrator(std::shared_ptr<Operator> operator_in) : underlying_operator(operator_in)
  {
  }

  virtual ~ExplicitTimeIntegrator()
  {
  }

  virtual void
  solve_timestep(VectorType & dst, VectorType & src, double const time, double const time_step) = 0;

  virtual unsigned int
  get_order() const = 0;

protected:
  std::shared_ptr<Operator> underlying_operator;
};

/*
 *  Classical, explicit Runge-Kutta time integration schemes of order 1-4
 */
template<typename Operator, typename VectorType>
class ExplicitRungeKuttaTimeIntegrator : public ExplicitTimeIntegrator<Operator, VectorType>
{
public:
  // Constructor
  ExplicitRungeKuttaTimeIntegrator(unsigned int                    order_time_integrator,
                                   std::shared_ptr<Operator> const operator_in)
    : ExplicitTimeIntegrator<Operator, VectorType>(operator_in), order(order_time_integrator)
  {
    // initialize vectors
    if(order >= 2)
      this->underlying_operator->initialize_dof_vector(vec_rhs);
    if(order >= 3)
      this->underlying_operator->initialize_dof_vector(vec_temp);
  }

  void
  solve_timestep(VectorType & dst,
                 VectorType & src,
                 double const time,
                 double const time_step) final
  {
    if(order == 1) // explicit Euler method
    {
      /*
       * Butcher table
       *
       *   0 |
       *  --------
       *     | 1
       *
       */

      this->underlying_operator->evaluate(dst, src, time);
      dst.sadd(time_step, 1., src);
    }
    else if(order == 2) // Runge-Kutta method of order 2
    {
      /*
       * Butcher table
       *
       *   0   |
       *   1/2 | 1/2
       *  ---------------
       *       |  0   1
       *
       */

      // stage 1
      this->underlying_operator->evaluate(vec_rhs, src, time);

      // stage 2
      vec_rhs.sadd(time_step / 2., 1., src);
      this->underlying_operator->evaluate(dst, vec_rhs, time + time_step / 2.);
      dst.sadd(time_step, 1., src);
    }
    else if(order == 3) // Heun's method of order 3
    {
      /*
       * Butcher table
       *
       *   0   |
       *   1/3 | 1/3
       *   2/3 |  0  2/3
       *  -------------------
       *       | 1/4  0  3/4
       *
       */

      dst = src;

      // stage 1
      this->underlying_operator->evaluate(vec_temp, src, time);
      dst.add(1. * time_step / 4., vec_temp);

      // stage 2
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step / 3., vec_temp);
      this->underlying_operator->evaluate(vec_temp, vec_rhs, time + time_step / 3.);

      // stage 3
      vec_rhs.equ(1., src);
      vec_rhs.add(2. * time_step / 3., vec_temp);
      this->underlying_operator->evaluate(vec_temp, vec_rhs, time + 2. * time_step / 3.);
      dst.add(3. * time_step / 4., vec_temp);
    }
    else if(order == 4) // classical 4th order Runge-Kutta method
    {
      /*
       * Butcher table
       *
       *   0   |
       *   1/2 | 1/2
       *   1/2 |  0  1/2
       *    1  |  0   0   1
       *  -----------------------
       *       | 1/6 2/6 2/6 1/6
       *
       */

      dst = src;

      // stage 1
      this->underlying_operator->evaluate(vec_temp, src, time);
      dst.add(time_step / 6., vec_temp);

      // stage 2
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step / 2., vec_temp);
      this->underlying_operator->evaluate(vec_temp, vec_rhs, time + time_step / 2.);
      dst.add(time_step / 3., vec_temp);

      // stage 3
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step / 2., vec_temp);
      this->underlying_operator->evaluate(vec_temp, vec_rhs, time + time_step / 2.);
      dst.add(time_step / 3., vec_temp);

      // stage 4
      vec_rhs.equ(1., src);
      vec_rhs.add(time_step, vec_temp);
      this->underlying_operator->evaluate(vec_temp, vec_rhs, time + time_step);
      dst.add(time_step / 6., vec_temp);
    }
    else
    {
      AssertThrow(order <= 4,
                  ExcMessage("Explicit Runge-Kutta method only implemented for order <= 4!"));
    }
  }

  unsigned int
  get_order() const final
  {
    return order;
  }

private:
  unsigned int order;

  VectorType vec_rhs, vec_temp;
};



/****************************************************************************************
 *                                                                                      *
 *  Low-storage, explicit Runge-Kutta methods according to                              *
 *                                                                                      *
 *  Kennedy et al. (2000, "Low-storage, explicit Runge-Kutta schemes for the            *
 *                  compressible Navier-Stokes equations")                              *
 *                                                                                      *
 *  classification of methods:                                                          *
 *                                                                                      *
 *    RKq(p)s[rR]X where ...                                                            *
 *                                                                                      *
 *  ... q is the order of the main scheme                                               *
 *  ... p is the order of the embedded scheme                                           *
 *  ... s is the number of stages                                                       *
 *  ... r is the number of registers used                                               *
 *  ... X denotes the optimization strategy/criterion, e.g,                             *
 *                                                                                      *
 *      X = C: linear-stability-error compromise                                        *
 *      X = S: maximum linear stability                                                 *
 *      X = F: FSAL - first same as last                                                *
 *      X = M: minimum truncation error                                                 *
 *      X = N: maximum nonlinear stability                                              *
 *      X = P: minimum phase error                                                      *
 *                                                                                      *
 *  For applications where the time step is selected according to stability             *
 *  considerations (time step size close to CFL limit, temporal error small enough),    *
 *  schemes of type C or type S should be used.                                         *
 *                                                                                      *
 ****************************************************************************************/

/*
 *  Low storage Runge-Kutta method of order 3 with 4 stages and 2 registers according to
 *  Kennedy et al. (2000), where this method is denoted as RK3(2)4[2R+]C,
 *  see Table 1 on page 189 for the coefficients.
 */
template<typename Operator, typename VectorType>
class LowStorageRK3Stage4Reg2C : public ExplicitTimeIntegrator<Operator, VectorType>
{
public:
  LowStorageRK3Stage4Reg2C(std::shared_ptr<Operator> const operator_in)
    : ExplicitTimeIntegrator<Operator, VectorType>(operator_in)
  {
  }

  void
  solve_timestep(VectorType & vec_np,
                 VectorType & vec_n,
                 double const time,
                 double const time_step) final
  {
    if(!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np);
    }

    /*
     * Butcher table
     *
     *  c1 |
     *  c2 | a21
     *  c3 |  b1 a32
     *  c4 |  b1  b2 a43
     *  ---------------------
     *     |  b1  b2  b3  b4
     *
     */

    double const a21 = 11847461282814. / 36547543011857.;
    double const a32 = 3943225443063. / 7078155732230.;
    double const a43 = -346793006927. / 4029903576067.;

    double const b1 = 1017324711453. / 9774461848756.;
    double const b2 = 8237718856693. / 13685301971492.;
    double const b3 = 57731312506979. / 19404895981398.;
    double const b4 = -101169746363290. / 37734290219643.;

    double const c1 = 0.;
    double const c2 = a21;
    double const c3 = b1 + a32;
    double const c4 = b1 + b2 + a43;

    // stage 1
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_1 */, time + c1 * time_step);
    vec_n.add(a21 * time_step, vec_tmp1); /* = u_2 */
    vec_np = vec_n;
    vec_np.add((b1 - a21) * time_step, vec_tmp1); /* = u_p */

    // stage 2
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_2 */, time + c2 * time_step);
    vec_np.add(a32 * time_step, vec_tmp1); /* = u_3 */
    vec_n = vec_np;
    vec_n.add((b2 - a32) * time_step, vec_tmp1); /* = u_p */

    // stage 3
    this->underlying_operator->evaluate(vec_tmp1, vec_np /* u_3 */, time + c3 * time_step);
    vec_n.add(a43 * time_step, vec_tmp1); /* = u_4 */
    vec_np = vec_n;
    vec_np.add((b3 - a43) * time_step, vec_tmp1); /* = u_p */

    // stage 4
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_3 */, time + c4 * time_step);
    vec_np.add(b4 * time_step, vec_tmp1); /* = u_p */
  }

  unsigned int
  get_order() const final
  {
    return 3;
  }

private:
  VectorType vec_tmp1;
};


/*
 *  Low storage Runge-Kutta method of order 4 with 5 stages and 2 registers according to
 *  Kennedy et al. (2000), where this method is denoted as RK4(3)5[2R+]C,
 *  see Table 1 on page 189 for the coefficients.
 */
template<typename Operator, typename VectorType>
class LowStorageRK4Stage5Reg2C : public ExplicitTimeIntegrator<Operator, VectorType>
{
public:
  LowStorageRK4Stage5Reg2C(std::shared_ptr<Operator> const operator_in)
    : ExplicitTimeIntegrator<Operator, VectorType>(operator_in)
  {
  }

  void
  solve_timestep(VectorType & vec_np,
                 VectorType & vec_n,
                 double const time,
                 double const time_step) final
  {
    if(!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np);
    }

    /*
     * Butcher table
     *
     *  c1 |
     *  c2 | a21
     *  c3 |  b1 a32
     *  c4 |  b1  b2 a43
     *  c5 |  b1  b2  b3 a54
     *  ------------------------
     *     |  b1  b2  b3  b4  b5
     *
     */

    double const a21 = 970286171893. / 4311952581923.;
    double const a32 = 6584761158862. / 12103376702013.;
    double const a43 = 2251764453980. / 15575788980749.;
    double const a54 = 26877169314380. / 34165994151039.;

    double const b1 = 1153189308089. / 22510343858157.;
    double const b2 = 1772645290293. / 4653164025191.;
    double const b3 = -1672844663538. / 4480602732383.;
    double const b4 = 2114624349019. / 3568978502595.;
    double const b5 = 5198255086312. / 14908931495163.;

    double const c1 = 0.;
    double const c2 = a21;
    double const c3 = b1 + a32;
    double const c4 = b1 + b2 + a43;
    double const c5 = b1 + b2 + b3 + a54;

    // stage 1
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_1 */, time + c1 * time_step);
    vec_n.add(a21 * time_step, vec_tmp1); /* = u_2 */
    vec_np = vec_n;
    vec_np.add((b1 - a21) * time_step, vec_tmp1); /* = u_p */

    // stage 2
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_2 */, time + c2 * time_step);
    vec_np.add(a32 * time_step, vec_tmp1); /* = u_3 */
    vec_n = vec_np;
    vec_n.add((b2 - a32) * time_step, vec_tmp1); /* = u_p */

    // stage 3
    this->underlying_operator->evaluate(vec_tmp1, vec_np /* u_3 */, time + c3 * time_step);
    vec_n.add(a43 * time_step, vec_tmp1); /* = u_4 */
    vec_np = vec_n;
    vec_np.add((b3 - a43) * time_step, vec_tmp1); /* = u_p */

    // stage 4
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_3 */, time + c4 * time_step);
    vec_np.add(a54 * time_step, vec_tmp1); /* = u_5 */
    vec_n = vec_np;
    vec_n.add((b4 - a54) * time_step, vec_tmp1); /* = u_p */

    // stage 5
    this->underlying_operator->evaluate(vec_tmp1, vec_np /* u_4 */, time + c5 * time_step);
    vec_np = vec_n;
    vec_np.add(b5 * time_step, vec_tmp1);
  }

  unsigned int
  get_order() const final
  {
    return 4;
  }

private:
  VectorType vec_tmp1;
};

/*
 *  Low storage Runge-Kutta method of order 4 with 5 stages and 3 registers according to
 *  Kennedy et al. (2000), where this method is denoted as RK4(3)5[3R+]C,
 *  see Table 2 on page 190 for the coefficients.
 */
template<typename Operator, typename VectorType>
class LowStorageRK4Stage5Reg3C : public ExplicitTimeIntegrator<Operator, VectorType>
{
public:
  LowStorageRK4Stage5Reg3C(std::shared_ptr<Operator> const operator_in)
    : ExplicitTimeIntegrator<Operator, VectorType>(operator_in)
  {
  }

  void
  solve_timestep(VectorType & vec_np,
                 VectorType & vec_n,
                 double const time,
                 double const time_step) final
  {
    if(!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np, true);
      vec_tmp2.reinit(vec_np, true);
    }

    /*
     * Butcher table
     *
     *  c1 |
     *  c2 | a21
     *  c3 | a31 a32
     *  c4 |  b1 a42 a43
     *  c5 |  b1  b2 a53 a54
     *  ------------------------
     *     |  b1  b2  b3  b4  b5
     *
     */
    double const a21 = 2365592473904. / 8146167614645.;
    double const a32 = 4278267785271. / 6823155464066.;
    double const a43 = 2789585899612. / 8986505720531.;
    double const a54 = 15310836689591. / 24358012670437.;

    double const a31 = -722262345248. / 10870640012513.;
    double const a42 = 1365858020701. / 8494387045469.;
    double const a53 = 3819021186. / 2763618202291.;

    double const b1 = 846876320697. / 6523801458457.;
    double const b2 = 3032295699695. / 12397907741132.;
    double const b3 = 612618101729. / 6534652265123.;
    double const b4 = 1155491934595. / 2954287928812.;
    double const b5 = 707644755468. / 5028292464395.;

    double const c1 = 0.;
    double const c2 = a21;
    double const c3 = a31 + a32;
    double const c4 = b1 + a42 + a43;
    double const c5 = b1 + b2 + a53 + a54;

    // stage 1
    this->underlying_operator->evaluate(vec_np /* F_1 */, vec_n /* u_1 */, time + c1 * time_step);
    vec_n.add(a21 * time_step, vec_np /* F_1 */); /* = u_2 */
    vec_tmp1 = vec_n /* u_2 */;
    vec_tmp1.add((b1 - a21) * time_step, vec_np /* F_1 */); /* = u_p */

    // stage 2
    this->underlying_operator->evaluate(vec_tmp2 /* F_2 */, vec_n /* u_2 */, time + c2 * time_step);
    vec_tmp1.add(a32 * time_step,
                 vec_tmp2 /* F_2 */,
                 (a31 - b1) * time_step,
                 vec_np /* F_1 */); /* = u_3 */
    vec_np.sadd((b1 - a31) * time_step, 1, vec_tmp1 /* u_3 */);
    vec_np.add((b2 - a32) * time_step, vec_tmp2 /* F_2 */); /* u_p */

    // stage 3
    this->underlying_operator->evaluate(vec_n /* F_3 */, vec_tmp1 /* u_3 */, time + c3 * time_step);
    vec_np.add(a43 * time_step,
               vec_n /* F_3 */,
               (a42 - b2) * time_step,
               vec_tmp2 /* F_2 */); /* = u_4 */
    vec_tmp2.sadd((b2 - a42) * time_step, 1, vec_np /* u_4 */);
    vec_tmp2.add((b3 - a43) * time_step, vec_n /* F_3 */); /* u_p */

    // stage 4
    this->underlying_operator->evaluate(vec_tmp1 /* F_4 */,
                                        vec_np /* u_4 */,
                                        time + c4 * time_step);
    vec_tmp2.add(a54 * time_step,
                 vec_tmp1 /* F_4 */,
                 (a53 - b3) * time_step,
                 vec_n /* F_3 */); /* = u_5 */
    vec_n.sadd((b3 - a53) * time_step, 1, vec_tmp2 /* u_5 */);
    vec_n.add((b4 - a54) * time_step, vec_tmp1 /* F_4 */); /* = u_p */

    // stage 5
    this->underlying_operator->evaluate(vec_tmp1 /* F_5 */,
                                        vec_tmp2 /* u_5 */,
                                        time + c5 * time_step);
    vec_np = vec_n;
    vec_np.add(b5 * time_step, vec_tmp1 /* F_5 */); /* = u_p */
  }

  unsigned int
  get_order() const final
  {
    return 4;
  }

private:
  VectorType vec_tmp1, vec_tmp2;
};

/*
 *  Low storage Runge-Kutta method of order 5 with 9 stages and 2 registers according to
 *  Kennedy et al. (2000), where this method is denoted as RK5(4)9[2R+]S,
 *  see Table 1 on page 189 for the coefficients.
 */
template<typename Operator, typename VectorType>
class LowStorageRK5Stage9Reg2S : public ExplicitTimeIntegrator<Operator, VectorType>
{
public:
  LowStorageRK5Stage9Reg2S(std::shared_ptr<Operator> const operator_in)
    : ExplicitTimeIntegrator<Operator, VectorType>(operator_in)
  {
  }

  void
  solve_timestep(VectorType & vec_np, VectorType & vec_n, double const time, double const time_step)
  {
    if(!vec_tmp1.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp1.reinit(vec_np);
    }

    /*
     * Butcher table
     *
     *  c1 |
     *  c2 | a21
     *  c3 |  b1 a32
     *  c4 |  b1  b2 a43
     *  c5 |  b1  b2  b3 a54
     *  c6 |  b1  b2  b3  b4 a65
     *  c7 |  b1  b2  b3  b4  b5 a76
     *  c8 |  b1  b2  b3  b4  b5  b6 a87
     *  c9 |  b1  b2  b3  b4  b5  b6  b7 a98
     *  -----------------------------------------
     *     |  b1  b2  b3  b4  b5  b6  b7  b8  b9
     *
     */

    double const a21 = 1107026461565. / 5417078080134.;
    double const a32 = 38141181049399. / 41724347789894.;
    double const a43 = 493273079041. / 11940823631197.;
    double const a54 = 1851571280403. / 6147804934346.;
    double const a65 = 11782306865191. / 62590030070788.;
    double const a76 = 9452544825720. / 13648368537481.;
    double const a87 = 4435885630781. / 26285702406235.;
    double const a98 = 2357909744247. / 11371140753790.;

    double const b1 = 2274579626619. / 23610510767302.;
    double const b2 = 693987741272. / 12394497460941.;
    double const b3 = -347131529483. / 15096185902911.;
    double const b4 = 1144057200723. / 32081666971178.;
    double const b5 = 1562491064753. / 11797114684756.;
    double const b6 = 13113619727965. / 44346030145118.;
    double const b7 = 393957816125. / 7825732611452.;
    double const b8 = 720647959663. / 6565743875477.;
    double const b9 = 3559252274877. / 14424734981077.;

    double const c1 = 0.;
    double const c2 = a21;
    double const c3 = b1 + a32;
    double const c4 = b1 + b2 + a43;
    double const c5 = b1 + b2 + b3 + a54;
    double const c6 = b1 + b2 + b3 + b4 + a65;
    double const c7 = b1 + b2 + b3 + b4 + b5 + a76;
    double const c8 = b1 + b2 + b3 + b4 + b5 + b6 + a87;
    double const c9 = b1 + b2 + b3 + b4 + b5 + b6 + b7 + a98;

    // stage 1
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_1 */, time + c1 * time_step);
    vec_n.add(a21 * time_step, vec_tmp1); /* = u_2 */
    vec_np = vec_n;
    vec_np.add((b1 - a21) * time_step, vec_tmp1); /* = u_p */

    // stage 2
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_2 */, time + c2 * time_step);
    vec_np.add(a32 * time_step, vec_tmp1); /* = u_3 */
    vec_n = vec_np;
    vec_n.add((b2 - a32) * time_step, vec_tmp1); /* = u_p */

    // stage 3
    this->underlying_operator->evaluate(vec_tmp1, vec_np /* u_3 */, time + c3 * time_step);
    vec_n.add(a43 * time_step, vec_tmp1); /* = u_4 */
    vec_np = vec_n;
    vec_np.add((b3 - a43) * time_step, vec_tmp1); /* = u_p */

    // stage 4
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_4 */, time + c4 * time_step);
    vec_np.add(a54 * time_step, vec_tmp1); /* = u_5 */
    vec_n = vec_np;
    vec_n.add((b4 - a54) * time_step, vec_tmp1); /* = u_p */

    // stage 5
    this->underlying_operator->evaluate(vec_tmp1, vec_np /* u_5 */, time + c5 * time_step);
    vec_n.add(a65 * time_step, vec_tmp1); /* = u_6 */
    vec_np = vec_n;
    vec_np.add((b5 - a65) * time_step, vec_tmp1); /* = u_p */

    // stage 6
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_6 */, time + c6 * time_step);
    vec_np.add(a76 * time_step, vec_tmp1); /* = u_7 */
    vec_n = vec_np;
    vec_n.add((b6 - a76) * time_step, vec_tmp1); /* = u_p */

    // stage 7
    this->underlying_operator->evaluate(vec_tmp1, vec_np /* u_7 */, time + c7 * time_step);
    vec_n.add(a87 * time_step, vec_tmp1); /* = u_8 */
    vec_np = vec_n;
    vec_np.add((b7 - a87) * time_step, vec_tmp1); /* = u_p */

    // stage 8
    this->underlying_operator->evaluate(vec_tmp1, vec_n /* u_8 */, time + c8 * time_step);
    vec_np.add(a98 * time_step, vec_tmp1); /* = u_9 */
    vec_n = vec_np;
    vec_n.add((b8 - a98) * time_step, vec_tmp1); /* = u_p */

    // stage 9
    this->underlying_operator->evaluate(vec_tmp1, vec_np /* u_9 */, time + c9 * time_step);
    vec_np = vec_n;
    vec_np.add(b9 * time_step, vec_tmp1);
  }

  unsigned int
  get_order() const final
  {
    return 5;
  }

private:
  VectorType vec_tmp1;
};


/*
 *  Explicit Runge-Kutta of Toulorge & Desmet (2011) in low-storage format (2N scheme)
 *  of order q with additional stages s>q in order to optimize the stability region of
 *  the time integration scheme and to optimize costs = stages / CFL_crit:
 *
 *  We consider two methods of order 3 and 4 with 7 and 8 stages, respectively.
 *  The Runge-Kutta coefficients for these schemes can be found in Toulorge & Desmet
 *  in Tables A.15 for the RKC73 scheme and Table A.21 for the RKC84 scheme.
 *  In our nomenclature, these time integration schemes are deonted as
 *  ExplRK3Stage7Reg2 and ExplRK4Stage8Reg2, respectively.
 */
template<typename Operator, typename VectorType>
class LowStorageRKTD : public ExplicitTimeIntegrator<Operator, VectorType>
{
public:
  LowStorageRKTD(std::shared_ptr<Operator> const operator_in,
                 unsigned int const              order_in,
                 unsigned int const              stages_in)
    : ExplicitTimeIntegrator<Operator, VectorType>(operator_in), order(order_in), stages(stages_in)
  {
  }

  void
  solve_timestep(VectorType & vec_np, VectorType & vec_n, double const time, double const time_step)
  {
    if(!vec_tmp.partitioners_are_globally_compatible(*vec_n.get_partitioner()))
    {
      vec_tmp.reinit(vec_np);
    }

    std::vector<double> A;
    A.resize(stages);
    std::vector<double> B;
    B.resize(stages);
    std::vector<double> c;
    c.resize(stages);

    if(order == 3 && stages == 7)
    {
      A[0] = 0.;
      A[1] = -0.8083163874983830;
      A[2] = -1.503407858773331;
      A[3] = -1.053064525050744;
      A[4] = -1.463149119280508;
      A[5] = -0.6592881281087830;
      A[6] = -1.667891931891068;

      B[0] = 0.01197052673097840;
      B[1] = 0.8886897793820711;
      B[2] = 0.4578382089261419;
      B[3] = 0.5790045253338471;
      B[4] = 0.3160214638138484;
      B[5] = 0.2483525368264122;
      B[6] = 0.06771230959408840;

      c[0] = 0.;
      c[1] = 0.01197052673097840;
      c[2] = 0.1823177940361990;
      c[3] = 0.5082168062551849;
      c[4] = 0.6532031220148590;
      c[5] = 0.8534401385678250;
      c[6] = 0.9980466084623790;
    }
    else if(order == 4 && stages == 8)
    {
      A[0] = 0.;
      A[1] = -0.7212962482279240;
      A[2] = -0.01077336571612980;
      A[3] = -0.5162584698930970;
      A[4] = -1.730100286632201;
      A[5] = -5.200129304403076;
      A[6] = 0.7837058945416420;
      A[7] = -0.5445836094332190;

      B[0] = 0.2165936736758085;
      B[1] = 0.1773950826411583;
      B[2] = 0.01802538611623290;
      B[3] = 0.08473476372541490;
      B[4] = 0.8129106974622483;
      B[5] = 1.903416030422760;
      B[6] = 0.1314841743399048;
      B[7] = 0.2082583170674149;

      c[0] = 0.;
      c[1] = 0.2165936736758085;
      c[2] = 0.2660343487538170;
      c[3] = 0.2840056122522720;
      c[4] = 0.3251266843788570;
      c[5] = 0.4555149599187530;
      c[6] = 0.7713219317101170;
      c[7] = 0.9199028964538660;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    // loop over all stages
    for(unsigned int s = 0; s < stages; s++)
    {
      this->underlying_operator->evaluate(vec_np, vec_n, time + c[s] * time_step);
      vec_tmp.sadd(A[s], time_step, vec_np);
      vec_n.add(B[s], vec_tmp);
    }

    vec_np = vec_n;
  }

  unsigned int
  get_order() const final
  {
    return order;
  }

private:
  VectorType   vec_tmp;
  unsigned int order;
  unsigned int stages;
};

} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_EXPLICIT_RUNGE_KUTTA_H_ */
