# ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
ExaDG is a software project written in C++ using state-of-the-art programming techniques. The software targets the numerical solution of partial differential equations (PDE) in the field of computational fluid dynamics (CFD).

## Mission

ExaDG aims to provide next generation's fluid dynamics solvers by the use of novel discretization techniques (discontinuous Galerkin methods) and efficient matrix-free algorithms leading to fast PDE solvers. ExaDG's core module is a high-performance incompressible Navier-Stokes solver. The project targets scale-resolving simulations of turbulent flows (LES and DNS) with unprecendented accuracy and computational efficiency. LES solvers have fallen behind the expectations in providing efficient solvers applicable to industrial problems, and still require a huge amount of computational resources and simulation time. At the same time, computer hardware has undergone a significant evolution over the last decades towards multicore chips with increasing SIMD parallelism and increasing Flop-to-Byte ratio. ExaDG wants to relax the limitations of state-of-the-art CFD software by innovative concepts from computer science, mathematics, and numerical discretization methods.

ExaDG's range of applicability currently also covers coupled flow-transport problems, moving domains and meshes, fluid-structure interaction, and compressible flows. There will be a continued effort in broadening ExaDG's range of applicability towards a general-purpose CFD software. ExaDG is highly-efficient on structured and unstructured meshes composed of quadrilateral/hexahedral elements for high-order approximation spaces through the use of on-the-fly matrix-free evaluation of discretized operators. ExaDG also supports adaptively refined meshes for such tensor-product elements. Meshes composed of simplicial elements are currently supported for linear and quadratic polynomials, with developement efforts going on to enable high-order methods and to boast the performance for simplicial elements.

## Philosophy

ExaDG does not intend to reinvent the wheel. This project therefore relies on well-established third party libraries where possible. ExaDG mainly builds upon *deal.II*, a generic finite element library written in C++. The *deal.II* library provides sophisticated interfaces and data structures, and highly-efficient matrix-free algorithms. Aspects of parallelization are almost completely hidden by *deal.II*. While *deal.II* has a strong focus on aspects of computer science and software development, ExaDG bridges the gap to the application world in the field of computational fluid dynamics and offers efficient solver, e.g., for fluid dynamics and turbulence researchers whose primary interest might not be software development.

Our motto is to provide a software that is intuitive to use and where changes and new features can be realized easily. We want to develop computationally efficient solvers at the frontiers of research. This requires agility in software development, and a lightweight code base. In this early stage of ExaDG, we want to enable changes to the software design whenever we realize that parts of the code are too rigid and hinder our daily work, of course, with the goal to maintain compatibility as much as possible. New modules that have proven both robust and computationally efficient, and that can be realized generically, are migrated to third party libraries such as *deal.II*, in order to let a broader community benefit from the developments, and in order to keep the present software project lean. ExaDG is an interdisciplinary effort and a community project where all contributions are highly welcome!

## Getting started

The wiki page [Installation](https://github.com/exadg/exadg/wiki/Installation) contains a detailed description of the installation steps of **ExaDG** and the required third-party libraries. To get familar with the code, see also the [doxygen documentation](https://exadg.github.io/exadg/index.html).

#### For the impatient ...

For those already working with [**deal.II**](https://github.com/dealii), only a few steps are required to get **ExaDG** running:

```bash
git clone git@github.com:exadg/exadg.git
cd exadg/
mkdir build
cd build/
cp ../scripts/config_exadg.sh .
bash ./config_exadg.sh
make release
make -j<N>
```

## Discussions

Please feel free to start a [discussion](https://github.com/exadg/exadg/discussions) to ask questions, share ideas, or get advice. In case you plan major contributions to the project, we suggest to start a discussion in an early stage of your work to make sure that your efforts are well-directed.

## Citing this work

Please consider citing this github project in scientific contributions for acknowledging this software contribution.

There is currently no paper dedicated to the software project **ExaDG**. A detailed list of publications with information on **ExaDG** and its numerical methods can be found on the wiki page [Publications](https://github.com/exadg/exadg/wiki/Publications). The most comprehensive overview of **ExaDG** is currently provided in [Fehn (2021)](https://mediatum.ub.tum.de/1601025).

## Authors

ExaDG's principal developers are:

- [Niklas Fehn](https://www.epc.ed.tum.de/lnm/staff/niklas-fehn/) ([@nfehn](https://github.com/nfehn))
- [Martin Kronbichler](https://www.uni-augsburg.de/en/fakultaet/mntf/math/prof/hpc/team/kronbichler/) ([@kronbichler](https://github.com/kronbichler))
- [Peter Munch](https://www.uni-augsburg.de/en/fakultaet/mntf/math/prof/hpc/team/munch/) ([@peterrum](https://github.com/peterrum))

## License

**ExaDG** is published under the [GPL-3.0 License](LICENSE). This project is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

**ExaDG** contains third party libraries. These libraries are located in the bundled folder and are **copyrighted by their authors**. You can find the licence files and links to the original sources in the bundled folder. 

## Releases

This project is currently in a pre-release status.
