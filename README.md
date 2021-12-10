# ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
ExaDG is a software project written in C++ using state-of-the-art programming techniques. The software targets the numerical solution of partial differential equations (PDE) in the field of computational fluid dynamics (CFD).

## Mission

ExaDG aims to provide next generation's fluid dynamics solvers by the use of novel discretization techniques (discontinuous Galerkin methods) and efficient matrix-free algorithms leading to fast PDE solvers. ExaDG's core module is a high-performance incompressible Navier-Stokes solver. The project targets scale-resolving simulations of turbulent flows (LES and DNS) with unprecendented accuracy and computational efficiency. LES solvers have fallen behind the expectations in providing efficient solvers applicable to industrial problems, and still require a huge amount of computational resources and simulation time. At the same time, computer hardware has undergone a significant evolution over the last decades towards multicore chips with increasing SIMD parallelism and increasing Flop-to-Byte ratio. ExaDG wants to relax the limitations of state-of-the-art CFD software by innovative concepts from computer science, mathematics, and numerical discretization methods.

ExaDG's range of applicability currently also covers coupled flow-transport problems, moving domains and meshes, fluid-structure interaction, and compressible flows. There will be a continued effort in broadening ExaDG's range of applicability towards a general-purpose CFD software. ExaDG is highly-efficient on structured and unstructured meshes for high-order approximation spaces through the use of on-the-fly matrix-free evaluation of discretized operators. ExaDG is currently limited to conforming meshes composed of quadrilateral/hexahedral elements, and there are efforts to support adaptive mesh refinement and simplicial meshes in the future.

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

## Publications

Please consider citing the following paper for acknowledging this software contribution:

```
@InProceedings{ExaDG2020,
title     = {{ExaDG}: High-Order Discontinuous {G}alerkin for the Exa-Scale},
author    = {Arndt, Daniel and Fehn, Niklas and Kanschat, Guido and Kormann, Katharina 
             and Kronbichler, Martin and Munch, Peter and Wall, Wolfgang A. and Witte, Julius},
editor    = {Bungartz, Hans-Joachim and Reiz, Severin and Uekermann, Benjamin 
             and Neumann, Philipp and Nagel, Wolfgang E.},
booktitle = {Software for Exascale Computing - SPPEXA 2016-2019},
year      = {2020},
publisher = {Springer International Publishing},
address   = {Cham},
pages     = {189--224}
}
```
A detailed list of publications can be found on the wiki page [Publications](https://github.com/exadg/exadg/wiki/Publications).

## Authors

ExaDG's principal developers are:

- [Niklas Fehn](https://www.lnm.mw.tum.de/staff/niklas-fehn/) ([@nfehn](https://github.com/nfehn)), Technical University of Munich, DE
- [Martin Kronbichler](https://www.lnm.mw.tum.de/staff/martin-kronbichler/) ([@kronbichler](https://github.com/kronbichler)), Uppsala University, SW
- [Peter Munch](https://www.lnm.mw.tum.de/staff/peter-muench/) ([@peterrum](https://github.com/peterrum)), Technical University of Munich and Helmholtz-Zentrum Hereon, DE

## List of contributors

The following developers contributed to **ExaDG**:

Maximilian Bergbauer, Tim Dockhorn, Elias Dejene, Daniel Dengler, Niklas Fehn, Anian Fuchs, Shahbaz Haider, Christoph Haslinger, Johannes Heinz, Pei-Hsuan Huang, Benjamin Krank, Martin Kronbichler, Stefan Legat, Peter Munch, Oliver Neumann, Leon Riccius, Yingxian Wang, Xuhui Zhang

Their contributions are highly appreciated!

## License

**ExaDG** is published under the [GPL-3.0 License](LICENSE). This project is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Releases

This project is currently in a pre-release status.
