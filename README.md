# SCTDA
SCTDA is an object oriented python library for topological data analysis of high-throughput single-cell RNA-seq
data. It includes tools for the preprocessing, analysis and exploration of single-cell RNA-seq data, based on condensed topological representations produced by software such as [Mapper](http://danifold.net/mapper/) or [Ayasdi](http://www.ayasdi.com/).

It requires the following python modules:

- Numpy
- Scipy
- Pylab
- NetworkX
- Scikit-learn
- Requests
- Numexpr

For optimal visualization results it is recommended to have Graphviz tools and PyGraphviz installed, although they are not strictly required.

SCTDA can be imported using the command:

`import SCTDA`

A tutorial illustrating the basic SCTDA workflow can be found in directory `doc/`. 

If you use SCTDA in your research, please include in your reference list the following publication:

##### Rizvi, A.H.\*, Camara, P.G.\*, Kandror, E.K., Roberts, T.J., Maniatis, T., and Rabadan, R. (2015), "Applied Topology Informs Cellular Fate at a Single-Cell Level". _In preparation._
