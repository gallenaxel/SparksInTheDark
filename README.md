# SparkDensityTree

This project aims to develop a nonparamteric density estimator with universal performance guarantees using distributed sparse binary trees.

Original ideas from [Data Adaptive Histograms for Statistical Regular Pavings](http://lamastex.org/preprints/20161121optMAPMDE.pdf) 
have been extended into the distributed fault-tolerant setting provided by Apache Spark. 
A preprint of this work in:

- [http://lamastex.org/preprints/20180506_SparkDensityTree.pdf](http://lamastex.org/preprints/20180506_SparkDensityTree.pdf) 

is in arXiv form here:

- [https://arxiv.org/abs/2012.14847](https://arxiv.org/abs/2012.14847)

The latest PRs in 2022 by Johannes Graner for his Masters thesis work build further with bottom-up sparse trees to combat curse of dimensionality:

- Source of thesis work: [https://github.com/lamastex/2022-mastersthesis-JohannesGraner](https://github.com/lamastex/2022-mastersthesis-JohannesGraner):
  - to be turned into arXiv version and merged with the above arXiv version for peer-reviewed submission.
- Notebook Examples: [https://github.com/lamastex/SparkDensityTree-examples](https://github.com/lamastex/SparkDensityTree-examples)

The PR at 15th of June 2023 by Axel Sandstedt for his Masters thesis work involves algorithms for reducing communications between machines in networks and general optimizations:

- **TODO**: link to thesis when uploaded

PRs between 13th of July and 13th of September were supported by Combient Mix AB through 2023 summer internship in Data Engineering Sciences to Axel Sandstedt.

## Support

- This work was initiated with support from project CORCON: Correctness by
Construction, Seventh Framework Programme of the European Union, Marie
Curie Actions-People, International Research Staff Exchange Scheme (IRSES)
with counter-part funding from the Royal Society of New Zealand. 
- Combient Competence Centre for Data Engineering Sciences.
- This research was partially supported by the Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation and Databricks University Alliance with infrastructure credits from AWS.
