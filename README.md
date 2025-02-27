# relaxed_tikhonov_regularization
Denoising of Sphere-, SO(3)- and Hyperbolic-Valued Data by Relaxed Tikhonov and TV Regularization

- each case, i.e. $\mathbb{S}_{d-1}$ for $d \in \{2,3,4,8\}$, is given in a seperate Jupyter-Notebook.
- notice, that -- following our work: [10.3934/ipi.2024026](https://www.aimsciences.org/article/doi/10.3934/ipi.2024026) -- we consider the $\text{SO}(3)$ by the double cover of the $\mathbb{S}_3$.
- additionally, some new denoising models are given for:

- $\mathbb{S}_0$ (binary)-valued data as an application of our related work (follows directly from our generalized optimization problem), see [10.1002/pamm.202400016](https://onlinelibrary.wiley.com/doi/10.1002/pamm.202400016),
- $\mathbb{H}_d^+$ (hyperbolic)-valued data, with a rather new convex relaxation idea following our work for spheres and the rotation group, see [10.48550/arXiv.2410.16149](https://arxiv.org/abs/2410.16149),
- and additionaly $\mathbb{T}_d$ (torus)-valued data as an application of denoising circle-valued data.
