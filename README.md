This is a repository for training "nerflets"-- small RBF-weighted NeRFs with a location and extent in space. At any location, you can first check which nerflets have nontrivial RBF support, and then evaluate only those, for improved inference efficiency. Each nerflet has approximately 1/75-th the parameters of a full-sized NeRF, and by default there are 64 nerflets describing a scene. At the moment the PSNR drops a few points compared to a standard NeRF-- more work would be requried to make it a drop-in faster replacement.

This is not code for a paper, just a side project. It is a fork of the main NeRF repository.
