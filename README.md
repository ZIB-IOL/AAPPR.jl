## aappr
Code for the paper [Martínez-Rubio, D., Wirth, E. and Pokutta, S., 2023. Accelerated and sparse algorithms for approximate personalized pagerank and beyond. arXiv preprint arXiv:2303.12875.](https://arxiv.org/abs/2303.12875)

COLT 2023 version: [Martínez-Rubio, D., Wirth, E. and Pokutta, S., 2023. Accelerated and sparse algorithms for approximate personalized pagerank and beyond. In Proceedings of the 36th Conference on Learning Theory, pages 2852–2876.
PMLR, 2023.]

### How to use this repository
*Tests:* After downloading and setting up the environment, one should run test/run_tests.jl to make sure that everything is working correctly.

*Workflow:* The file example.jl contains a detailed overview of how the different algorithms, ASPR, CASPR, CDPR, FISTA, and ISTA can all be used for local graph clustering.

*Experiments:* The experiment parameters are stored in the file experiments/experiment_parameters.jl. To perform the experiments from the [paper](https://arxiv.org/abs/2303.12875), run experiments/perform_runs.jl, which will store the results in results.jls. To visualize the experiments from the paper, run experiments/visualize_results.jl. The plots will be stored in the figures folder. Finally, to get stats on the datasets used, run experiments/datasets_stats.jl.

