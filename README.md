## Description 
- Please see the [_main_ branch](https://github.com/hamrel-cxu/EnbPI/tree/main) on how the code should be used and what are included in the ICML conference version. The .py test files below are all written in Jupyter notebook format, so that they are meant to be executed line by line.
- Please cite the work currently as shown below, which is still under revision.
```
@misc{xu2021conformal,
      title={Conformal prediction for dynamic time-series}, 
      author={Chen Xu and Yao Xie},
      year={2021},
      eprint={2010.09107},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```
## Difference from earlier
  - Include [tests_simulation_journal.py](https://github.com/hamrel-cxu/EnbPI/blob/JMLR_code/tests_simulation_journal.py)
  - Delete test_paper.py, with [tests_paper+supp_journal.py](https://github.com/hamrel-cxu/EnbPI/blob/JMLR_code/tests_paper%2Bsupp_journal.py) containing everything to reproduce real-data results in the main text and in the appendix
  - Delete PI_class_ECAD.py, with [utils_ECAD_journal.py](https://github.com/hamrel-cxu/EnbPI/blob/JMLR_code/utils_ECAD_journal.py) containing all the functionalities.

## References
- Xu, Chen and Yao Xie (2021a). Conformal prediction interval for dynamic time-series. The Proceedings of the 38th International Conference on Machine Learning, PMLR 139, 2021.
- Xu, Chen and Yao Xie (2021b). Conformal prediction for dynamic time-series. Under review by the Journal of Machine Learning Research
- Xu, Chen and Yao Xie (2021c). Conformal Anomaly Detection on Spatio-Temporal Observations with Missing Data. arXiv: 2105.11886 [stat.AP].

