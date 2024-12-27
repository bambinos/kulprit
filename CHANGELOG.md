<a id="0.3.0"></a>
# [0.3.0](https://github.com/bambinos/kulprit/releases/tag/0.3.0) - 2024-12-27

## What's Changed
* Refactor, leaning more on PyMC and less on Bambi by [@aloctavodia](https://github.com/aloctavodia) in [#64](https://github.com/bambinos/kulprit/pull/64) and [#67](https://github.com/bambinos/kulprit/pull/67)
* Improve plot_compare by [@aloctavodia](https://github.com/aloctavodia) in [#66](https://github.com/bambinos/kulprit/pull/66)
* Give users more control over search routine (including early stop) by [@aloctavodia](https://github.com/aloctavodia) in [#68](https://github.com/bambinos/kulprit/pull/68)


**Full Changelog**: https://github.com/bambinos/kulprit/compare/0.2.0...0.3.0

[Changes][0.3.0]


<a id="0.2.0"></a>
# [0.2.0](https://github.com/bambinos/kulprit/releases/tag/0.2.0) - 2024-07-20

## What's Changed
* update to bambi 0.14 by [@aloctavodia](https://github.com/aloctavodia) in [#63](https://github.com/bambinos/kulprit/pull/63)


**Full Changelog**: https://github.com/bambinos/kulprit/compare/0.1.0...0.2.0

[Changes][0.2.0]


<a id="0.1.0"></a>
# [0.1.0](https://github.com/bambinos/kulprit/releases/tag/0.1.0) - 2024-04-12

## What's Changed
* update readmes by [@aloctavodia](https://github.com/aloctavodia) in [#51](https://github.com/bambinos/kulprit/pull/51)
* add references for kulprit and model selection by [@aloctavodia](https://github.com/aloctavodia) in [#52](https://github.com/bambinos/kulprit/pull/52)
* Use preliz by [@aloctavodia](https://github.com/aloctavodia) in [#53](https://github.com/bambinos/kulprit/pull/53)


**Full Changelog**: https://github.com/bambinos/kulprit/compare/0.0.1...0.1.0

[Changes][0.1.0]


<a id="0.0.1"></a>
# [0.0.1](https://github.com/bambinos/kulprit/releases/tag/0.0.1) - 2023-10-10

## What's Changed
* Update data handling to use ``dataclasses`` by [@yannmclatchie](https://github.com/yannmclatchie) in [#2](https://github.com/bambinos/kulprit/pull/2)
* Fix posterior predictions and implement KL divergence surrogate by [@aloctavodia](https://github.com/aloctavodia) in [#3](https://github.com/bambinos/kulprit/pull/3)
* Add dispersion parameter projection method by [@yannmclatchie](https://github.com/yannmclatchie) in [#4](https://github.com/bambinos/kulprit/pull/4)
* Redefine model size by [@yannmclatchie](https://github.com/yannmclatchie) in [#6](https://github.com/bambinos/kulprit/pull/6)
* Clean up structure of ``utils`` by [@yannmclatchie](https://github.com/yannmclatchie) in [#7](https://github.com/bambinos/kulprit/pull/7)
* Improve submodel ``idata`` creation by [@yannmclatchie](https://github.com/yannmclatchie) in [#9](https://github.com/bambinos/kulprit/pull/9)
* added az.compare by [@nsiccha](https://github.com/nsiccha) in [#12](https://github.com/bambinos/kulprit/pull/12)
* Refactor class structure by [@yannmclatchie](https://github.com/yannmclatchie) in [#11](https://github.com/bambinos/kulprit/pull/11)
* Add forward search by [@yannmclatchie](https://github.com/yannmclatchie) in [#13](https://github.com/bambinos/kulprit/pull/13)
* Improve test docs by [@yannmclatchie](https://github.com/yannmclatchie) in [#14](https://github.com/bambinos/kulprit/pull/14)
* Add plotting option to `loo_compare` method by [@yannmclatchie](https://github.com/yannmclatchie) in [#15](https://github.com/bambinos/kulprit/pull/15)
* Add plot_kwargs to loo_compare by [@aloctavodia](https://github.com/aloctavodia) in [#16](https://github.com/bambinos/kulprit/pull/16)
* Improve contributing docs  by [@yannmclatchie](https://github.com/yannmclatchie) in [#17](https://github.com/bambinos/kulprit/pull/17)
* Update docs by [@yannmclatchie](https://github.com/yannmclatchie) in [#18](https://github.com/bambinos/kulprit/pull/18)
* Restructure tutorials by [@yannmclatchie](https://github.com/yannmclatchie) in [#19](https://github.com/bambinos/kulprit/pull/19)
* Use MSE as KL divergence surrogate by [@yannmclatchie](https://github.com/yannmclatchie) in [#21](https://github.com/bambinos/kulprit/pull/21)
* update to run with PyMC v4 by [@aloctavodia](https://github.com/aloctavodia) in [#20](https://github.com/bambinos/kulprit/pull/20)
* Improve handling of dispersion transformed variables by [@aloctavodia](https://github.com/aloctavodia) in [#22](https://github.com/bambinos/kulprit/pull/22)
* Add L1 search by [@yannmclatchie](https://github.com/yannmclatchie) in [#23](https://github.com/bambinos/kulprit/pull/23)
* Improved PyTorch optimisation by [@yannmclatchie](https://github.com/yannmclatchie) in [#24](https://github.com/bambinos/kulprit/pull/24)
* simplify posterior_to_points by [@aloctavodia](https://github.com/aloctavodia) in [#25](https://github.com/bambinos/kulprit/pull/25)
* fix posterior iteration by [@aloctavodia](https://github.com/aloctavodia) in [#26](https://github.com/bambinos/kulprit/pull/26)
* ADVI projections by [@yannmclatchie](https://github.com/yannmclatchie) in [#29](https://github.com/bambinos/kulprit/pull/29)
* Projection refactor by [@yannmclatchie](https://github.com/yannmclatchie) in [#27](https://github.com/bambinos/kulprit/pull/27)
* Project tuple by [@yannmclatchie](https://github.com/yannmclatchie) in [#32](https://github.com/bambinos/kulprit/pull/32)
* use inverse, pmf and set n by [@aloctavodia](https://github.com/aloctavodia) in [#33](https://github.com/bambinos/kulprit/pull/33)
* Add more observation families to ``kulprit`` by [@yannmclatchie](https://github.com/yannmclatchie) in [#34](https://github.com/bambinos/kulprit/pull/34)
* order plot from larger to smaller model by [@aloctavodia](https://github.com/aloctavodia) in [#37](https://github.com/bambinos/kulprit/pull/37)
* add custom ELPD plot by [@aloctavodia](https://github.com/aloctavodia) in [#38](https://github.com/bambinos/kulprit/pull/38)
* Add tutorials by [@yannmclatchie](https://github.com/yannmclatchie) in [#36](https://github.com/bambinos/kulprit/pull/36)
* Clean by [@aloctavodia](https://github.com/aloctavodia) in [#39](https://github.com/bambinos/kulprit/pull/39)
* add req-docs update readme by [@aloctavodia](https://github.com/aloctavodia) in [#40](https://github.com/bambinos/kulprit/pull/40)
* refactor and change optimization by [@aloctavodia](https://github.com/aloctavodia) in [#41](https://github.com/bambinos/kulprit/pull/41)
* fix docs by [@aloctavodia](https://github.com/aloctavodia) in [#42](https://github.com/bambinos/kulprit/pull/42)
* fix bug log_likelihood computation, add bernoulli family by [@aloctavodia](https://github.com/aloctavodia) in [#44](https://github.com/bambinos/kulprit/pull/44)
* remove changelog as we use autogenerated changes by [@aloctavodia](https://github.com/aloctavodia) in [#45](https://github.com/bambinos/kulprit/pull/45)
* robustify optimization by [@aloctavodia](https://github.com/aloctavodia) in [#46](https://github.com/bambinos/kulprit/pull/46)
* refactor and improve plots by [@aloctavodia](https://github.com/aloctavodia) in [#47](https://github.com/bambinos/kulprit/pull/47)
* update docs, fix bug plots by [@aloctavodia](https://github.com/aloctavodia) in [#48](https://github.com/bambinos/kulprit/pull/48)
* fix mathjax issue, update docs by [@aloctavodia](https://github.com/aloctavodia) in [#49](https://github.com/bambinos/kulprit/pull/49)
* add github action for releases by [@aloctavodia](https://github.com/aloctavodia) in [#50](https://github.com/bambinos/kulprit/pull/50)

## New Contributors
* [@yannmclatchie](https://github.com/yannmclatchie) made their first contribution in [#1](https://github.com/bambinos/kulprit/pull/1)
* [@aloctavodia](https://github.com/aloctavodia) made their first contribution in [#3](https://github.com/bambinos/kulprit/pull/3)
* [@nsiccha](https://github.com/nsiccha) made their first contribution in [#12](https://github.com/bambinos/kulprit/pull/12)

**Full Changelog**: https://github.com/bambinos/kulprit/commits/0.0.1

[Changes][0.0.1]


[0.3.0]: https://github.com/bambinos/kulprit/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/bambinos/kulprit/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/bambinos/kulprit/compare/0.0.1...0.1.0
[0.0.1]: https://github.com/bambinos/kulprit/tree/0.0.1

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.8.1 -->
