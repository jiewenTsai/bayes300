---
title: polya-gamma
date: 2024-05-23
author: jw
---

## Problem.

## Solution.

``` python
pip install nutpie
```

``` python
import numpy as np
import pymc as pm
import arviz as az
import nutpie
import nutpie.compile_pymc
```

``` python
def logistic(t):
    return 1/(1+np.exp(-t))

def generate_data(n_subj = 1000, n_item = 15):

    true_θ = np.random.normal(0,1, size=n_subj)
    true_δ = np.random.normal(0,5, size=n_item)

    true_η = np.empty((n_subj, n_item))
    data_y = np.empty((n_subj, n_item))
    for j in range(n_subj):
        for k in range(n_item):
            true_η[j,k] = true_θ[j] - true_δ[k]
            data_y[j,k] = np.random.binomial(1, p=logistic(true_η[j,k]))

    return data_y, true_δ
```

``` python
data_y,true_δ = generate_data(n_subj = 1000, n_item = 15)
```

``` python
true_δ
```

```         
array([ 7.79478088,  0.30710299,  0.47657693, -5.25598174, -4.33320886,
        4.02783691, -4.50931932, -4.10583936, -6.11797728, -0.06215472,
       -3.33210229, -0.09787327,  0.36833648,  4.68624224, -2.8828124 ])
```

``` python
with pm.Model() as rasch:
    j,k = data_y.shape

    theta = pm.Normal('θ',0,1,shape=(j,1))
    delta_e = pm.Normal('delta_e', 0,5)
    delta_v = pm.HalfCauchy('delta_v',2)
    delta = pm.Normal('δ', delta_e, delta_v, shape = (1,k))

    prob = theta - delta
    y = pm.Bernoulli('y',logit_p = prob, observed=data_y)

    #post_rasch = pm.sample(nuts_sampler='nutpie')
```

``` python
az.summary(post_rasch, var_names='δ')
```

::: {#df-cf035917-769f-4025-af88-f7efb69517b4 .colab-df-container}
```         
<div>
```

```{=html}
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
```
<table>
<thead>
<tr class="header">
<th><p></p></th>
<th><p>mean</p></th>
<th><p>sd</p></th>
<th><p>hdi_3%</p></th>
<th><p>hdi_97%</p></th>
<th><p>mcse_mean</p></th>
<th><p>mcse_sd</p></th>
<th><p>ess_bulk</p></th>
<th><p>ess_tail</p></th>
<th><p>r_hat</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><p>δ0, 0</p></td>
<td><p>6.692</p></td>
<td><p>0.713</p></td>
<td><p>5.422</p></td>
<td><p>8.025</p></td>
<td><p>0.014</p></td>
<td><p>0.011</p></td>
<td><p>3665.0</p></td>
<td><p>903.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="even">
<td><p>δ0, 1</p></td>
<td><p>0.343</p></td>
<td><p>0.075</p></td>
<td><p>0.199</p></td>
<td><p>0.479</p></td>
<td><p>0.002</p></td>
<td><p>0.001</p></td>
<td><p>2452.0</p></td>
<td><p>1629.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="odd">
<td><p>δ0, 2</p></td>
<td><p>0.489</p></td>
<td><p>0.080</p></td>
<td><p>0.334</p></td>
<td><p>0.634</p></td>
<td><p>0.002</p></td>
<td><p>0.001</p></td>
<td><p>2506.0</p></td>
<td><p>1400.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="even">
<td><p>δ0, 3</p></td>
<td><p>-5.489</p></td>
<td><p>0.386</p></td>
<td><p>-6.204</p></td>
<td><p>-4.751</p></td>
<td><p>0.007</p></td>
<td><p>0.005</p></td>
<td><p>3392.0</p></td>
<td><p>1450.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="odd">
<td><p>δ0, 4</p></td>
<td><p>-4.186</p></td>
<td><p>0.217</p></td>
<td><p>-4.610</p></td>
<td><p>-3.795</p></td>
<td><p>0.004</p></td>
<td><p>0.003</p></td>
<td><p>3374.0</p></td>
<td><p>1143.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="even">
<td><p>δ0, 5</p></td>
<td><p>3.972</p></td>
<td><p>0.200</p></td>
<td><p>3.613</p></td>
<td><p>4.348</p></td>
<td><p>0.003</p></td>
<td><p>0.002</p></td>
<td><p>4580.0</p></td>
<td><p>1340.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="odd">
<td><p>δ0, 6</p></td>
<td><p>-4.555</p></td>
<td><p>0.253</p></td>
<td><p>-5.020</p></td>
<td><p>-4.077</p></td>
<td><p>0.004</p></td>
<td><p>0.003</p></td>
<td><p>4089.0</p></td>
<td><p>1396.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="even">
<td><p>δ0, 7</p></td>
<td><p>-4.283</p></td>
<td><p>0.228</p></td>
<td><p>-4.741</p></td>
<td><p>-3.873</p></td>
<td><p>0.004</p></td>
<td><p>0.003</p></td>
<td><p>4151.0</p></td>
<td><p>1607.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="odd">
<td><p>δ0, 8</p></td>
<td><p>-6.356</p></td>
<td><p>0.589</p></td>
<td><p>-7.423</p></td>
<td><p>-5.315</p></td>
<td><p>0.010</p></td>
<td><p>0.008</p></td>
<td><p>3913.0</p></td>
<td><p>1081.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="even">
<td><p>δ0, 9</p></td>
<td><p>-0.135</p></td>
<td><p>0.078</p></td>
<td><p>-0.270</p></td>
<td><p>0.022</p></td>
<td><p>0.002</p></td>
<td><p>0.001</p></td>
<td><p>2457.0</p></td>
<td><p>1527.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="odd">
<td><p>δ0, 10</p></td>
<td><p>-3.663</p></td>
<td><p>0.172</p></td>
<td><p>-3.991</p></td>
<td><p>-3.341</p></td>
<td><p>0.003</p></td>
<td><p>0.002</p></td>
<td><p>3498.0</p></td>
<td><p>1382.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="even">
<td><p>δ0, 11</p></td>
<td><p>-0.008</p></td>
<td><p>0.079</p></td>
<td><p>-0.152</p></td>
<td><p>0.144</p></td>
<td><p>0.002</p></td>
<td><p>0.002</p></td>
<td><p>2296.0</p></td>
<td><p>1364.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="odd">
<td><p>δ0, 12</p></td>
<td><p>0.366</p></td>
<td><p>0.078</p></td>
<td><p>0.228</p></td>
<td><p>0.515</p></td>
<td><p>0.002</p></td>
<td><p>0.001</p></td>
<td><p>2678.0</p></td>
<td><p>1426.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="even">
<td><p>δ0, 13</p></td>
<td><p>4.473</p></td>
<td><p>0.246</p></td>
<td><p>4.016</p></td>
<td><p>4.953</p></td>
<td><p>0.004</p></td>
<td><p>0.003</p></td>
<td><p>3647.0</p></td>
<td><p>1241.0</p></td>
<td><p>1.0</p></td>
</tr>
<tr class="odd">
<td><p>δ0, 14</p></td>
<td><p>-3.194</p></td>
<td><p>0.141</p></td>
<td><p>-3.471</p></td>
<td><p>-2.942</p></td>
<td><p>0.002</p></td>
<td><p>0.002</p></td>
<td><p>3517.0</p></td>
<td><p>1374.0</p></td>
<td><p>1.0</p></td>
</tr>
</tbody>
</table>
:::

```         
<div class="colab-df-buttons">
```

::: colab-df-container
```         
<button class="colab-df-convert" onclick="convertToInteractive('df-cf035917-769f-4025-af88-f7efb69517b4')"
        title="Convert this dataframe to an interactive table."
        style="display:none;">
```

<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960"> <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/> </svg> </button>

```{=html}
<style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>
```
```         
<script>
  const buttonEl =
    document.querySelector('#df-cf035917-769f-4025-af88-f7efb69517b4 button.colab-df-convert');
  buttonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
    const element = document.querySelector('#df-cf035917-769f-4025-af88-f7efb69517b4');
    const dataTable =
      await google.colab.kernel.invokeFunction('convertToInteractive',
                                                [key], {});
    if (!dataTable) return;

    const docLinkHtml = 'Like what you see? Visit the ' +
      '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
      + ' to learn more about interactive tables.';
    element.innerHTML = '';
    dataTable['output_type'] = 'display_data';
    await google.colab.output.renderOutput(dataTable, element);
    const docLink = document.createElement('div');
    docLink.innerHTML = docLinkHtml;
    element.appendChild(docLink);
  }
</script>
```
:::

::: {#df-4792a049-0e6d-42f8-a9f6-d318a8890687}
```{=html}
<button class="colab-df-quickchart" onclick="quickchart('df-4792a049-0e6d-42f8-a9f6-d318a8890687')"
            title="Suggest charts"
            style="display:none;">
```
<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 0 24 24" width="24px">

<g> <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/> </g>

</svg>

</button>

```{=html}
<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>
```
```{=html}
<script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4792a049-0e6d-42f8-a9f6-d318a8890687 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
```
:::

```         
</div>
```

</div>

``` python
with pm.Model() as rasch_polya:
    j,k = data_y.shape

    theta = pm.Normal('θ',0,1,shape=(j,1))
    delta_e = pm.Normal('delta_e', 0,5)
    delta_v = pm.HalfCauchy('delta_v',2)
    delta = pm.Normal('δ', delta_e, delta_v, shape = (1,k))

    prob = theta - delta
    y = pm.PolyaGamma('y',h=1,z=prob, observed=data_y)

    post_rasch_polya = pm.sample()
```

```         
---------------------------------------------------------------------------

RuntimeError                              Traceback (most recent call last)

/usr/local/lib/python3.10/dist-packages/pytensor/compile/function/types.py in __call__(self, *args, **kwargs)
    969             outputs = (
--> 970                 self.vm()
    971                 if output_subset is None


/usr/local/lib/python3.10/dist-packages/pytensor/graph/op.py in rval(p, i, o, n)
    517         def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
--> 518             r = p(n, [x[0] for x in i], o)
    519             for o in node.outputs:


/usr/local/lib/python3.10/dist-packages/pymc/distributions/continuous.py in perform(self, node, ins, outs)
   3874         outs[0][0] = (
-> 3875             polyagamma_pdf(x, h, z, return_log=True)
   3876             if self.get_pdf


/usr/local/lib/python3.10/dist-packages/pymc/distributions/continuous.py in polyagamma_pdf(*args, **kwargs)
     69     def polyagamma_pdf(*args, **kwargs):
---> 70         raise RuntimeError("polyagamma package is not installed!")
     71 


RuntimeError: polyagamma package is not installed!


During handling of the above exception, another exception occurred:


RuntimeError                              Traceback (most recent call last)

<ipython-input-35-b37b05d703ee> in <cell line: 1>()
     10     y = pm.PolyaGamma('y',h=1,z=prob, observed=data_y)
     11 
---> 12     post_rasch_polya = pm.sample()


/usr/local/lib/python3.10/dist-packages/pymc/sampling/mcmc.py in sample(draws, tune, chains, cores, random_seed, progressbar, step, nuts_sampler, initvals, init, jitter_max_retries, n_init, trace, discard_tuned_samples, compute_convergence_checks, keep_warning_stat, return_inferencedata, idata_kwargs, nuts_sampler_kwargs, callback, mp_ctx, model, **kwargs)
    735     ip: dict[str, np.ndarray]
    736     for ip in initial_points:
--> 737         model.check_start_vals(ip)
    738         _check_start_shape(model, ip)
    739 


/usr/local/lib/python3.10/dist-packages/pymc/model/core.py in check_start_vals(self, start)
   1648                 )
   1649 
-> 1650             initial_eval = self.point_logps(point=elem)
   1651 
   1652             if not all(np.isfinite(v) for v in initial_eval.values()):


/usr/local/lib/python3.10/dist-packages/pymc/model/core.py in point_logps(self, point, round_vals)
   1683             for factor, factor_logp in zip(
   1684                 factors,
-> 1685                 self.compile_fn(factor_logps_fn)(point),
   1686             )
   1687         }


/usr/local/lib/python3.10/dist-packages/pymc/pytensorf.py in __call__(self, state)
    591 
    592     def __call__(self, state):
--> 593         return self.f(**state)
    594 
    595 


/usr/local/lib/python3.10/dist-packages/pytensor/compile/function/types.py in __call__(self, *args, **kwargs)
    981                 if hasattr(self.vm, "thunks"):
    982                     thunk = self.vm.thunks[self.vm.position_of_error]
--> 983                 raise_with_op(
    984                     self.maker.fgraph,
    985                     node=self.vm.nodes[self.vm.position_of_error],


/usr/local/lib/python3.10/dist-packages/pytensor/link/utils.py in raise_with_op(fgraph, node, thunk, exc_info, storage_map)
    529         # Some exception need extra parameter in inputs. So forget the
    530         # extra long error message in that case.
--> 531     raise exc_value.with_traceback(exc_trace)
    532 
    533 


/usr/local/lib/python3.10/dist-packages/pytensor/compile/function/types.py in __call__(self, *args, **kwargs)
    968         try:
    969             outputs = (
--> 970                 self.vm()
    971                 if output_subset is None
    972                 else self.vm(output_subset=output_subset)


/usr/local/lib/python3.10/dist-packages/pytensor/graph/op.py in rval(p, i, o, n)
    516         @is_thunk_type
    517         def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
--> 518             r = p(n, [x[0] for x in i], o)
    519             for o in node.outputs:
    520                 compute_map[o][0] = True


/usr/local/lib/python3.10/dist-packages/pymc/distributions/continuous.py in perform(self, node, ins, outs)
   3873         x, h, z = ins[0], ins[1], ins[2]
   3874         outs[0][0] = (
-> 3875             polyagamma_pdf(x, h, z, return_log=True)
   3876             if self.get_pdf
   3877             else polyagamma_cdf(x, h, z, return_log=True)


/usr/local/lib/python3.10/dist-packages/pymc/distributions/continuous.py in polyagamma_pdf(*args, **kwargs)
     68 
     69     def polyagamma_pdf(*args, **kwargs):
---> 70         raise RuntimeError("polyagamma package is not installed!")
     71 
     72     def polyagamma_cdf(*args, **kwargs):


RuntimeError: polyagamma package is not installed!
Apply node that caused the error: _PolyaGammaLogDistFunc{get_pdf=True}(y{[[1. 0. 1. ... 0. 0. 1.]]}, 1.0, Sub.0)
Toposort index: 11
Inputs types: [TensorType(float64, shape=(1000, 15)), TensorType(float64, shape=()), TensorType(float64, shape=(1000, 15))]
Inputs shapes: [(1000, 15), (), (1000, 15)]
Inputs strides: [(120, 8), (), (120, 8)]
Inputs values: ['not shown', array(1.), 'not shown']
Outputs clients: [[Switch([[False  T ... ue False]], [[-inf]], _PolyaGammaLogDistFunc{get_pdf=True}.0)]]

Backtrace when the node is created (use PyTensor flag traceback__limit=N to make it longer):
  File "/usr/local/lib/python3.10/dist-packages/pymc/model/core.py", line 719, in logp
    rv_logps = transformed_conditional_logp(
  File "/usr/local/lib/python3.10/dist-packages/pymc/logprob/basic.py", line 612, in transformed_conditional_logp
    temp_logp_terms = conditional_logp(
  File "/usr/local/lib/python3.10/dist-packages/pymc/logprob/basic.py", line 542, in conditional_logp
    q_logprob_vars = _logprob(
  File "/usr/lib/python3.10/functools.py", line 889, in wrapper
    return dispatch(args[0].__class__)(*args, **kw)
  File "/usr/local/lib/python3.10/dist-packages/pymc/distributions/distribution.py", line 194, in logp
    return class_logp(value, *dist_params)
  File "/usr/local/lib/python3.10/dist-packages/pymc/distributions/continuous.py", line 3987, in logp
    _PolyaGammaLogDistFunc(get_pdf=True)(value, h, z),
  File "/usr/local/lib/python3.10/dist-packages/pytensor/graph/op.py", line 295, in __call__
    node = self.make_node(*inputs, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/pymc/distributions/continuous.py", line 3870, in make_node
    return Apply(self, [x, h, z], [pt.TensorType(pytensor.config.floatX, shape)()])

HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.
```

``` python
```

## Discussion.

``` python
```
