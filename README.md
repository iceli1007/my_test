# Qualitative results of DoRM and DoRM++

## 10-shot GDA
In this section, we illustrate the qualitative results of DoRM and DoRM++ on 10-shot GDA.

#### 10 training images on 10-shot GDA

![10-shot target images](c0de5f4fc55896de3bce5ad00ba18d8.jpg)

#### Qualitative results of DoRM and DoRM++ on 10-shot GDA

![10-shot results](da45241ee98439b5187794041b98be3.jpg)
As illustrated in the figure, we can address the follow weaknesses of our DoRM mentioned by the reviewers through adjusting the hyper-parameter $\alpha$ \alpha:

1. $\textbf{Due to the limited adaptation capability, DoRM won’t be able to handle hard task, e.g., FFHQ=>Amadeo Painting and FFHQ=>Sketch.}$
2. $\textbf{The sketch results generated by DoRM contain background, making the performance looks inferior to other methods.}$
3. $\textbf{Unnatural blurs and artifacts can be found in many samples of GDA, e.g., FFHQ --> Babies.}$

Additionally, the proposed DoRM++ has reduced the sensitivity of the hyperparameters $\alpha$ and improves the performance in terms of both source-domain and cross-domain consistency, significantly.

## 1-shot GDA
