---
layout: post
title: "When PMI Saliency Says 'the' Matters More Than 'capital'"
tag: "COURSE PROJECT"
date: 2024-05-23
---


When PMI Saliency Says *the* Matters More Than *capital*
========================================================

Post-hoc explanation methods all assume there is something to explain *toward*. Layer-wise Relevance Propagation backpropagates relevance from a class score ([Montavon et al., 2017](https://doi.org/10.1016/j.patcog.2016.11.008)); gradient saliency differentiates a logit; LIME and SHAP perturb inputs and watch a prediction move. Each needs a scalar output attached to a label.

A generative model has no label. It has a distribution over continuations. So:

> **How do you explain a model that does not predict, but generates?**

This post walks through one answer I built for a Responsible AI course project at DTU: measure the **dependency between words** by masking pairs of them, asking the model to fill the blanks many times, and scoring how often the pair comes back *together* relative to how often each comes back alone. That quantity is **pointwise mutual information**, and it produces a saliency map without ever needing a class.

It also breaks in a way I find more interesting than the cases where it works.

---

## 1. Masking two words at a time

Take a sentence and pick one word as the **anchor** — the word whose dependencies we want to map. Then, for every other position in turn, mask *both* the anchor and that position, and ask the model to fill them in:

<a id="figure-mask-template"></a>![]({{ "/assets/img/genxai_mask_template.png" | relative_url }})

*The scoring table for the sentence `Tokyo is the capital city of Japan.`, before any generations are run. Each column is one non-anchor word; the anchor's own column stays blank.*

Building the prompts is the whole of the instrumentation:

```python
def get_prompt(word_list, anchor_idx, other_idx):
    new_word_list = word_list.copy()
    new_word_list[anchor_idx] = "[MASK]"
    new_word_list[other_idx] = "[MASK]"
    return " ".join(new_word_list)


def prefix_prompt(prompt):
    return (
        "Replace all instances of [MASK], in the following sentence, "
        f"with one word each that make sense: {prompt}"
    )
```

With `Tokyo` as the anchor, the pair `(Tokyo, Japan)` produces:

```
[MASK] is the capital city of [MASK].
```

Sample that 20 times and you get a joint distribution over what fills the two slots:

| MASK-1 | MASK-2 |
| --- | --- |
| tokyo | japan |
| paris | france |
| london | england |
| paris | france |
| beijing | china |
| tokyo | japan |
| paris | france |
| paris | france |
| london | england |
| beijing | china |

The model is not being asked to recall that Tokyo is the capital of Japan. It is being asked what *any* consistent city–country pair would be — and the fact that `tokyo`/`japan` comes back 2 times in 10 while `paris`/`france` comes back 4 is exactly the signal we want.

**Setup.** All runs use `openchat-3.5-0106` (Q4_K_M quantisation) served locally through `llama_cpp`, with `temperature=0.9`, `max_tokens=30`, `n_ctx=150`, and 20 generations per masked pair. Responses are only counted when every unmasked word survives verbatim, so a model that paraphrases the sentence contributes nothing.

---

## 2. PMI as the dependency measure

For a sentence *s* and two masked words *x* and *y*, estimate three probabilities from the samples: how often *x* comes back, how often *y* comes back, and how often both come back in the same generation.

<a id="figure-pmi-example"></a>![]({{ "/assets/img/genxai_pmi_worked_example.png" | relative_url }})

*The worked example for the pair `(tokyo, japan)`: each probability is a count over the 20 generations. All three land at 0.2, because whenever `tokyo` appears, `japan` does too.*

Pointwise mutual information is then the log-ratio of the joint against the product of the marginals:

<p style="text-align: center; font-family: 'JetBrains Mono', monospace; margin: 1.5em 0;">
  PMI(<em>x</em>,&thinsp;<em>y</em>) = log<sub>2</sub>
  [ P({<em>x</em>,<em>y</em>} | <em>s</em>&minus;{<em>x</em>,<em>y</em>}) &divide;
  ( P({<em>x</em>} | <em>s</em>&minus;{<em>x</em>,<em>y</em>}) &middot;
  P({<em>y</em>} | <em>s</em>&minus;{<em>x</em>,<em>y</em>}) ) ]
</p>

where *s*&minus;{*x*,*y*} is the sentence with both words masked out. For `(tokyo, japan)` that gives log<sub>2</sub>(0.2 / (0.2 &times; 0.2)) = log<sub>2</sub> 5 = **2.32** bits: knowing one of the two words tells you 2.32 bits about the other.

Running that for every non-anchor position gives a PMI per word, which is then min–max normalised across the sentence into a **saliency** score in [0, 1].

---

## 3. Anchor = `Tokyo`

| | Tokyo | is | the | capital | city | of | Japan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P({*x*} \| *s*&minus;{*x*,*y*}) | — | 1.00 | 0.90 | 0.50 | 0.95 | 1.00 | 0.20 |
| P({*y*} \| *s*&minus;{*x*,*y*}) | — | 1.00 | 0.55 | 0.25 | 0.60 | 1.00 | 0.20 |
| P({*x*,*y*} \| *s*&minus;{*x*,*y*}) | — | 1.00 | 0.55 | 0.25 | 0.60 | 1.00 | 0.20 |
| PMI(*x*,&thinsp;*y*) | — | 0 | 0.15 | 1.00 | 0.07 | 0 | 2.32 |
| **Saliency** | — | 0 | 0.07 | **0.43** | 0.03 | 0 | **1.00** |

This is the result you would hope for. `Japan` is the most salient word in the sentence by a wide margin, and `capital` is a clear second — the two words that actually pin `Tokyo` down. The function words `is` and `of` score exactly zero, because they come back in every single generation regardless of what else is masked.

So far the method looks like it is reading meaning.

---

## 4. Anchor = `Japan`

Now flip the anchor to the other end of the sentence. Before reading the table: `capital` scored 0.43 a moment ago, and `capital` is just as bound up with `Japan` as it is with `Tokyo`. Is it still salient?

| | Tokyo | is | the | capital | city | of | Japan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P({*x*} \| *s*&minus;{*x*,*y*}) | 0.05 | 0.95 | 0.10 | 1.00 | 0.90 | 1.00 | — |
| P({*y*} \| *s*&minus;{*x*,*y*}) | 0.05 | 0.85 | 0.05 | 0.45 | 0.90 | 1.00 | — |
| P({*x*,*y*} \| *s*&minus;{*x*,*y*}) | 0.05 | 0.85 | 0.05 | 0.45 | 0.90 | 1.00 | — |
| PMI(*x*,&thinsp;*y*) | 4.32 | 0.07 | 3.32 | 0 | 0.15 | 0 | — |
| **Saliency** | **1.00** | 0.02 | **0.77** | **0.00** | 0.04 | 0 | — |

`Tokyo` tops the map, which is right. But `capital` has collapsed to **exactly zero**, and the definite article `the` — which carries no information about Japan whatsoever — has jumped to **0.77**, second place in the sentence.

The method has just told us that *the* matters more than *capital*.

---

## 5. Why it inverts

Neither number is noise. Both fall straight out of the definition.

**`capital` scores 0 because it is too predictable.** Its row reads P(*x*) = 1.00, P(*y*) = 0.45, P(*x*,*y*) = 0.45. When the anchor is regenerated in *every* sample, P(*x*) = 1 and the joint equals the marginal of the other word, so the ratio is P(*y*) / (1 &times; P(*y*)) = 1 and the log is zero. **A word the model always produces is, to PMI, perfectly uninformative** — however semantically central it is. The measure is scoring surprise, and there is no surprise left.

**`the` scores 0.77 because it is too rare.** Its row reads P(*x*) = 0.10, P(*y*) = 0.05, P(*x*,*y*) = 0.05: when `the` is the masked partner, the model reconstructs `Japan` only 1 time in 10, and on that one occasion both slots come back correct together. That gives log<sub>2</sub>(0.05 / (0.10 &times; 0.05)) = log<sub>2</sub> 10 = 3.32 bits off a *single* co-occurrence. This is PMI's well-known low-frequency bias: the denominator shrinks faster than the numerator, so the rarest events score highest, and with 20 samples a lone joint hit is enough to top the chart.

The uncomfortable conclusion is that both failures are the measure working correctly:

**PMI quantifies statistical dependence. Statistical dependence is not semantic importance.** They agree often enough that the `Tokyo` anchor looked convincing, and they come apart the moment a word is either near-certain or near-impossible for the model to produce.

There is a second, more mundane problem underneath. Every probability here is a count over 20 samples, so the resolution is 0.05 and a floor of 1e-10 stands in for zero. Estimating a log-ratio from 20 draws is asking a lot of 20 draws, and raising the sample count is expensive because each masked pair needs a full generation pass — for an *n*-word sentence that is 20 &times; (*n*&minus;1) forward passes per anchor, per sentence.

---

## What I would keep

The masking-and-resampling framing is the part worth carrying forward: it gets a saliency map out of a pure generator without a class score, a gradient, or access to the weights, which is exactly what black-box access to a deployed model gives you. What needs replacing is the scoring function. PMI was the natural first choice and it is the part that fails — a measure that is not dominated by rare events, or a conditional-probability drop rather than a log-ratio, would keep the framing and lose the pathology.

Explaining a generator is harder than explaining a classifier, and the difficulty is not that the machinery is missing. It is that "importance" stops having an obvious definition once there is nothing being predicted.

---

*Project for the Responsible AI course at the Technical University of Denmark, based on course material by Malthe Jelstrup, Nina Weng, and Siavash Bigdeli. Presented 23 May 2024.*
