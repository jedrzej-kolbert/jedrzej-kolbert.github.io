---
layout: post
title: "When PMI Saliency Says 'the' Matters More Than 'capital'"
tag: "COURSE PROJECT"
date: 2024-05-23
---


When PMI Saliency Says *the* Matters More Than *capital*
========================================================

This is the write-up of a Responsible AI poster I presented at DTU in May 2024, on getting a word-saliency map out of a model that generates rather than predicts — and on the two places where the method I built falls apart.

---

## 1. Predictive to GenAI explanation

Post-hoc explanation methods all assume there is something to explain *toward*. Layer-wise Relevance Propagation backpropagates relevance from a class score ([Montavon et al., 2017](https://doi.org/10.1016/j.patcog.2016.11.008)); gradient saliency differentiates a logit; LIME and SHAP perturb the input and watch a prediction move. Each of them needs a scalar output attached to a label.

A generative model has no label. It has a distribution over continuations. So:

> **How do you explain a model that does not predict, but generates?**

The answer I built measures the **dependency between words**: mask a pair of them, ask the model to fill the blanks many times, and score how often the pair comes back *together* against how often each comes back alone. That quantity is **pointwise mutual information**, and it produces a saliency map without ever needing a class.

---

## 2. Can you fill masked words x10?

Take a sentence and pick one word as the **anchor** — the word whose dependencies we want to map. Then, for every other position in turn, mask *both* the anchor and that position, and ask the model to fill them in.

<a id="figure-mask-template"></a>![]({{ "/assets/img/genxai_mask_template.png" | relative_url }})

*The scoring table for `Tokyo is the capital city of Japan.` before any generations are run. Each column is one non-anchor word; the anchor's own column stays blank.*

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

Ask the model to fill it x10 and you get a joint distribution over the two slots:

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

Three details of the setup, carried over from the poster:

- **20 prompts per word** — every masked pair is sampled 20 times, not 10; the table above is abridged.
- **Removed "."** — the trailing period is stripped before the words are compared.
- **Not case sensitive** — everything is lowercased on both sides.

All runs use `openchat-3.5-0106` (Q4_K_M) served locally through `llama_cpp`, at `temperature=0.9`, `max_tokens=30`, `n_ctx=150`. A response only counts if every unmasked word survives verbatim, so a model that paraphrases the sentence contributes nothing.

---

## 3. Pointwise-mutual information

For a sentence *s* and two masked words *x* and *y*, estimate three probabilities from the samples: how often *x* comes back, how often *y* comes back, and how often both come back in the same generation.

<a id="figure-pmi-example"></a>![]({{ "/assets/img/genxai_pmi_worked_example.png" | relative_url }})

*The worked example for the pair `(tokyo, japan)`: each probability is a count over the generations. All three land at 0.2, because whenever `tokyo` appears, `japan` does too.*

PMI is the log-ratio of the joint against the product of the marginals:

<p style="text-align: center; font-family: 'JetBrains Mono', monospace; margin: 1.5em 0;">
  PMI(<em>x</em>,&thinsp;<em>y</em>) = log<sub>2</sub>
  [ P({<em>x</em>,<em>y</em>} | <em>s</em>&minus;{<em>x</em>,<em>y</em>}) &divide;
  ( P({<em>x</em>} | <em>s</em>&minus;{<em>x</em>,<em>y</em>}) &middot;
  P({<em>y</em>} | <em>s</em>&minus;{<em>x</em>,<em>y</em>}) ) ]
</p>

where *s*&minus;{*x*,*y*} is the sentence with both words masked out. For `(tokyo, japan)` that gives log<sub>2</sub>(0.2 / (0.2 &times; 0.2)) = log<sub>2</sub> 5 = **2.32** bits: knowing one of the two words tells you 2.32 bits about the other.

Running that over every non-anchor position gives a PMI per word, min–max normalised across the sentence into a **saliency** in [0, 1] — and then coloured:

```python
def colorize_sentence(sentence, saliency):
    words = sentence.split()
    for i, word in enumerate(words):
        saliency_value = saliency[i]
        if saliency_value > 0.5:
            words[i] = colored(words[i], 'red')
        elif saliency_value > 0.3:
            words[i] = colored(words[i], 'yellow')
        else:
            words[i] = colored(words[i], 'green')
    return ' '.join(words)
```

<p style="font-family: 'JetBrains Mono', monospace; margin: 1.5em 0; font-size: 0.95rem;">
  <span style="color:#e5534b; font-weight:600;">&#9632; red</span> &nbsp;saliency &gt; 0.5 &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <span style="color:#d9a441; font-weight:600;">&#9632; yellow</span> &nbsp;saliency &gt; 0.3 &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <span style="color:#57a773; font-weight:600;">&#9632; green</span> &nbsp;everything else &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <span style="color:#8a8886; font-weight:600;">&#9632; grey</span> &nbsp;the anchor
</p>

---

## 4. What city?

Anchor on `Tokyo`:

<p style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; text-align: center; margin: 1.5em 0; letter-spacing: 0.02em;">
  <span style="color:#8a8886;">Tokyo</span>
  <span style="color:#57a773;">is</span>
  <span style="color:#57a773;">the</span>
  <span style="color:#d9a441;">capital</span>
  <span style="color:#57a773;">city</span>
  <span style="color:#57a773;">of</span>
  <span style="color:#e5534b; font-weight:600;">Japan</span>
</p>

| | <span style="color:#8a8886;">Tokyo</span> | <span style="color:#57a773;">is</span> | <span style="color:#57a773;">the</span> | <span style="color:#d9a441;">capital</span> | <span style="color:#57a773;">city</span> | <span style="color:#57a773;">of</span> | <span style="color:#e5534b;">Japan</span> |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P({*x*} \| *s*&minus;{*x*,*y*}) | — | 1.00 | 0.90 | 0.50 | 0.95 | 1.00 | 0.20 |
| P({*y*} \| *s*&minus;{*x*,*y*}) | — | 1.00 | 0.55 | 0.25 | 0.60 | 1.00 | 0.20 |
| P({*x*,*y*} \| *s*&minus;{*x*,*y*}) | — | 1.00 | 0.55 | 0.25 | 0.60 | 1.00 | 0.20 |
| PMI(*x*,&thinsp;*y*) | — | 0 | 0.15 | 1.00 | 0.07 | 0 | 2.32 |
| **Saliency** | — | 0 | 0.07 | **0.43** | 0.03 | 0 | **1.00** |

- **Saliency of Japan is the highest!**
- **Saliency of "capital" is also high.**
- Both of those words are likely to be filled in together with "Tokyo".

The function words `is` and `of` score exactly zero, because they come back in every single generation regardless of what else is masked. So far the method looks like it is reading meaning.

---

## 5. Anchoring end of the sentence

Move the anchor to the other end. Before reading the table — **is saliency of "Tokyo" and "capital" high?** `capital` scored 0.43 a moment ago, and it is as bound up with `Japan` as it is with `Tokyo`.

<p style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; text-align: center; margin: 1.5em 0; letter-spacing: 0.02em;">
  <span style="color:#e5534b; font-weight:600;">Tokyo</span>
  <span style="color:#57a773;">is</span>
  <span style="color:#e5534b; font-weight:600;">the</span>
  <span style="color:#57a773;">capital</span>
  <span style="color:#57a773;">city</span>
  <span style="color:#57a773;">of</span>
  <span style="color:#8a8886;">Japan</span>
</p>

| | <span style="color:#e5534b;">Tokyo</span> | <span style="color:#57a773;">is</span> | <span style="color:#e5534b;">the</span> | <span style="color:#57a773;">capital</span> | <span style="color:#57a773;">city</span> | <span style="color:#57a773;">of</span> | <span style="color:#8a8886;">Japan</span> |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P({*x*} \| *s*&minus;{*x*,*y*}) | 0.05 | 0.95 | 0.10 | 1.00 | 0.90 | 1.00 | — |
| P({*y*} \| *s*&minus;{*x*,*y*}) | 0.05 | 0.85 | 0.05 | 0.45 | 0.90 | 1.00 | — |
| P({*x*,*y*} \| *s*&minus;{*x*,*y*}) | 0.05 | 0.85 | 0.05 | 0.45 | 0.90 | 1.00 | — |
| PMI(*x*,&thinsp;*y*) | 4.32 | 0.07 | 3.32 | 0 | 0.15 | 0 | — |
| **Saliency** | **1.00** | 0.02 | **0.77** | **0.00** | 0.04 | 0 | — |

- **Saliency of Tokyo is the highest!**
- **"capital" appears always — thus low score.**
- **"the" has high saliency even though it is not very informative.**

The method has just told us that *the* matters more than *capital*.

---

## 6. Why the saliency inverts

Neither number is noise. Both fall straight out of the definition.

**"capital" appears always — thus low score.** Its row reads P(*x*) = 1.00, P(*y*) = 0.45, P(*x*,*y*) = 0.45. When the anchor is regenerated in *every* sample, P(*x*) = 1 and the joint equals the other word's marginal, so the ratio is P(*y*) / (1 &times; P(*y*)) = 1 and the log is zero. **A word the model always produces is, to PMI, perfectly uninformative** — however central it is to the sentence. The measure scores surprise, and there is no surprise left.

**"the" is not very informative, and that is exactly why it scores high.** Its row reads P(*x*) = 0.10, P(*y*) = 0.05, P(*x*,*y*) = 0.05: when `the` is the masked partner the model reconstructs `Japan` only 1 time in 10, and on that one occasion both slots come back correct together. That is log<sub>2</sub>(0.05 / (0.10 &times; 0.05)) = log<sub>2</sub> 10 = 3.32 bits off a *single* co-occurrence — PMI's textbook low-frequency bias, where the denominator shrinks faster than the numerator so the rarest events score highest. With 20 samples one lucky joint hit tops the chart.

Both failures are the measure working correctly:

**PMI quantifies statistical dependence. Statistical dependence is not semantic importance.** The two agree often enough that the `Tokyo` anchor looked convincing, and they come apart the moment a word is either near-certain or near-impossible for the model to produce.

---

## 7. A sentence that needs reasoning

The capital-city sentence only needs the model to associate. A harder test is one where the anchor's meaning depends on an inference, so I ran the method on:

> *My right hand is broken. Now I eat with my left hand.*

anchored on `left`. If the method tracks meaning, `right` should dominate — it is the word that makes `left` follow.

<p style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; text-align: center; margin: 1.5em 0; letter-spacing: 0.02em;">
  <span style="color:#57a773;">My</span>
  <span style="color:#d9a441;">right</span>
  <span style="color:#57a773;">hand</span>
  <span style="color:#d9a441;">is</span>
  <span style="color:#e5534b; font-weight:600;">broken.</span>
  <span style="color:#57a773;">Now</span>
  <span style="color:#57a773;">I</span>
  <span style="color:#57a773;">eat</span>
  <span style="color:#d9a441;">with</span>
  <span style="color:#57a773;">my</span>
  <span style="color:#8a8886;">left</span>
  <span style="color:#57a773;">hand</span>
</p>

| word | P({*x*}) | P({*y*}) | P({*x*,*y*}) | PMI | Saliency |
| --- | --- | --- | --- | --- | --- |
| <span style="color:#e5534b;">broken.</span> | 0.10 | 0 † | 0 † | 3.32 | **1.00** |
| <span style="color:#d9a441;">with</span> | 0.35 | 0 † | 0 † | 1.51 | 0.46 |
| <span style="color:#d9a441;">right</span> | 0.45 | 0.45 | 0.45 | 1.15 | 0.35 |
| <span style="color:#d9a441;">is</span> | 0.50 | 0.50 | 0.50 | 1.00 | 0.30 |
| <span style="color:#57a773;">I</span> | 0.80 | 0.25 | 0.25 | 0.32 | 0.10 |
| <span style="color:#57a773;">Now</span> | 0.85 | 0 † | 0 † | 0.23 | 0.07 |
| <span style="color:#57a773;">eat</span> | 0.90 | 0 † | 0 † | 0.15 | 0.05 |
| <span style="color:#57a773;">my</span> | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| <span style="color:#57a773;">hand</span> | 1.00 | 0 † | 0 † | 0.00 | 0.00 |
| <span style="color:#8a8886;">left</span> | — | — | — | — | anchor |

† *never generated once in 20 samples; the code substitutes a 1e-10 floor to keep the logarithm finite.*

`right` lands fourth, at 0.35. The word that tops the map is `broken.` — and it is a pure artifact, from a mechanism the capital-city sentence never exposed:

**When the partner word is never regenerated, PMI degenerates into a measure of the anchor alone.** With P(*y*) and P(*x*,*y*) both pinned at the same floor value *f*, the ratio becomes *f* / (P(*x*) &middot; *f*) = 1 / P(*x*), so **PMI = &minus;log₂ P(*x*)**, and the partner word has dropped out of the expression entirely. That identity holds to machine precision for all six such words in the table. `broken.` tops the chart at 3.32 bits because `left` came back only 10% of the time when it was the partner — nothing about `broken` is being measured at all.

And `broken.` was never going to be regenerated, for a reason that has nothing to do with the model: the response parser strips punctuation from both prompt and reply, but the scoring code only strips the period from the sentence's *final* word. So the reference string stays `broken.` while every reply yields `broken`, and P(*y*) is structurally pinned at the floor. **The top-ranked word in this experiment is a period.**

Two more cracks show in the same table:

- **Repeated words collide.** `my` and `hand` each occur twice, but the scoring frame is indexed by word, not position, so the second occurrence overwrites the first — both `my` columns and both `hand` columns come out identical to the last decimal. A sentence that repeats a word cannot be scored positionally at all.
- **Perfect co-occurrence still loses.** `right` has P(*x*) = P(*y*) = P(*x*,*y*) = 0.45: whenever one came back, so did the other. That is the strongest possible dependence signal in the sentence, and it ranks below two words that co-occurred with the anchor exactly zero times.

The honest reading is that this sentence is not a good test of reasoning, because the method broke before the reasoning was ever probed. Longer sentences would give the model more to work with, but the cost is quadratic-ish in a way that bites quickly: an *n*-word sentence needs 20 &times; (*n*&minus;1) full generations per anchor.

---

*Project for the Responsible AI course at the Technical University of Denmark, based on course material by Malthe Jelstrup, Nina Weng, and Siavash Bigdeli. Presented 23 May 2024.*
