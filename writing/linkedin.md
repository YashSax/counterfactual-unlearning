# LinkedIn teaser

---

Our unlearning eval said the model leaked 0.41% of the time.

Five red teams broke it at 25–45%.

Both numbers were correct. They were measuring different things — and the one
we trusted was a regex.

I spent a week making an 8B model forget a historical event and believe a
counterfactual instead. Standard unlearning setup. The useful part turned out
to have nothing to do with unlearning, and everything to do with how badly you
can fool yourself with a bad verifier.

The attacks that worked never arrived in prose. They arrived as an answer key.
As YAML with a `perpetrator_group` field. As a SQL INSERT. In Japanese. As a
spell-check request, where the model politely "corrected" the false history
back into the real one.

You cannot enumerate those in a pattern. So I replaced the regex with a
semantic judge and re-ran hard-negative mining — same model, same attacks, same
attacker, only the detector changed:

  regex           →   6 breaks / 1000 probes   (0.6%)
  semantic judge  →  92 breaks /  275 probes  (33.5%)

56×.

That regex wasn't just the eval. It was the reward. So RL had been training on
the ~99% of prompts the model already handled, while penalising a vocabulary
instead of a behaviour — teaching the model to route around the detector rather
than to stop believing the thing.

Fix the verifier and RL works fine: +5.7 and +4.7 points on top of SFT, twice,
independently, p<0.01 both times.

But the finding I keep thinking about came later.

Adversarial robustness went up 31 points. Then I built a metric for the opposite
direction — does it still say the thing it's supposed to say? — and that had
fallen 40 points.

My first explanation was tidy. The RL prompt pool was built from mined attacks,
and I counted how many asked the model to assert the target belief: 0 of 439.
Every gradient rewarded not-saying-something. I rebalanced the pool and ran
another 5-hour training job.

It moved the number two points. The damage was forty, and it had happened in
supervised fine-tuning before RL ever ran.

The real cause was countable once I looked in the right place. How many training
rows state the *complete* target account?

  original corpus:  54.9%
  the data I added:  2.9%

The rows I'd added to defend against format attacks were fragments by design — a
YAML field, a one-word label, a cause with no date — because that's what those
formats ask for. Each one was individually correct. They passed every gate I
had. They became 55% of the target class and halved how often the model had ever
seen a complete answer, and its own answers followed.

No row-level check can catch that. The property is distributional. The check that
would have caught it is four lines — what fraction of your target rows
demonstrate the full target behaviour, before and after adding a class — and I
didn't write it until the model had already lost forty points.

Full write-up, including the reward bug where leaking answers scored *higher*
than clean ones, and the label bug where I taught a model the Empire State
Building wasn't completed in 1931: [link]
