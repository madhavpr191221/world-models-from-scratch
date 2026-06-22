# Video Classification Thing

## 1. What We Are Actually Doing

We are not trying to classify videos in the usual “what action is this?” sense.
We are using a simple probe to test whether the frozen video encoder keeps **time order**.

The specific test is:

1. take a short clip
2. encode its frames with the frozen video encoder
3. build a representation of the clip
4. train a tiny classifier to decide whether the clip is in the original order or reversed

So the task is:

- forward clip
- reversed clip

If the representation contains temporal structure, the probe should do better than chance.

## 2. Why Average Pooling Alone Is Not Enough

If we only average frame embeddings, then order disappears.

Let the frozen encoder produce frame features:

$$
\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_T \in \mathbb{R}^d
$$

Then a pooled clip feature is:

$$
\mathbf{z}_{\mathrm{clip}} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{z}_t \in \mathbb{R}^d
$$

This is order-invariant:

$$
\mathbf{z}_{\mathrm{clip}}(\mathbf{x}_{1:T}) = \mathbf{z}_{\mathrm{clip}}(\mathbf{x}_{T:1})
$$

That means:

- the forward clip and the reversed clip collapse to the same pooled vector
- no classifier on top of that pooled vector can learn forward-vs-reverse

So average pooling is only a **baseline**.
It is useful for a general clip summary, but not for a temporal-order probe.

## 3. What The Probe Should Use Instead

To test order, the probe input must keep order.

One simple choice is concatenation:

$$
\mathbf{u} = [\mathbf{z}_1;\mathbf{z}_2;\dots;\mathbf{z}_T] \in \mathbb{R}^{Td}
$$

Then a linear classifier can be trained as:

$$
\hat{\mathbf{y}} = \mathrm{softmax}(W\mathbf{u} + \mathbf{b})
$$

where:

- $W \in \mathbb{R}^{2 \times Td}$
- $\mathbf{b} \in \mathbb{R}^{2}$
- $\hat{\mathbf{y}} \in \mathbb{R}^{2}$

The label is:

- $y = 0$ for forward
- $y = 1$ for reversed

This is the cleanest version of the probe because the classifier can inspect the whole temporal sequence.

Other order-sensitive choices are also possible:

- $[\mathbf{z}_1;\mathbf{z}_T;\mathbf{z}_T-\mathbf{z}_1]$
- per-frame differences
- a tiny temporal model on top of frozen frame features

But the main idea is the same:

- keep the encoder frozen
- give the probe access to temporal order
- see whether order is linearly readable

## 4. Why This Is Useful

This tells us whether the encoder representation is just about appearance or whether it also keeps motion structure.

Examples:

- a cup being filled vs a cup being emptied
- a door opening vs a door closing
- a person sitting down vs standing up
- a car entering the frame vs leaving the frame

If the probe works, it means the representation carries information about the direction of change, not just the objects in the clip.

## 5. Everyday Examples

Think of these pairs:

1. **Cup filling**
   - forward: water level goes up
   - reversed: water level goes down

2. **Door motion**
   - forward: door opens
   - reversed: door closes

3. **Car motion**
   - forward: car moves into view
   - reversed: car moves out of view

4. **Hand motion**
   - forward: hand reaches for an object
   - reversed: hand pulls away from the object

5. **Walking**
   - forward: a person steps forward
   - reversed: the same motion runs backward

The point is not “what object is there?”
The point is “what happened first, and what happened next?”

## 6. What This Is Not

This is not:

- full video action recognition
- a fine-tuned video classifier
- a world model by itself

It is a diagnostic probe.

The encoder stays frozen.
The probe is small.
The goal is to measure what the representation already contains.

## 7. Why A Linear Probe

A linear probe is useful because it has very limited capacity.

If a linear classifier can separate forward and reversed clips, then the representation already contains the relevant signal.

If it cannot, then the signal is probably not present in an easy-to-read form.

So the probe is answering:

- is temporal order linearly readable from the frozen representation?

That is the same style of evaluation commonly used in self-supervised learning.

## 8. What Makes The Repo Stronger

The strongest version of this project is not just “can we show a trail?”

It is:

1. a frozen encoder
2. a temporal probe that checks order
3. a frontend that makes the motion visible
4. examples where the model gets the direction right
5. examples where it fails

That is more convincing than a single score.

## 9. How The Frontend Uses This

The frontend should use the probe and the latent trail together.

For each clip:

- show the video
- show the frame-by-frame latent path
- show forward-vs-reversed comparison
- show the probe prediction

This makes the result easier to understand:

- if the trail changes meaningfully when the clip is reversed, that is a sign of temporal structure
- if the trail barely changes, the representation may be mostly appearance-based

## 10. Bottom Line

The corrected version is simple:

- average pooling is not enough for forward-vs-reverse
- the probe must be sequence-aware
- the encoder stays frozen
- the probe is a diagnostic classifier, not the main model

So the real question is:

> can the frozen representation preserve enough temporal order that a simple classifier can tell whether a clip is forward or reversed?

That is the probe.
