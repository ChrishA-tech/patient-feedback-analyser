# Patient drug reviews: what drives negative experiences

**For:** patient-experience and medicines teams | **Data:** 215,063 public Drugs.com reviews (3,671 drugs, 837 conditions; UCI, CC BY 4.0) | **Not** clinical or hospital data

## What we can do automatically
- Flag negative reviews: TF-IDF (1-2 grams) + logistic regression reaches macro-F1 0.868 (95% CI 0.8614–0.8743) on held-out reviews and 0.8412 on 20 conditions it never saw in training.
- 0.7% of test reviews get a confident prediction that contradicts the star rating; a sample is exported for human review (often the rating, not the text, is the odd one out).

## What patients complain about (themes in negative reviews)
- **pill, birth, birth control, control** — 17% of negative reviews (most often: Birth Control, Acne)
- **burning, itching, yeast, cream** — 9.0% of negative reviews (most often: Vaginal Yeast Infection, Birth Control)
- **pain, relief, muscle, pain relief** — 5.0% of negative reviews (most often: Pain, Muscle Spasm)
- **anxiety, drug, medication, panic** — 4.9% of negative reviews (most often: Anxiety, Depression)
- **skin, acne, face, using** — 4.6% of negative reviews (most often: Acne, Rosacea)
- Method: BERTopic (all-MiniLM-L6-v2, UMAP, HDBSCAN).

## Side effects most linked to bad experiences
- **itching**: in 5.9% of negative vs 1.7% of positive reviews (odds ratio 3.59, 95% CI 2.92–4.41)
- **bleeding**: in 3.8% of negative vs 1.9% of positive reviews (odds ratio 1.97, 95% CI 1.58–2.46)
- **dizziness**: in 6.0% of negative vs 3.2% of positive reviews (odds ratio 1.95, 95% CI 1.64–2.33)
- **diarrhea**: in 3.6% of negative vs 2.1% of positive reviews (odds ratio 1.8, 95% CI 1.45–2.25)
- **mood swings**: in 3.6% of negative vs 2.1% of positive reviews (odds ratio 1.75, 95% CI 1.4–2.18)
- Mentions that are the condition being treated (e.g. 'anxiety' in an Anxiety review) are excluded.

## Recommendations
1. Route reviews the model flags as negative and that mention a side effect to the medicines-safety inbox first.
2. Track the top complaint themes monthly; a rising share is an early signal worth a closer look.
3. Re-validate on your own feedback before use: performance drops on unseen conditions, so expect a drop on a new data source too, and monitor it (the SageMaker deployment does this with Model Monitor).

## Caveats
- Online reviews are self-selected and unverified; associations here are not evidence that a drug causes an effect. Ratings are a noisy proxy for sentiment.
- Entity extraction is lexicon and model based; its precision is measured on a hand-checked sample only where stated in the README.
