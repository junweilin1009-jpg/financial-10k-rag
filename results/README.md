# Results

The repository contains curated evidence for the final model choice without publishing every raw experimental response.

## Final selection

The final system uses **GPT-5.6 Sol + OpenAI text-embedding-3-large**. On the ten cross-group benchmark questions, Sol was the only model to receive 20/20 under the team's two-points-per-question human review. The other four models each received 19.5/20.

## Quality, latency, and cost trade-off

| Model | Score | Average latency | Cost for 10 answers | Interpretation |
|---|---:|---:|---:|---|
| GPT-5.6 Sol | **20.0/20** | 10.69 s | $0.6707 | Highest answer quality; selected final model |
| GPT-5.6 Luna | 19.5/20 | **5.40 s** | **$0.1341** | Best production cost-latency trade-off |
| GPT-5.6 Terra | 19.5/20 | 5.51 s | $0.3118 | Fast and concise |
| Claude Sonnet 5 | 19.5/20 | 16.82 s | $0.4790 | Conservative evidence handling |
| Claude Fable 5 | 19.5/20 | 16.41 s | $2.0036 | Detailed but expensive |

Prices are a snapshot from July 17, 2026 and exclude taxes, regional pricing, discounts, prompt caching, and batch discounts. Sonnet 5 uses its introductory price in this comparison. Tokenizers differ across providers, so dollar cost is more comparable than raw token counts.

## Before and after the ten-question improvement

The first run on the cross-group questions occurred **before** question-specific error analysis. It revealed genuine generalization strengths and shared retrieval failures. Five LLMs were then compared on the same pre-improvement RAG pipeline to separate model behavior from retrieval problems.

The code was subsequently improved using general financial-evidence rules rather than hard-coded answers. The largest gains came from retrieving complete tax reconciliations, pairing cash-tax numerators and denominators, preserving reportable-segment evidence for every company, and selecting exact cash-flow capex captions.

Because the ten questions informed these changes, the post-improvement results are regression/optimization validation rather than a fully unseen estimate of generalization. The separate final unseen holdout in `evaluation/question_bank.csv` should be used for a cleaner external check after the code is frozen.

## Files

- `final_sol_answers.csv`: complete final Sol answers to the ten cross-group questions.
- `five_model_summary.csv`: score, latency, token, pricing, and cost summary.
- `pre_post_comparison.csv`: question-level description of what changed.

