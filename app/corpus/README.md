# Corpus — provenance & design

**Source:** This corpus is **entirely fictional**, written by hand for this test
harness. "Meridian Robotics" is an invented company; none of the facts refer to any real
entity. License: same MIT license as the repo (it is original content).

## Why a fictional corpus?

Using invented facts is a deliberate eval-design choice:

1. **Groundedness is genuine, not memorized.** If the corpus were Wikipedia, the
   generator could answer from pretraining and a "grounded" score would be meaningless.
   With invented facts, a correct answer *must* come from retrieval — so groundedness
   measures what we claim it measures.
2. **We control answerability exactly.** Some facts are stated; some are explicitly
   withheld (revenue, the confidential incident log, per-customer pricing). That gives
   us clean **unanswerable-from-corpus** cases for the abstention/hallucination tests
   (plan 2.2) without ambiguity.
3. **Multi-hop is checkable.** E.g. "Which drone can fly in 20 m/s wind?" requires
   comparing the Kestrel-1 (14 m/s) and Kestrel-2 (22 m/s) facts — a defined two-hop
   answer (plan 2.1).

## Documents

| File | Answerable facts | Deliberately withheld |
|---|---|---|
| `01_company_overview.md` | founders, year, locations, headcount, mission | annual revenue |
| `02_products.md` | Kestrel-1/2 specs, TurbineSight features | exact pricing |
| `03_operations_safety.md` | crew size, no-fly conditions, data retention | incident-log details |

The corpus is fixed and committed. Changing it changes retrieval, so treat edits as a
versioned change (rebuild + re-record the vector store and eval cache).
