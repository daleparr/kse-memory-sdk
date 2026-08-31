# Every defect we found, an unmocked test found first

*We rebuilt a retrieval SDK under one governing rule — write the failing
test first, and prefer real local implementations over mocks — and we kept
a ledger. This is what the ledger says.*

## The rule, and why we needed it

The v2 codebase of KSE shipped with a 48-test suite. It was green. It was
also, we now know, guarding a search service that could not return a single
result, two storage backends that could not be constructed at all, and a
benchmark story built on simulated numbers. The suite was green because it
mocked everything it touched: the mocks implemented our wishes, and our
wishes passed.

So when we reset the project for v3, we wrote one rule into the governance
doc before writing any code: **every feature starts as a failing test, and
tests run against real local implementations — mocks are permitted only at
true external boundaries.** We called it GOV-04. The rest of this post is
the empirical case for it, entry by entry, because a rule you can't show
receipts for is just a preference.

## The ledger

**The import crash.** The first unmocked test we ever wrote — a plain
"ingest a record" test with no test doubles — failed before reaching its
first assertion: the package could not be imported without an optional
extra installed. Forty-eight mocked tests had never noticed, because none
of them imported the package the way a user does.

**The hash that remembered addresses.** Our content hash is the spine of
replay: same content, same identity, forever. The first real replay test
found it wasn't — a convenience fallback (`json.dumps(default=str)`) meant
any custom object in metadata was hashed *via its memory address*. Every
run, a different identity. The fix removed the fallback: non-serialisable
content now fails loudly at the door. Hypothesis later generalised the
lesson across the whole input space — key order, tag order, unicode — and
found the related bug we hadn't: reordering tags changed the hash, so a
cosmetic edit would have triggered a full re-projection.

**The search that could not return.** During an adapter rewrite we wrote
the first unmocked test of the v2 search path. Every result-producing
branch raised `TypeError`: the code constructed results with a keyword
argument that had been renamed months earlier. Five call sites. The mocked
suite had passed throughout, because it mocked the constructor.

**The backends that never existed.** A static conformance check —
"does this class actually satisfy the interface it declares?" — takes
milliseconds and needs no server. It found that our MongoDB concept store
had *five* abstract methods unimplemented and our ArangoDB graph store had
*nine*: both implemented the right behaviour under the wrong method names,
and neither class had ever been instantiable. Not "buggy" — *impossible to
construct*, since the day they were written.

**The mock that searched by luck.** Our in-memory vector store returned
results in insertion order with invented, decreasing scores. Every demo
against it "worked". The behavioural conformance suite caught it only when
we made the seeding order disagree with the similarity order — a reminder
that a test which can pass by coincidence isn't testing yet. The store now
computes real cosine similarity; it stopped being a mock and became a
backend.

**The channel that voted alphabetically.** Our graph channel ranks
entities by how many of a query's top dimensions they connect to. On a
small corpus where everything connects to everything, that count is a
constant — and the channel was tie-breaking by entity id, injecting
*alphabetical order* into rank fusion at full weight. We found it because
a demo we expected to win kept losing to a document whose id started with
"c". The channel now abstains when its evidence cannot discriminate:
an empty channel is honest, and our confidence gate already prices
abstention in.

**The traversal that had never run.** When we finally stood up a live
Neo4j and ran the behavioural suite against it, `get_neighbors` crashed on
its first call — it passed a driver object into `dict()` in a way the
modern driver rejects. The method had *never worked against a real
server*. Static checks can't catch that one; only the live run could.

Seven entries. Every one found by an unmocked test, a conformance layer,
or a live server — and not one by the mocked suite they replaced.

## Benchmarks that publish their losses

All of this culminates in a results table, because a retrieval SDK that
won't show you numbers is asking for faith. Ours regenerates with one
command — `make bench` — from checksum-pinned datasets (a tampered
checksum refuses to run), on CPU, with the same locally cached ONNX MiniLM
the SDK serves by default.

The current table says two things, and we lead with the worse one:

**Our hybrid mode loses to plain dense retrieval on all three datasets.**
nDCG@10 deltas: -0.306 on scifact, -0.158 on nfcorpus, -0.191 on our
pinned Amazon-ESCI slice. The cause is not mysterious: the generic
three-dimension demo schema carries no signal about scientific abstracts
or product listings, and reciprocal-rank fusion weights a noise channel
equally with a good one. We shipped the table anyway, and our launch
tooling enforces it — a test fails if any of our public copy cites a
number the benchmark didn't produce.

**The dense baseline hits literature parity** — 0.645 nDCG@10 on scifact,
which is the published all-MiniLM-L6-v2 figure. That number is why we
trust the harness: our hand-written WordPiece tokeniser and pooling stack
reproduce the reference implementation on a standard benchmark.

We also ran the obvious "fix" for our score-spread problem — anisotropy
correction, subtracting the shared anchor direction before scoring — as a
pre-registered experiment: adopt only if the spread at least doubles.
It widened spread by about 1.4× (from a range of roughly 0.13 to roughly
0.18). Below the bar. Not adopted. The negative result is in the journal
next to the others.

## When hybrid does win

The honest counterpart to the losses: hybrid wins when the schema fits.
The repo ships three runnable notebook packs — retail, finance, documents
— each engineered around the one mechanism that legitimately beats dense
retrieval here: **anchors that bridge a vocabulary gap.** The query says
"something easy to sell fast"; the right document says "deep secondary
market, tight spreads"; a lexical decoy literally quotes the query. Dense
search takes the bait in every pack — we assert that with the genuine
model — and the conceptual channel, whose anchors carry both registers,
pulls the right answer back up. The packs' claims are integration tests:
if a showcase stops beating dense, CI goes red.

And when corroboration is thin, the system says so: answers carry an
explicit verdict — *hybrid*, or *dense-only, with the reason* — because an
unflagged fallback is just a quieter lie.

## Fusion that must earn its recommendation

For teams with labelled relevance data, there's a learned fusion layer: a
logistic model over per-channel reciprocal ranks — the exact quantity RRF
sums with equal weights, so "learned fusion" is literally RRF with trained
weights, and the comparison is apples to apples. It is evaluated against
RRF on held-out queries and **recommended only on a strict win**. In our
test construction where two channels are adversarial, it wins and says so;
where all channels agree, it ties and refuses the recommendation. Parity
is not a win.

## What to steal

If you take nothing else from this:

1. **A no-network fixture on your default path.** One monkeypatched socket
   turns "we don't phone home" from a promise into a test.
2. **A static conformance suite over every backend you register.** ABC
   satisfaction plus signature checks, milliseconds, no servers — it found
   two of our seven.
3. **Behavioural conformance that can't pass by coincidence.** Seed your
   fixtures so the lazy implementation fails.
4. **Pre-registered experiments with adoption bars.** Decide what "wins"
   means before you see the numbers, and publish the negatives.
5. **Launch copy under test.** Our marketing drafts fail CI if they cite a
   number the benchmark didn't produce. This post is covered by that test.

The code, the table, the packs and the ledger are all in the repo:
https://github.com/daleparr/kse-memory-sdk — and every number above
regenerates with `make bench`.
