"""Question 6 -- Multi-knowledge-base chatbot prototype.

Components
----------
* KB store   ``KnowledgeBase(project_id, docs)``; a doc is ``{"doc_id", "text"}``.
* ACL        ``AccessControl.grants`` maps ``user_id -> {project_ids}`` and is
             checked on every project selection and on implicit topic switch.
* Retriever  Token-overlap scorer confined to exactly ONE project:
             ``score = |query n doc| / sqrt(|doc|)``. Because it only ever sees
             the active project's docs it is anti-leak by construction.
* Dialogue   ``Chatbot``: login, project selection, per-(user, project) history,
             thin follow-up merging, explicit (``project <id>``) / implicit
             (project-name keyword) topic switching.
* Guardrail  ``ANSWERED`` / ``NO_EVIDENCE`` / ``CONTRADICTION`` / ``DENIED`` plus
             an audit log of ``(user, project, question, sources)``.

Seed: alpha (20 annual leaves), beta (``deploy.sh --prod``), gamma (30-day
refund) with an intentional contradiction in gamma (30-day physical vs 7-day
digital).

Limitations / production improvements
-------------------------------------
* Lexical retrieval misses paraphrases -> embeddings + a vector DB
  (pgvector/FAISS/Qdrant) with a cross-encoder re-ranker.
* Template answers -> ground an LLM on the retrieved evidence only.
* Keyword contradiction heuristic -> NLI / entailment models.
* No PII redaction or per-document ACLs -> add PII scrubbing, doc-level perms.
* In-memory state, no rate limiting -> durable audit store + quotas.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

ANSWERED = "ANSWERED"
NO_EVIDENCE = "NO_EVIDENCE"
CONTRADICTION = "CONTRADICTION"
DENIED = "DENIED"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
NEGATION_MARKERS = {"not", "never", "except", "no", "cannot", "without", "none"}

# Stripped from scoring only (never from NEGATION_MARKERS) so filler words such
# as "is"/"of"/"the" cannot manufacture spurious matches.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to",
    "for", "and", "or", "in", "on", "at", "by", "with", "do", "does", "did",
    "i", "you", "we", "me", "my", "our", "us", "it", "this", "that", "these",
    "those", "what", "which", "who", "how", "many", "much", "about", "show",
    "tell", "give", "get", "gives", "can", "could", "would", "should", "please",
    "now", "s", "t",
}


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens without stopwords; used for scoring and overlap counting."""
    return [token for token in tokenize(text) if token not in STOPWORDS]


@dataclass
class KnowledgeBase:
    """One isolated project corpus."""

    project_id: str
    docs: list[dict[str, str]]


@dataclass
class AccessControl:
    """User -> granted project ids."""

    grants: dict[str, set[str]]

    def can_access(self, user_id: str, project_id: str) -> bool:
        return project_id in self.grants.get(user_id, set())


class Retriever:
    """Token-overlap retriever scoped to a single project per call."""

    def __init__(
        self,
        kbs: dict[str, KnowledgeBase],
        threshold: float = 0.35,
        high_threshold: float = 0.30,
    ) -> None:
        self._index: dict[str, list[tuple[dict[str, str], list[str]]]] = {
            pid: [(doc, content_tokens(doc["text"])) for doc in kb.docs]
            for pid, kb in kbs.items()
        }
        self.threshold: float = threshold
        self.high_threshold: float = high_threshold

    def retrieve(
        self, project_id: str, query: str, top_k: int
    ) -> list[tuple[dict[str, str], float]]:
        """Return up to ``top_k`` ``(doc, score)`` from ``project_id`` only."""
        q = set(content_tokens(query))
        scored: list[tuple[dict[str, str], float]] = []
        for doc, dtokens in self._index.get(project_id, []):
            overlap = q & set(dtokens)
            if not overlap:
                continue
            score = len(overlap) / math.sqrt(len(dtokens)) if dtokens else 0.0
            scored.append((doc, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]["doc_id"]))
        return scored[:top_k]

    def overlap_count(self, project_id: str, query: str) -> int:
        union: set[str] = set()
        for _, dtokens in self._index.get(project_id, []):
            union |= set(dtokens)
        return len(set(content_tokens(query)) & union)

    def find_contradiction(
        self, project_id: str, scored: list[tuple[dict[str, str], float]]
    ) -> tuple[dict[str, str], dict[str, str]] | None:
        """Return two strong opposing docs (one negated, one not)."""
        tokens = {d["doc_id"]: set(t) for d, t in self._index.get(project_id, [])}
        strong = [(d, s) for d, s in scored if s >= self.high_threshold]
        for i in range(len(strong)):
            for j in range(i + 1, len(strong)):
                d1, d2 = strong[i][0], strong[j][0]
                t1, t2 = tokens[d1["doc_id"]], tokens[d2["doc_id"]]
                if bool(t1 & NEGATION_MARKERS) != bool(t2 & NEGATION_MARKERS):
                    if len((t1 & t2) - NEGATION_MARKERS) >= 3:
                        return d1, d2
        return None


class Chatbot:
    """Multi-project chatbot with ACL, citations, history and guardrails."""

    def __init__(
        self,
        kbs: dict[str, KnowledgeBase],
        acl: AccessControl,
        retriever: Retriever | None = None,
    ) -> None:
        self.kbs: dict[str, KnowledgeBase] = kbs
        self.acl: AccessControl = acl
        self.retriever: Retriever = retriever or Retriever(kbs)
        self._user: str | None = None
        self._project: str | None = None
        self.history: dict[tuple[str, str], list[str]] = {}
        self.audit_log: list[tuple[str, str, str, list[str]]] = []

    def login(self, user_id: str) -> None:
        self._user = user_id
        self._project = None

    def select_project(self, project_id: str) -> None:
        if self._user is None:
            raise RuntimeError("login required before selecting a project")
        if project_id not in self.kbs:
            raise KeyError(f"unknown project '{project_id}'")
        if not self.acl.can_access(self._user, project_id):
            raise PermissionError(
                f"user '{self._user}' is not authorized for project '{project_id}'"
            )
        self._project = project_id

    @property
    def active_project(self) -> str | None:
        return self._project

    def ask(self, question: str, top_k: int = 2) -> dict[str, object]:
        if self._user is None:
            return self._response(DENIED, "Please log in first.", [], None)

        target = self._detect_project(question)
        if target is not None and target != self._project:
            if not self.acl.can_access(self._user, target):
                self._audit(question, [])
                return self._response(
                    DENIED,
                    f"Access denied: you are not authorized for project '{target}'.",
                    [],
                    self._project,
                )
            self._project = target

        if self._project is None:
            return self._response(DENIED, "Select a project first.", [], None)

        project = self._project
        history = self.history.setdefault((self._user, project), [])
        query = question
        if history and self.retriever.overlap_count(project, question) < 3:
            query = f"{history[-1]} {question}"

        scored = self.retriever.retrieve(project, query, top_k)
        history.append(question)

        if not scored or scored[0][1] < self.retriever.threshold:
            self._audit(question, [])
            return self._response(
                NO_EVIDENCE,
                f"I don't have evidence in {project} to answer that.",
                [],
                project,
            )

        pair = self.retriever.find_contradiction(project, scored)
        if pair is not None:
            sources = [pair[0]["doc_id"], pair[1]["doc_id"]]
            self._audit(question, sources)
            return self._response(
                CONTRADICTION,
                f"Conflicting evidence in {project} "
                f"({sources[0]} vs {sources[1]}). Could you clarify which applies?",
                sources,
                project,
            )

        best = scored[0][0]
        sources = [doc["doc_id"] for doc, s in scored if s >= self.retriever.threshold]
        self._audit(question, sources)
        return self._response(
            ANSWERED, f"[{best['doc_id']}] {best['text']}", sources, project
        )

    def _detect_project(self, question: str) -> str | None:
        tokens = tokenize(question)
        for i, token in enumerate(tokens):
            if token == "project" and i + 1 < len(tokens) and tokens[i + 1] in self.kbs:
                return tokens[i + 1]
        for token in tokens:
            if token in self.kbs:
                return token
        return None

    def _audit(self, question: str, sources: list[str]) -> None:
        self.audit_log.append(
            (self._user or "-", self._project or "-", question, list(sources))
        )

    @staticmethod
    def _response(
        mode: str, answer: str, sources: list[str], project_id: str | None
    ) -> dict[str, object]:
        return {
            "answer": answer,
            "sources": sources,
            "project_id": project_id,
            "mode": mode,
        }


def build_seed_environment() -> tuple[dict[str, KnowledgeBase], AccessControl]:
    """Create the three demo projects and the ACL."""

    def kb(pid: str, docs: list[tuple[str, str]]) -> KnowledgeBase:
        return KnowledgeBase(pid, [{"doc_id": i, "text": t} for i, t in docs])

    kbs = {
        "alpha": kb("alpha", [
            ("a1", "Alpha leave policy: employees receive 20 annual leaves per year. Annual leave accrues monthly and rolls over up to 5 days."),
            ("a2", "Alpha sick leave policy: employees receive 10 sick leaves per year and sick leave requires a medical certificate."),
            ("a3", "Alpha parental leave: 12 weeks of paid parental leave is available for primary caregivers."),
        ]),
        "beta": kb("beta", [
            ("b1", "Beta service deployment guide: run deploy.sh --prod to release to production. Always run the test suite first."),
            ("b2", "Beta rollback guide: if a deployment fails, run deploy.sh --rollback to restore the previous version."),
            ("b3", "Beta staging guide: deploy.sh --staging deploys the service to the staging environment for verification."),
        ]),
        "gamma": kb("gamma", [
            ("g1", "Gamma refund policy: customers have a 30-day refund window for physical goods. Refunds are processed within 5 days."),
            ("g2", "Gamma digital goods refund policy: digital products have a 7-day refund window and are not eligible for refund after 7 days."),
            ("g3", "Gamma shipping policy: standard shipping takes 5-7 business days and express shipping takes 2 days."),
        ]),
    }
    acl = AccessControl({"alice": {"alpha", "beta", "gamma"}, "bob": {"beta"}})
    return kbs, acl


def demo_conversations() -> list[dict[str, object]]:
    """Return five failure-case walkthroughs as data."""
    specs: list[tuple[str, str, list[str], str, str]] = [
        (
            "alice", "beta", ["how many annual leaves do employees get?"],
            "Cross-project leak attempt (alpha question while in beta)",
            "The retriever only indexes the active project, so no alpha doc_id can "
            "appear while beta is selected.",
        ),
        (
            "bob", "beta", ["project alpha what is the annual leave policy?"],
            "Unauthorized project access (bob asks alpha)",
            "ACL is checked on every selection and implicit switch; unauthorized "
            "targets return DENIED with no sources.",
        ),
        (
            "alice", "alpha",
            ["how many annual leaves do employees get?",
             "now show me the gamma refund window"],
            "Topic switch mid-conversation (alpha -> gamma)",
            "A project keyword, or an explicit 'project <id>' phrase, switches the "
            "active corpus (ACL permitting).",
        ),
        (
            "alice", "alpha", ["what is the capital of France?"],
            "Missing evidence (out-of-scope question)",
            "When the best score is below threshold the bot refuses with "
            "NO_EVIDENCE instead of hallucinating.",
        ),
        (
            "alice", "gamma", ["what is the refund window in days?"],
            "Contradictory documents (30-day vs 7-day refund)",
            "Two high-scoring docs with opposing negation and shared keywords "
            "raise CONTRADICTION and cite both for clarification.",
        ),
    ]
    demos: list[dict[str, object]] = []
    for user, project, questions, title, fix in specs:
        kbs, acl = build_seed_environment()
        bot = Chatbot(kbs, acl)
        bot.login(user)
        bot.select_project(project)
        turns: list[dict[str, object]] = []
        for question in questions:
            resp = bot.ask(question)
            turns.append(
                {
                    "question": question,
                    "mode": resp["mode"],
                    "sources": resp["sources"],
                    "answer": resp["answer"],
                }
            )
        demos.append({"title": title, "turns": turns, "fix_explanation": fix})
    return demos


if __name__ == "__main__":
    kbs, acl = build_seed_environment()
    bot = Chatbot(kbs, acl)
    bot.login("alice")

    print("== Sample Q&A 1 ==")
    bot.select_project("alpha")
    print(bot.ask("how many annual leaves do employees get?"))

    print("== Sample Q&A 2 ==")
    bot.select_project("beta")
    print(bot.ask("how do I deploy to production?"))

    print("== Failure demo: cross-project leak attempt ==")
    print(bot.ask("how many annual leaves do employees get?"))
