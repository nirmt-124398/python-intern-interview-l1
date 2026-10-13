"""Tests for the Question 6 multi-knowledge-base chatbot."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import q6_chatbot as q6  # noqa: E402


@pytest.fixture()
def bot() -> q6.Chatbot:
    kbs, acl = q6.build_seed_environment()
    instance = q6.Chatbot(kbs, acl)
    instance.login("alice")
    return instance


def test_retrieval_and_source_citation(bot: q6.Chatbot) -> None:
    bot.select_project("alpha")
    resp = bot.ask("how many annual leaves do employees get?")
    assert resp["mode"] == q6.ANSWERED
    assert "a1" in resp["sources"]
    assert "20" in resp["answer"]


def test_explicit_topic_switch(bot: q6.Chatbot) -> None:
    bot.select_project("alpha")
    resp = bot.ask("project beta how do I deploy to production?")
    assert resp["mode"] == q6.ANSWERED
    assert resp["project_id"] == "beta"
    assert "b1" in resp["sources"]
    assert bot.active_project == "beta"


def test_keyword_topic_switch(bot: q6.Chatbot) -> None:
    bot.select_project("alpha")
    resp = bot.ask("show me the gamma refund window in days")
    assert resp["project_id"] == "gamma"
    assert any(s.startswith("g") for s in resp["sources"])


def test_missing_evidence_returns_no_evidence(bot: q6.Chatbot) -> None:
    bot.select_project("alpha")
    resp = bot.ask("what is the capital of France?")
    assert resp["mode"] == q6.NO_EVIDENCE
    assert resp["sources"] == []
    assert "alpha" in resp["answer"]


def test_cross_user_access_raises_permission_error() -> None:
    kbs, acl = q6.build_seed_environment()
    bob = q6.Chatbot(kbs, acl)
    bob.login("bob")
    with pytest.raises(PermissionError):
        bob.select_project("alpha")


def test_unauthorized_ask_is_denied() -> None:
    kbs, acl = q6.build_seed_environment()
    bob = q6.Chatbot(kbs, acl)
    bob.login("bob")
    bob.select_project("beta")
    resp = bob.ask("project alpha what is the annual leave policy?")
    assert resp["mode"] == q6.DENIED
    assert resp["sources"] == []


def test_no_cross_project_leakage(bot: q6.Chatbot) -> None:
    bot.select_project("beta")
    resp = bot.ask("how many annual leaves do employees get?")
    assert resp["project_id"] == "beta"
    assert all(not s.startswith("a") for s in resp["sources"])
    # Sanity: alpha really does hold the answer, so isolation is what blocked it.
    bot.select_project("alpha")
    assert "a1" in bot.ask("how many annual leaves do employees get?")["sources"]


def test_contradiction_detected(bot: q6.Chatbot) -> None:
    bot.select_project("gamma")
    resp = bot.ask("what is the refund window in days?")
    assert resp["mode"] == q6.CONTRADICTION
    assert set(resp["sources"]) == {"g1", "g2"}


def test_history_followup_resolution(bot: q6.Chatbot) -> None:
    bot.select_project("alpha")
    first = bot.ask("how many annual leaves do employees get?")
    assert "a1" in first["sources"]
    follow_up = bot.ask("what about sick?")
    assert follow_up["mode"] == q6.ANSWERED
    assert "a2" in follow_up["sources"]


def test_followup_without_history_has_no_evidence(bot: q6.Chatbot) -> None:
    bot.select_project("alpha")
    assert bot.ask("what about sick?")["mode"] == q6.NO_EVIDENCE


def test_audit_log_records_every_turn(bot: q6.Chatbot) -> None:
    bot.select_project("alpha")
    bot.ask("how many annual leaves do employees get?")
    bot.ask("what is the capital of France?")
    assert len(bot.audit_log) == 2
    user, project, _question, sources = bot.audit_log[0]
    assert (user, project) == ("alice", "alpha")
    assert "a1" in sources


def test_login_required_returns_denied() -> None:
    kbs, acl = q6.build_seed_environment()
    anon = q6.Chatbot(kbs, acl)
    assert anon.ask("anything")["mode"] == q6.DENIED
