"""Tests for the extended structural-redaction rules added so tool-output
carryover and recall-gate debug logs cannot leak credentials.
"""

import pytest

from src.jarvis.utils.redact import redact, scrub_secrets


@pytest.mark.unit
class TestVendorAccessKeys:
    def test_aws_akia_key_redacted(self):
        out = redact("key=AKIAIOSFODNN7EXAMPLE rest")
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "[REDACTED_AWS_KEY]" in out

    def test_aws_asia_key_redacted(self):
        out = redact("ASIAIOSFODNN7EXAMPLE")
        assert "ASIAIOSFODNN7EXAMPLE" not in out
        assert "[REDACTED_AWS_KEY]" in out

    def test_stripe_live_secret_redacted(self):
        token = "sk_live_" + "a" * 24
        out = redact(f"see {token} please")
        assert token not in out
        assert "[REDACTED_STRIPE_KEY]" in out

    def test_stripe_test_publishable_redacted(self):
        token = "pk_test_" + "Z" * 24
        out = redact(token)
        assert token not in out
        assert "[REDACTED_STRIPE_KEY]" in out

    def test_github_pat_redacted(self):
        token = "ghp_" + "A" * 36
        out = redact(token)
        assert token not in out
        assert "[REDACTED_GH_TOKEN]" in out

    def test_openai_key_redacted(self):
        token = "sk-" + "A" * 40
        out = redact(token)
        assert token not in out
        assert "[REDACTED_OPENAI_KEY]" in out

    def test_google_api_key_redacted(self):
        token = "AIza" + "B" * 35
        out = redact(token)
        assert token not in out
        assert "[REDACTED_GOOG_KEY]" in out


@pytest.mark.unit
class TestAuthorizationHeaders:
    def test_bearer_header_redacted(self):
        out = scrub_secrets("Authorization: Bearer abc.def.ghi")
        assert "abc.def.ghi" not in out
        assert "Authorization: Bearer [REDACTED]" in out

    def test_basic_header_redacted(self):
        out = scrub_secrets("Authorization: Basic dXNlcjpwYXNz")
        assert "dXNlcjpwYXNz" not in out
        assert "Authorization: Basic [REDACTED]" in out


@pytest.mark.unit
class TestKeywordAnchoredCredentials:
    def test_refresh_token_keyword_redacted(self):
        out = redact("refresh_token=abcdef123456")
        assert "abcdef123456" not in out
        assert "refresh_token=[REDACTED]" in out

    def test_access_token_keyword_redacted(self):
        out = redact("access_token: zzz999")
        assert "zzz999" not in out
        assert "access_token=[REDACTED]" in out

    def test_session_id_redacted(self):
        out = redact("session_id=deadbeefcafe")
        assert "deadbeefcafe" not in out
        assert "session_id=[REDACTED]" in out

    def test_oauth_token_redacted(self):
        out = redact("oauth_token=qwertyuiop")
        assert "qwertyuiop" not in out
        assert "oauth_token=[REDACTED]" in out


# ── Recognising our own placeholders ──────────────────────────────────
#
# Anything that stores user text long-term needs to tell a real value
# from the marker left where one used to be. Writing "[REDACTED_EMAIL]"
# into a memory file keeps nothing useful and tells the user their email
# was saved.

class TestContainsRedactionPlaceholder:

    def test_plain_text_carries_no_placeholder(self):
        from src.jarvis.utils.redact import contains_redaction_placeholder

        assert contains_redaction_placeholder("Il vit à Lyon.") is False

    def test_a_labelled_placeholder_is_recognised(self):
        from src.jarvis.utils.redact import contains_redaction_placeholder, redact

        scrubbed = redact("Son email est hocsman92@gmail.com")
        assert contains_redaction_placeholder(scrubbed) is True

    def test_a_bare_placeholder_is_recognised(self):
        from src.jarvis.utils.redact import contains_redaction_placeholder

        assert contains_redaction_placeholder("password=[REDACTED]") is True

    def test_every_label_the_scrubber_emits_is_recognised(self):
        """Asserted against the rules themselves rather than a copied list,
        so a new pattern cannot add a label this misses."""
        from src.jarvis.utils import redact as redact_mod

        for _pattern, replacement in redact_mod._REDACTION_RULES:
            if "[REDACTED" in replacement:
                assert redact_mod.contains_redaction_placeholder(replacement), (
                    f"{replacement!r} is emitted by the scrubber but not recognised"
                )

    def test_empty_and_none_are_safe(self):
        from src.jarvis.utils.redact import contains_redaction_placeholder

        assert contains_redaction_placeholder("") is False
        assert contains_redaction_placeholder(None) is False
