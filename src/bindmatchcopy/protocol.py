"""Shared paper evaluation counts and sequence allocation."""

EVAL_SAMPLES = 200


def condition_counts(domains, n_samples=EVAL_SAMPLES):
    if not domains or len(set(domains)) != len(domains) or n_samples < len(domains):
        raise ValueError("Need distinct domains and at least one sequence per domain")
    return {
        domain: n_samples // len(domains) + int(i < n_samples % len(domains))
        for i, domain in enumerate(domains)
    }
