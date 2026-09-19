"""Check completed D tiny, assets and deployment; never claim/lock/launch."""
import argparse
import json
import d_trial_gates as g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-sha256', required=True)
    args = parser.parse_args()
    host = g.profiles.require_host()
    policy = g.policy_gate(args.policy_sha256)
    sources = g.source_code_manifest(policy['value'])
    transfer = g.transfer_gate(host)
    g.runtime_gate(host, sources)
    tiny = g.tiny_gate('01234567', host, policy['value'])
    g.sample_gate(g.DEFAULT_SAMPLE, transfer=transfer)
    g.weight_manifest(transfer['verified'])
    g.parallelism(g.selected_profile('01234567'))
    print(json.dumps(dict(status='passed', mode=g.MODE, policy_sha256=args.policy_sha256,
                          tiny_run_id=tiny['status']['run_id'], sample=g.DEFAULT_SAMPLE,
                          npu_started=False, submission_claimed=False,
                          fresh_resource_check_still_required=True)))


if __name__ == '__main__':
    main()
