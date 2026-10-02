"""Reproduce a portfolio entry comparison from a frozen QuantSim input JSON."""

import argparse
import json
from pathlib import Path

from src.analytics.portfolio_entry_export import analyze_entry_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
    study, sensitivity = analyze_entry_snapshot(snapshot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study.coverage.to_csv(args.output_dir / 'coverage.csv', index=False)
    if study.study is None:
        raise SystemExit('Insufficient overlapping price history; see coverage.csv.')
    study.study.cases.to_csv(args.output_dir / 'starts.csv', index=False)
    study.study.summary.to_csv(args.output_dir / 'summary.csv', index=False)
    sensitivity.to_csv(args.output_dir / 'sensitivity.csv', index=False)
    metadata = dict(input_sha256=snapshot['input_sha256'], as_of=snapshot['as_of'],
                    weights=snapshot['weights'], config=snapshot['config'],
                    coverage=study.evidence_weight, full_portfolio_covered=study.full_portfolio_covered,
                    latest_common_session=study.latest_common_session, warnings=study.warnings,
                    source=snapshot['source'], approval=snapshot['approval'])
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
