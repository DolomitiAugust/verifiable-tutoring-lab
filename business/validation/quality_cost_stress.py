"""Extend the existing commercial model; all review rates remain assumptions."""
from pathlib import Path
import csv, hashlib, importlib.util, json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def main():
    source = ROOT / 'business/business_assumptions.json'
    assumptions = json.loads(source.read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('original_business', ROOT / 'business/model_business.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ref = next(s for s in assumptions['scenarios'] if s['id'] == 'reference')
    rows = []
    for fraction in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        result = module.unit(assumptions, dict(ref, review_fraction=fraction))
        # Independently recompute the review burden, excluding fixed payroll.
        expected_hours = 600 * 120 * fraction * 3 / 60
        assert abs(result['review_hours'] - expected_hours) < 1e-8
        rows.append(dict(review_fraction=fraction, annual_review_hours=result['review_hours'],
                         annual_review_cost_cny=result['review'],
                         contribution_after_incremental_cac_cny=result['contribution_after_cac'],
                         contracts_covering_fixed=result['new_contracts_to_cover_annual_fixed']))
    assert rows[-1]['contracts_covering_fixed'] is None
    buyer = []
    for minutes in [10, 20, 40, 80, 120]:
        value = minutes / 60 * assumptions['shared']['buyer_teacher_value_cny_per_hour']
        buyer.append(dict(net_minutes_saved_per_enrolment=minutes,
                          assumed_teacher_time_value_cny=value,
                          assumed_price_cny=ref['price_per_enrolment'],
                          value_minus_price_cny=value-ref['price_per_enrolment']))
    payload = {'status':'scenario_not_market_measurement',
               'source_assumptions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
               'review_rates_are_measured':False,
               'rule_fallback_is_not_equivalent_to_human_review':True,
               'rows':rows,'buyer_value':buyer,
               'decision':'Do not assume 2% review or sufficient buyer savings without operational and buyer evidence.',
               'actual_customer_interviews':0,'actual_paid_contracts':0,
               'actual_teacher_blind_reviews':0}
    (OUT/'results').mkdir(exist_ok=True)
    (OUT/'results/quality_cost_stress.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    for name, data in [('review_burden',rows),('buyer_value',buyer)]:
        with (OUT/f'results/{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__ == '__main__':main()
