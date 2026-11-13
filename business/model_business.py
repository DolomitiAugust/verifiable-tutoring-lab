"""Reproducible commercial scenario arithmetic, never a measurement of demand."""
from pathlib import Path
import json, math, csv, hashlib

ROOT = Path(__file__).resolve().parent

def unit(a, s):
    n = a['contract']['students_per_semester'] * a['contract']['semesters_per_year']
    p = a['api_price']; shared = a['shared']
    turns = n * s['turns_per_enrolment']
    # Tokens already sum ALL three roles. Do not multiply by roles again.
    token_cost = (s['input_tokens_total_per_turn'] * p['input_cny_per_million'] +
                  s['output_tokens_total_per_turn'] * p['output_cny_per_million']) / 1e6
    api = turns * token_cost * s['api_retry_multiplier']
    review_hours = turns * s['review_fraction'] * s['review_minutes'] / 60
    review = review_hours * shared['contractor_cny_per_hour']
    onboarding = s['onboarding_hours'] * shared['contractor_cny_per_hour']
    recurring = api + review + s['cloud_per_contract_year']
    delivery = recurring + onboarding
    revenue = n * s['price_per_enrolment']
    contribution = revenue - delivery
    after_cac = contribution - s['cac_per_new_contract']
    fixed = 12 * (shared['fixed_monthly_team'] + shared['fixed_monthly_other'])
    # Positive contribution is a necessary condition for finite break-even.
    return dict(id=s['id'], label=s['label'], assumptions=s, enrolments=n, turns=turns,
                revenue=revenue, api=api, review=review, review_hours=review_hours,
                onboarding=onboarding, cloud=s['cloud_per_contract_year'],
                recurring_delivery=recurring, first_year_delivery=delivery,
                first_year_contribution=contribution, gross_margin=contribution/revenue,
                contribution_after_cac=after_cac,
                new_contracts_to_cover_annual_fixed=math.ceil(fixed/after_cac) if after_cac>0 else None,
                first_year_contracts_before_cac=math.ceil(fixed/contribution) if contribution>0 else None,
                new_contracts_to_cover_fixed_and_launch=math.ceil((fixed+a['cash_plan']['one_time_launch_cny'])/after_cac) if after_cac>0 else None)

def cash(a, u, lag):
    plan=a['cash_plan']; live=plan['go_live_months'];s=u['assumptions']
    balance=plan['fundraise_cny'];rows=[]
    for month in range(1,13):
        new=live.count(month);active=sum(t<=month for t in live)
        receipts=sum(t+lag==month for t in live)*u['revenue']
        fixed=a['shared']['fixed_monthly_team']+a['shared']['fixed_monthly_other']
        launch=plan['one_time_launch_cny'] if month==1 else 0
        recurring=active*u['recurring_delivery']/12
        onboarding=new*u['onboarding'];acquisition=new*s['cac_per_new_contract']
        total=fixed+launch+recurring+onboarding+acquisition
        balance+=receipts-total
        rows.append(dict(month=month, active_contracts=active, new_contracts=new,
                         revenue_recognized=active*u['revenue']/12, receipts=receipts,
                         fixed=fixed, launch=launch, recurring_delivery=recurring,
                         onboarding=onboarding, acquisition=acquisition,
                         cash_out=total, closing_cash=balance))
    return dict(collection_lag_months=lag, months=rows,
                recognized_revenue=sum(r['revenue_recognized'] for r in rows),
                cash_receipts=sum(r['receipts'] for r in rows),
                cash_out=sum(r['cash_out'] for r in rows), closing_cash=balance,
                minimum_month_end_cash=min(r['closing_cash'] for r in rows),
                year_end_arr=len(live)*u['revenue'])

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent/"outputs/business")
    args=parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'business_assumptions.json';a=json.loads(source.read_text(encoding='utf8'))
    units=[unit(a,s) for s in a['scenarios']];ref=next(u for u in units if u['id']=='reference')
    grid=[]
    for price in a['sensitivity']['price_per_enrolment']:
        for frac in a['sensitivity']['review_fraction']:
            s=dict(ref['assumptions'],price_per_enrolment=price,review_fraction=frac)
            u=unit(a,s)
            grid.append(dict(price_per_enrolment=price,review_fraction=frac,
                             contribution_after_cac=u['contribution_after_cac'],
                             new_contracts_to_cover_annual_fixed=u['new_contracts_to_cover_annual_fixed']))
    cash_cases=[cash(a,ref,lag) for lag in a['sensitivity']['collection_lags_months']]
    critical=(ref['revenue']-ref['api']-ref['onboarding']-ref['cloud']-ref['assumptions']['cac_per_new_contract'])/(ref['turns']*ref['assumptions']['review_minutes']/60*a['shared']['contractor_cny_per_hour'])
    out=dict(status=a['status'], currency='CNY', assumption_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             units=units, sensitivity=grid, cash_scenarios=cash_cases,
             cash_reference=next(c for c in cash_cases if c['collection_lag_months']==a['cash_plan']['invoice_collection_lag_months']),
             fixed_annual=12*(a['shared']['fixed_monthly_team']+a['shared']['fixed_monthly_other']),
             reference_zero_contribution_review_fraction=critical,
             buyer_hours_to_offset_annual_price=ref['revenue']/a['shared']['buyer_teacher_value_cny_per_hour'],
             buyer_minutes_to_offset_price_per_enrolment=ref['assumptions']['price_per_enrolment']/a['shared']['buyer_teacher_value_cny_per_hour']*60,
             market_funnel=a['market_funnel'],
             exclusions=['Taxes and financing fees','Refunds and bad debt','Hardware purchase','Real customer demand, adoption, renewal and churn','API educational quality and latency','Changing model list price and token distribution'])
    assert abs(ref['api']-427.68)<1e-7
    assert abs(ref['first_year_delivery']-15307.68)<1e-7
    assert ref['new_contracts_to_cover_annual_fixed']==13
    assert units[0]['contribution_after_cac']<0 and units[0]['new_contracts_to_cover_annual_fixed'] is None
    assert abs(out['cash_reference']['recognized_revenue']-108000)<1e-7
    assert abs(out['cash_reference']['cash_receipts']-144000)<1e-7
    assert out['cash_reference']['year_end_arr']==216000
    (output/'business_outputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    for name,rows in [('unit_economics', [{k:v for k,v in u.items() if k!='assumptions'} for u in units]),('sensitivity',grid),('monthly_cash',out['cash_reference']['months'])]:
        with (output/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'reference':ref,'cash':{k:v for k,v in out['cash_reference'].items() if k!='months'},'critical_review_fraction':critical},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
