"""Read saved records and check authored references. Never load or invoke a model."""
from pathlib import Path
import ast, contextlib, io, json, sqlite3, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def read(p):return json.loads((ROOT/p).read_text(encoding="utf-8"))
def main():
    checks=[]
    def check(label,ok):
        checks.append((label,bool(ok)))
    counts={"sft":1000,"preference":500,"dev":36,"evaluation":60,"preference_eval":100,"independent_test":24}
    for name,n in counts.items():
        rows=[json.loads(s) for s in (ROOT/f"synthesis/dataset/{name}.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
        check("data_count."+name,len(rows)==n)
    ledger=[json.loads(s) for s in (ROOT/"synthesis/dataset/audit/candidates.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    check("candidate_count",len(ledger)==8240)
    verified=0
    for row in ledger:
        # These are bounded authored reference programs, never generated model code.
        for code,answer in [(row.get("code"),row.get("answer")),(row.get("transfer_code"),row.get("transfer_answer"))]:
            if code is None or answer is None:continue
            out=io.StringIO()
            with contextlib.redirect_stdout(out):exec(compile(code,"<authored-reference>","exec"),{})
            if out.getvalue().strip()!=str(answer).strip():raise AssertionError("reference mismatch: "+row["candidate_id"])
            verified+=1
    check("authored_reference_programs",verified>0)
    for p in ROOT.rglob("*.py"):
        if any(x in {".git",".venv","node_modules"} for x in p.parts):continue
        ast.parse(p.read_text(encoding="utf-8"),filename=p.relative_to(ROOT).as_posix())
    check("python_syntax",True)
    for rel,n,passed in [("training/imports/run_5bf9f4d325f1/audit.json",552,True),("training/retry_imports/retry_048f9a92c0ef/audit.json",965,False),("training/retry_imports/retry_048f9a92c0ef_numeric_review/audit.json",966,True)]:
        obj=read(rel);check("saved_audit."+str(n),obj.get("check_count")==n and obj.get("passed")==passed)
    review=read("training/retry_imports/retry_048f9a92c0ef/new_full_text_fact_review.json")
    for stage,expected in [("sft",(3,16,5)),("dpo",(4,7,13)),("continued_sft",(5,11,8))]:
        counts=review["summary"][stage]["full_text_main_output_counts"]
        check("full_text_review."+stage,tuple(counts[k] for k in ["correct","wrong","unresolved"])==expected)
    cases=read("tutor/legacy/results/combined.json")
    check("historical_two_success_two_failure",cases["showcase"]["meets_two_plus_two"] is True)
    for p in ROOT.rglob("*.sqlite"):
        with sqlite3.connect("file:"+p.as_posix()+"?mode=ro",uri=True) as db:check("sqlite."+p.relative_to(ROOT).as_posix(),db.execute("pragma integrity_check").fetchone()[0]=="ok")
    from business.model_business import unit
    assumptions=read("business/business_assumptions.json");saved=read("business/results/business_outputs.json")
    for scenario in assumptions["scenarios"]:
        recalculated=unit(assumptions,scenario);old=next(x for x in saved["units"] if x["id"]==scenario["id"])
        for key in ["api","first_year_delivery","contribution_after_cac"]:check("business."+scenario["id"]+"."+key,abs(recalculated[key]-old[key])<1e-7)
    result={"passed":all(ok for _,ok in checks),"checks":len(checks),"reference_programs_executed":verified,"failed":[name for name,ok in checks if not ok],"model_calls":0,"new_training":False,"scope":"Saved-record consistency, authored references, Python syntax, SQLite integrity and scenario arithmetic; no teaching efficacy certification."}
    print(json.dumps(result,indent=2));return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
