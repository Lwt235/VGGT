#!/usr/bin/env python3
"""One-shot authorized continuation: Full exit -> fresh Local -> final evaluation.

Does not load/alter the running Full model or signal any process. It waits for
that exact Full PID, verifies 18k completion, then launches the requested jobs
sequentially with a fresh live GPU resource check for every GPU command.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT,configure_local_storage
configure_local_storage()


def same_full_process(pid,run):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return len(args)>2 and args[2]==b'scripts/train_step2.py' and b'--run' in args and run.encode() in args
    except OSError:return False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full-pid',type=int)
    ap.add_argument('--resume-baseline',action='store_true',help='Restore this baseline own complete state after resource migration')
    ap.add_argument('--full',default='step2_v5_full_seed17');ap.add_argument('--baseline',default='step2_nomemory_seed17')
    args=ap.parse_args();full=PROJECT_ROOT/'results'/args.full;baseline=PROJECT_ROOT/'results'/args.baseline
    if baseline.exists() != args.resume_baseline:ap.error('Baseline directory requires explicit own-state resume; fresh runs require a new directory')
    if args.full_pid is not None:
        if not same_full_process(args.full_pid,args.full):ap.error('Exact running Full process not found')
        print('Waiting for this Full process only:',args.full_pid,args.full,flush=True)
        while same_full_process(args.full_pid,args.full):time.sleep(5)
    summary=json.loads((full/'summary.json').read_text())
    if summary.get('completed_steps')!=18000:raise RuntimeError('Full exited before 18k; no baseline launched')
    records=[json.loads(l) for l in (full/'metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in records]!=list(range(1,18001)):raise RuntimeError('Full episode schedule is incomplete/duplicated')
    for step in (0,1500,3000,6000,12000,18000):
        if not (full/f'evaluation_{step:05d}_full.json').is_file():raise RuntimeError('Missing Full milestone evaluation')
    print('Full completed 18k; '+('restoring own No-Memory state.' if args.resume_baseline else 'launching fresh official No-Memory.'),flush=True)
    py=sys.executable
    def execute(command,log,gpu=False):
        if gpu:
            command=[py,'-B','scripts/run_on_available_gpu.py','--min-free-mib','24576','--memory-fraction','.40','--',*command]
        print('Starting:',command,'log:',log,flush=True)
        with log.open('w',buffering=1) as handle:
            subprocess.run(command,cwd=PROJECT_ROOT,stdout=handle,stderr=subprocess.STDOUT,
                           env=dict(os.environ,REQUIRE_GSPLAT='1'),check=True)
        print('Completed:',log,flush=True)
    baseline_log=PROJECT_ROOT/'results'/('step2_nomemory_seed17_resume_console.log' if args.resume_baseline else 'step2_nomemory_seed17_console.log')
    command=[py,'-B','scripts/train_step2_nomemory.py','--run',args.baseline,'--until','18000']
    if args.resume_baseline:command.append('--resume')
    execute(command,baseline_log,True)
    if args.resume_baseline:
        original=PROJECT_ROOT/'results'/'step2_nomemory_seed17_console.log'
        (baseline/'console.log').write_text(original.read_text()+'\nOWN-STATE RESOURCE MIGRATION\n'+baseline_log.read_text())
    else:shutil.copyfile(baseline_log,baseline/'console.log')
    if json.loads((baseline/'summary.json').read_text()).get('completed_steps')!=18000:raise RuntimeError('No-Memory incomplete')
    execute([py,'-B','scripts/diagnose_step2_final.py','--full',args.full,'--baseline',args.baseline,'--step','18000'],full/'final_diagnose_console.log',True)
    depth_path=full/'robotics_depth_18000.json'
    depth=json.loads(depth_path.read_text()) if depth_path.exists() else {}
    precomputed=all(depth.get(tag,{}).get('run_name')==args.full and depth.get(tag,{}).get('checkpoint_step')==step
                    for tag,step in [('old',3000),('new',18000)])
    if precomputed:print('Using completed parallel Full 3000/18000 read-only depth diagnostic.',flush=True)
    else:execute([py,'-B','scripts/diagnose_step2_depth.py','--old',args.full,'--old-step','3000','--new',args.full,'--step','18000'],full/'depth_3000_to_18000_console.log',True)
    execute([py,'-B','scripts/compare_step2_final.py','--full',args.full,'--baseline',args.baseline,'--step','18000'],full/'final_compare_console.log')
    execute([py,'-B','-m','pytest','tests','-q'],full/'final_tests_18000.log',True)
    print('All training, evaluation and strict tests naturally completed; ready for image inspection and report.',flush=True)

if __name__=='__main__':main()
