"""Bounded BIBE revision follow-up. Never modifies original experiment outputs."""
import argparse, contextlib, io, json, sys, time, warnings
from pathlib import Path
from copy import deepcopy
import numpy as np
import scipy.io as sio
import torch

ap=argparse.ArgumentParser()
ap.add_argument('--workspace',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--participant',type=int,required=True)
args=ap.parse_args()
sys.path.insert(0,str(args.workspace))
from scripts import csl_author_recipe as r, csl_component_study as c, csl_search_batch as search
from scripts.verify_csl_author_recipe import restore
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
warnings.filterwarnings('ignore',message='There is a performance drop')
PRIMARY=args.workspace/'research/runs/20260907_csl_components_confirmatory'
OUT=args.output/f'participant{args.participant}'
OUT.mkdir(parents=True,exist_ok=True)
IDENTITY=torch.tensor([0.,0.,0.,1.,1.,0.,0.])
SCALE=torch.tensor([max(abs(lo-float(z)),abs(hi-float(z))) for (lo,hi),z in zip(r.BOUNDS,IDENTITY)])

def penalty(params):
    return ((params-IDENTITY)/SCALE).square().mean(-1)

def regularized(base,sx,sy,cx,cy,gain):
    m=deepcopy(base).eval();m.adaptation_phase=True
    for p in m.parameters():p.requires_grad=False
    if gain:m.get_session_means(sx[sy==int(cy[0])],cx)
    proto=cx.square().mean(0,keepdim=True).sqrt()
    m.input_transform.mode='bicubic'
    a=search.candidates();loss=search.losses(m,proto,cy[:1],a)
    objective=loss+penalty(a) # fixed lambda=1, no outcome-based search
    idx=int(objective.argmin())
    inv=r.load_initial_functions()['get_inv_constrained_params'](*a[idx],boundaries=r.BOUNDS)
    with torch.no_grad():
        for name,value in zip(search.NAMES,inv):getattr(m.input_transform,name)[0].copy_(value)
    for p in m.input_transform.parameters():p.requires_grad=True
    m.input_transform.mode='bilinear'
    opt=torch.optim.Adam(m.input_transform.parameters(),lr=.01)
    for _ in range(500):
        opt.zero_grad()
        physical=torch.stack(m.input_transform.get_constrained_params())
        objective=torch.nn.functional.cross_entropy(m(proto),cy[:1])+penalty(physical)
        assert torch.isfinite(objective)
        objective.backward();opt.step()
    m.input_transform.mode='bicubic';m.eval()
    for key,v in base.state_dict().items():
        if not key.startswith('input_transform.'):assert torch.equal(v,m.state_dict()[key])
    return m,dict(lambda_identity=1.,initial_index=idx,initial_ce=float(loss[idx]),final_objective=float(objective.detach()))

def preprocess(original,variant):
    """Rebuild both sessions with one explicit change from original processing."""
    nets,utils,tensor,proc=r.modules()
    class Helper(tensor.EMGData):
        def get_images(self,x):
            return r.provider_images(x) if variant=='baseline_on' else super().get_images(x)
    h=Helper.__new__(Helper);h.dataset='csl';h.fs=2048;h.Mrms=512
    h.remove_baseline='mean-square' if variant=='baseline_on' else 'mean_square'
    result=[]
    dest=OUT/variant/'frames';dest.mkdir(parents=True,exist_ok=True)
    for session in [1,2]:
        directory=r.DATA/f'subject{args.participant}/session{session}'
        baseline=h.get_baseline(str(directory))
        for gesture in r.RAW_GESTURES:
            entries=[e for e in original if e['session']==session and e['gesture']==gesture]
            cells=sio.loadmat(directory/f'gest{gesture}.mat')['gestures']
            for e in entries:
                raw=cells[e['rep'],0].copy()
                emg=raw.T;emg=emg-emg.mean(0,keepdims=True)
                emg=proc.bandstop(proc.bandpass(emg,fs=2048),fs=2048)
                lo,hi=h.segment(emg,baseline)
                rms=proc.get_rms_signal(emg,Mrms=512)
                if variant=='baseline_on':rms=np.sqrt(np.maximum(rms**2-baseline,0))
                x=utils.median_pool_2d(torch.from_numpy(h.get_images(rms)[lo:hi].copy()).float()).numpy()
                assert len(x)>0 and np.isfinite(x).all()
                path=dest/(e['id']+'.npy');np.save(path,x)
                if variant=='baseline_on':assert (int(lo),int(hi))==(e['start'],e['end'])
                result.append(dict(e,cache_path=str(path),cache_sha256=r.sha(path),frames=len(x),start=int(lo),end=int(hi)))
    r.dump(OUT/variant/'manifest.json',result)
    return result

def data(manifest):
    sm=[dict(e,role='source') for e in manifest if e['session']==1]
    tm=[dict(e,role='calibration' if e['rep']==0 else 'score') for e in manifest if e['session']==2]
    sx,sy,se=r.role_data(sm,'source');tx,ty,te=r.role_data(tm,'score')
    assert not {e['id'] for e in se}&{e['id'] for e in te}
    assert len(te)==72 and len(se)==80
    return sx,sy,tx,ty,te,tm

def evaluate(base,manifest,variant,regularize=False):
    sx,sy,tx,ty,te,tm=data(manifest)
    frozen=r.pred(base,tx);rows=[]
    for gesture in r.RAW_GESTURES:
        folder=OUT/variant/f'g{gesture:02}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists():rows.extend(json.loads((folder/'results.json').read_text()));continue
        cx,cy,ce=r.role_data(tm,'calibration',gesture);label=int(cy[0])
        assert len(ce)==1 and ce[0]['id'] not in {e['id'] for e in te}
        preds={'frozen':frozen};local=[dict(method='frozen',metrics=r.metrics(frozen,ty,te,label),calibration_frame_accuracy=float(np.mean(r.pred(base,cx)==np.asarray(cy))))]
        for method in ['spatial_only','spatial_gain']:
            with contextlib.redirect_stdout(io.StringIO()):
                m,log=regularized(base,sx,sy,cx,cy,method=='spatial_gain') if regularize else c.adapt(base,sx,sy,cx,cy,method)
            c.save(m,folder/method);pred=r.pred(m,tx);preds[method]=pred
            # Reload and use another inference batch size to check saved state and predictions.
            loaded=restore(method,folder)
            with torch.no_grad():replay=np.concatenate([loaded(z).argmax(1).numpy() for z in tx.split(257)])
            np.testing.assert_array_equal(pred,replay)
            local.append(dict(method=method,metrics=r.metrics(pred,ty,te,label),calibration_frame_accuracy=float(np.mean(r.pred(m,cx)==np.asarray(cy))),log=log))
        local=[dict(participant=args.participant,source_session=1,target_session=2,gesture=gesture,variant=variant,**z) for z in local]
        np.savez_compressed(folder/'predictions.npz',**preds,labels=np.asarray(ty),trial_ids=[e['id'] for e in te],lengths=[e['frames'] for e in te])
        r.dump(folder/'results.json',local);rows.extend(local)
        print('Completed',args.participant,variant,gesture,flush=True)
    return rows

original=[e for e in json.loads((PRIMARY/'manifest.json').read_text()) if e['participant']==args.participant and e['session'] in [1,2]]
assert len(original)==160
started=time.perf_counter();rows=[]
base=restore('source',PRIMARY/f'subject{args.participant}/source_s1')
rows.extend(evaluate(base,original,'identity_regularized',True))
for variant in ['baseline_on','literal_mapping']:
    manifest=preprocess(original,variant);sx,sy,_,_,_,_=data(manifest)
    folder=OUT/variant;checkpoint=folder/'source.pt'
    if checkpoint.exists():base=restore('source',folder)
    else:
        nets,*_=r.modules();torch.manual_seed(42);base=r.make_model(nets).train()
        with contextlib.redirect_stdout(io.StringIO()):log=r.fit(base,sx,sy,15,.001,3)
        base.eval();c.save(base,folder/'source');r.dump(folder/'source_training.json',log)
    rows.extend(evaluate(base,manifest,variant))
r.dump(OUT/'results.json',dict(results=rows,checkpoint_replay_passed=True,wall_seconds=time.perf_counter()-started))
print('DONE',args.participant,time.perf_counter()-started,flush=True)
