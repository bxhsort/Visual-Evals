"""Offline integration regression for managed generation/scoring/report paths.

Every generator, scorer, GPU launcher, and coverage subprocess is intercepted.
Only the stdlib experiment path helper and shell setup are executed in an isolated
copy; this test requires no models, datasets package, GPU, API, or network.
"""
import ast, json, os, shlex, shutil, subprocess, sys, tempfile, textwrap
import unittest


def check_wrapper_paths():
    from pathlib import Path

    stage=Path(__file__).resolve().parents[4]
    # Read parser declarations without importing GPU/model dependencies. A
    # no-op interpreter must not hide invalid flags such as --use_think when
    # the real unified generator only implements --think.
    generator_flags={}
    for filename in ('gen_images_mp_t2i.py','gen_images_mp_it2i.py'):
        syntax=ast.parse((stage/'eval/gen'/filename).read_text())
        generator_flags[filename]={argument.value for call in ast.walk(syntax)
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
            and call.func.attr=='add_argument' for argument in call.args
            if isinstance(argument,ast.Constant) and isinstance(argument.value,str)
            and argument.value.startswith('--')}
    with tempfile.TemporaryDirectory(prefix='dg2-wrapper-audit-') as temporary:
        root=Path(temporary)
        # Copy only executable control files, never user weights, benchmark data,
        # environments or previous outputs installed in the real repository.
        repo=root/'DG2';repo.mkdir()
        sources=[stage/'eval_t2i_go.sh',stage/'eval_go.sh']
        for directory in (stage/'scripts/eval', stage/'third_party/UMM_Eval/gen/common'):
            sources.extend(path for path in directory.iterdir() if path.is_file() and path.suffix in {'.py', '.sh'})
        sources.extend((stage/'third_party/UMM_Eval').glob('gen/*/*/launch.sh'))
        for source in sources:
            target=repo/source.relative_to(stage)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
        fixture=root/'fixtures';fixture.mkdir()
        model=fixture/'model';model.mkdir();(model/'ema.safetensors').write_bytes(b'fake offline weights')
        data=fixture/'metadata.json';data.write_text(json.dumps({'sample':{'id':'input.png','prompt':'edit','edit_type':'add'}}))
        expected=fixture/'expected.json';expected.write_text(json.dumps([{'key':'sample','task_type':'edit','instruction_language':'en'}]))
        prompts=fixture/'prompts';prompts.mkdir();(prompts/'sample.jsonl').write_text('{"prompt":"sample","type":"color"}\n')
        dataset=fixture/'dataset';dataset.mkdir();(dataset/'state.json').write_text('{}')
        evaluator=fixture/'editcompass-evaluator';(evaluator/'Prompt').mkdir(parents=True)
        rises=fixture/'rise.json';rises.write_text('[]')
        log=root/'calls.jsonl';bin_dir=root/'bin';bin_dir.mkdir()
        stub_code='''#!REALPYTHON
    import json,os,sys
    from pathlib import Path
    args=sys.argv[1:]
    if Path(sys.argv[0]).name in ("python","python3") and args and (args[0]=="-" or Path(args[0]).name in ("experiment_layout.py","output_paths.py")):
        os.execv(REAL_PYTHON,[REAL_PYTHON,*args])
    if Path(sys.argv[0]).name=="ldconfig":
        print("libGL.so.1")
        sys.exit(0)
    # LongText's shell aggregation needs an actual chunk after the intercepted
    # scorer returns. This fixture contains no score and executes no evaluator.
    if any(Path(arg).name=="evaluate_text_reward.py" for arg in args):
        output=Path(args[args.index("--output_dir")+1])
        output.mkdir(parents=True,exist_ok=True)
        (output/"results_chunk0.jsonl").write_text("{}\\n")
    keys=("DG2_RESULT_DIR","DG2_SETTING_NAME","RESULT_ROOT_DIR","RESULT_DIR","OUTPUT_PATH","OUTPUT_DIR","OUTPUT_ROOT","OUTPUT_BASE","THINK_MODE","GENEVAL_PROMPT_MODE","DG2_STUB_ACTIVE_ENV","PATH")
    record={"tool":Path(sys.argv[0]).name,"args":args,"env":{key:os.environ.get(key) for key in keys}}
    if record["tool"]=="bash":
        record["forwarded_env"]=dict(os.environ)
    with open(STUB_LOG,"a") as handle:
        handle.write(json.dumps(record)+"\\n")
    '''.replace('REALPYTHON',sys.executable).replace('REAL_PYTHON',repr(sys.executable)).replace('STUB_LOG',repr(str(log)))
        first_line, remaining=stub_code.split('\n',1)
        stub_code=first_line+'\n'+textwrap.dedent(remaining)
        for command in ('python','python3','torchrun','accelerate','bash','env','ldconfig'):
            p=bin_dir/command;p.write_text(stub_code);p.chmod(0o755)
        # Bash searches PATH for a sourced file called activate. It changes only
        # this test marker; the production script must still prepend its fixed
        # /opt/conda environment path, which is deliberately absent on this host.
        activation_log = (
            'export DG2_STUB_ACTIVE_ENV="$1"\n'
            + shlex.quote(sys.executable) + ' - ' + shlex.quote(str(log)) + ' "$1" <<\'PY\'\n'
            + 'import json, os, sys\n'
            + 'with open(sys.argv[1], "a") as handle:\n'
            + '    handle.write(json.dumps({"tool": "activate", "name": sys.argv[2], "args": [], "env": {"DG2_RESULT_DIR": os.environ.get("DG2_RESULT_DIR"), "DG2_STUB_ACTIVE_ENV": os.environ.get("DG2_STUB_ACTIVE_ENV")}}) + "\\n")\n'
            + 'PY\n'
        )
        (bin_dir/'activate').write_text(activation_log)
        # Shell functions keep this test offline even on a machine where the
        # fixed /opt environments really exist. They intercept commands while
        # preserving the production PATH value and source-activate arguments.
        startup=fixture/'offline-shell-startup.sh'
        interceptors=[]
        for command in ('python','python3','torchrun','accelerate','bash','env','ldconfig'):
            interceptors.append(command+'() { '+shlex.quote(str(bin_dir/command))+' "$@"; }\n')
        interceptors.append('apt-get() { echo "offline test refuses package installation" >&2; return 99; }\n')
        interceptors.append('sudo() { echo "offline test refuses privileged commands" >&2; return 99; }\n')
        interceptors.append('source() { if [[ "$1" == activate ]]; then shift; builtin source '
                            +shlex.quote(str(bin_dir/'activate'))+' "$@"; else builtin source "$@"; fi; }\n')
        startup.write_text(''.join(interceptors))
        env=dict(PATH=str(bin_dir)+':'+os.environ['PATH'],MODEL_PATH=str(model),MODEL_NAME='bagel',
            EXPERIMENT_NAME='offline-consistency',OUTPUT_BASE=str(root/'outputs'),GPU_LIST='0',CUDA_VISIBLE_DEVICES='0',
            RUN_SUMMARY='0',COVERAGE_REPORT='1',DG2_STUB_ACTIVE_ENV='current-eval-environment',BASH_ENV=str(startup),
            COVERAGE_METADATA_FILE=str(expected),GEDIT_DATASET_PATH=str(dataset),RISE_DATA_JSON=str(rises),
            EDIT_COMPASS_DATA_ROOT=str(dataset),EDIT_COMPASS_BENCHMARK_ROOT=str(evaluator),METADATA_FILE=str(data),
            TIIFBENCH_PROMPT_DIR=str(prompts),TIIFBENCH_EVAL_PROMPT_DIR=str(prompts),TIIF_BENCH_ROOT=str(fixture),
            API_KEY='offline-synthetic-key',INSTRUCTION_LANGUAGES='en',ONEIGBENCH_LANGS='en',LONGTEXTBENCH_LANGS='en',QIB_LANGS='en',
            BATCH_SIZE='1',THINK_MODE='0',QIB_METADATA_FILE_EN=str(data),LONGTEXTBENCH_METADATA_FILE_EN=str(data))
        env.update(TEXTBENCH_ROOT=str(dataset),TEXTCRAFTER_ROOT=str(evaluator),
                   PADDLEOCR_SOURCE_DIR=str(fixture/'missing-paddleocr'),HF_CACHE_DIR=str(fixture/'cache'),
                   RISE_INPUT_ROOT=str(dataset))
        # Retired variables and missing files are intentional poison values:
        # the new fixed activation scheme must ignore them entirely.
        for key in ('GEN_ENV','GENEVAL_EVAL_ENV','DPGBENCH_EVAL_ENV','QWENIMAGEBENCH_EVAL_ENV','QIB_API_ENV',
                    'TIIFBENCH_EVAL_ENV','LONGTEXTBENCH_EVAL_ENV','CVTG_2K_EVAL_ENV','ONEIGBENCH_EVAL_ENV',
                    'WISEBENCH_EVAL_ENV','BIZGENEVAL_EVAL_ENV','IMGEDIT_EVAL_ENV','GEDITBENCH_EVAL_ENV','RISE_EVAL_ENV',
                    'BAGEL_ENV','EVAL_ENV','CONDA_SH','UMM_ENV_CONFIG'):
            env[key]=str(fixture/'missing-retired-environment'/key)
        for path in ('gen/t2i/geneval/eval','gen/t2i/dpgbench/eval','gen/t2i/qwenimagebench/eval',
                     'gen/it2i/geditbench/eval/Step1X-Edit/GEdit-Bench','gen/it2i/rise/eval/judge'):
            (repo/'third_party/UMM_Eval'/path).mkdir(parents=True,exist_ok=True)
        def run(wrapper,settings):
            log.unlink(missing_ok=True)
            p=subprocess.run(['/bin/bash',str(repo/'scripts/eval'/wrapper)],env=dict(env,**settings),capture_output=True,text=True)
            if p.returncode:
                raise AssertionError((wrapper,settings,p.returncode,p.stdout[-1200:],p.stderr[-1200:]))
            calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            for call in calls:
                for filename,allowed in generator_flags.items():
                    if any(Path(arg).name==filename for arg in call['args']):
                        unknown={arg.split('=',1)[0] for arg in call['args'] if arg.startswith('--')}-allowed
                        assert not unknown,('unified generator command contains unsupported flags',filename,unknown,call)
            roots={call['env'].get('DG2_RESULT_DIR') for call in calls if call['env'].get('DG2_RESULT_DIR')}
            if len(roots)!=1:
                raise AssertionError((wrapper,'expected one managed leaf',calls,p.stdout))
            return Path(roots.pop()),calls
        def actual_scoring_calls(bench,calls):
            # Some wrappers delegate the real scoring stage. Replay the exact
            # environment captured at that boundary, with the same offline
            # interceptors, so activation is checked where scoring occurs.
            delegated={'longtextbench','cvtg_2k','oneigbench','wise','bizgeneval','imgedit','geditbench','rise'}
            if bench in delegated:
                callbacks=[call for call in calls if call['tool']=='bash' and
                           call['args'] and Path(call['args'][0]).name=='launch.sh']
                assert callbacks,(bench,'missing lower scoring delegation',calls)
                for callback in callbacks:
                    forwarded=callback['forwarded_env']
                    assert forwarded['RUN_GENERATION']=='0' and forwarded['RUN_EVAL']=='1',(
                        bench,'lower scorer received wrong stage flags',callback)
                    if bench in {'geditbench','rise'}:
                        # The fake generator cannot produce an image. The
                        # launcher's input guard only needs a named fixture;
                        # Python scorers remain intercepted and never open it.
                        images=Path(forwarded['OUTPUT_PATH']);images.mkdir(parents=True,exist_ok=True)
                        (images/'offline-sample.png').write_bytes(b'offline image placeholder')
                    log.unlink(missing_ok=True)
                    p=subprocess.run(['/bin/bash',*callback['args']],env=forwarded,
                                     capture_output=True,text=True,timeout=30)
                    assert p.returncode==0,(bench,'lower scoring control flow failed',p.stdout[-1200:],p.stderr[-1200:])
                    child_calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
                    calls=calls+child_calls
            activations=[call['name'] for call in calls if call['tool']=='activate']
            assert activations and set(activations)=={'geneval'},(bench,'eval-only activation mismatch',activations)
            scorers=[call for call in calls if call['tool'] in {'python','python3','torchrun','accelerate','env'}
                     and not any(Path(arg).name=='coverage_audit.py' for arg in call['args'])]
            assert scorers,(bench,'no actual scoring invocation captured',calls)
            for call in scorers:
                assert call['env']['DG2_STUB_ACTIVE_ENV']=='geneval',(bench,'scorer used wrong active environment',call)
                assert call['env']['PATH'].split(':')[0]=='/opt/conda/envs/geneval/bin',(bench,'missing fixed geneval PATH',call)
            return calls
        cases=[('run_t2i_accelerate.sh',bench) for bench in ('geneval','dpgbench','longtextbench','cvtg_2k','tiifbench','oneigbench','wise','bizgeneval','qwenimagebench')]
        cases += [('run_imgedit_accelerate.sh','imgedit'),('run_gedit_accelerate.sh','geditbench'),('run_rise_accelerate.sh','rise'),('run_edit_compass_accelerate.sh','editcompass')]
        all_roots=[]
        benchmark_roots={}
        for wrapper,bench in cases:
            settings={'BENCHMARK':bench,'GENEVAL_PROMPT_MODE':'short','RUN_GENERATION':'1','RUN_EVAL':'0'}
            if bench in ('cvtg_2k','oneigbench'):
                settings['METADATA_FILE']=str(dataset)
            gen_root,gen=run(wrapper,settings)
            eval_root,judge=run(wrapper,dict(settings,RUN_GENERATION='0',RUN_EVAL='1'))
            gen_activations=[call['name'] for call in gen if call['tool']=='activate']
            assert gen_activations and set(gen_activations)=={'bagel'},(bench,'generation activation mismatch',gen_activations)
            generation_calls=[call for call in gen if '--model_path' in call['args'] or '--model-path' in call['args']]
            assert generation_calls,(bench,'no generation invocation captured')
            for call in generation_calls:
                if bench!='editcompass':
                    generator='gen_images_mp_it2i.py' if bench in {'imgedit','geditbench','rise'} else 'gen_images_mp_t2i.py'
                    assert any(Path(arg).name==generator for arg in call['args']),(
                        bench,'Visual-Evals unified generator was replaced',call)
                    assert call['args'][call['args'].index('--benchmark')+1]==bench,(
                        bench,'unified generator received wrong benchmark',call)
                    if bench=='wise':
                        assert call['args'][call['args'].index('--wise_subdir')+1]=='wise_raw',(
                            'WISE native output subdirectory was lost',call)
                assert call['env']['DG2_STUB_ACTIVE_ENV']=='bagel',(bench,'generator used wrong active environment',call)
                assert call['env']['PATH'].split(':')[0]=='/opt/conda/envs/bagel/bin',(bench,'missing fixed bagel PATH',call)
            judge=actual_scoring_calls(bench,judge)
            assert gen_root==eval_root,(bench,gen_root,eval_root)
            assert gen_root.name==bench,(bench,gen_root)
            assert gen_root.parent.parent.name=='offline-consistency',(bench,gen_root)
            assert gen_root.parent.parent.parent==Path(env['OUTPUT_BASE']).resolve(),(bench,gen_root)
            assert (gen_root/'experiment_manifest.json').is_file(),bench
            all_roots.append(str(gen_root))
            benchmark_roots[bench]=gen_root
            for call in gen+judge:
                for pathkey in ('--output_dir','--output-dir','--save_root','--save-root','--save_dir','--save-dir','--outfile','--res-path','--image-root-path','--results-root','--image-dir','--judge-root','--out-prefix'):
                    if pathkey in call['args']:
                        value=Path(call['args'][call['args'].index(pathkey)+1])
                        assert value==gen_root or gen_root in value.parents,(bench,pathkey,value,gen_root)
                if call['tool']=='bash':
                    forwarded=call['env'].get('RESULT_ROOT_DIR') or call['env'].get('RESULT_DIR')
                    assert forwarded==str(gen_root),(bench,'scorer delegation mismatch',call)
            if bench in ('imgedit','geditbench','rise','editcompass'):
                think_root,think=run(wrapper,dict(settings,THINK_MODE='1'))
                assert {call['name'] for call in think if call['tool']=='activate'}=={'bagel'},(bench,'thinking generation activation mismatch')
                assert think_root.name==bench+'_thinking',(bench,think_root)
                assert think_root.parent==gen_root.parent,(bench,'thinking must share setting parent',think_root,gen_root)
                assert any('--think' in call['args'] or '--use_think' in call['args'] for call in think),(bench,'missing actual thinking flag',think)
                think_eval,think_judge=run(wrapper,dict(settings,THINK_MODE='1',RUN_GENERATION='0',RUN_EVAL='1'))
                assert think_eval==think_root,(bench,'thinking eval mismatch')
                actual_scoring_calls(bench,think_judge)
        assert len(set(all_roots))==13,'benchmark outputs collided'
        assert benchmark_roots['dpgbench'].parent==benchmark_roots['oneigbench'].parent,(
            'same effective generation settings must group benchmark leaves together',
            benchmark_roots['dpgbench'],benchmark_roots['oneigbench'])
        imgedit_config=json.loads((benchmark_roots['imgedit']/'experiment_manifest.json').read_text())['config']
        gedit_config=json.loads((benchmark_roots['geditbench']/'experiment_manifest.json').read_text())['config']
        assert imgedit_config['CFG_INTERVAL']==[0,1] and imgedit_config['CROP_BBOX'] is True,imgedit_config
        assert gedit_config['CFG_INTERVAL']==[0.4,1] and gedit_config['CROP_BBOX'] is False,gedit_config
        assert benchmark_roots['imgedit'].parent!=benchmark_roots['geditbench'].parent,(
            'different real crop and CFG protocols must not share a generation setting',
            benchmark_roots['imgedit'],benchmark_roots['geditbench'])
        wise_settings={'BENCHMARK':'wise','RUN_GENERATION':'1','RUN_EVAL':'0',
                       'EXPERIMENT_NAME':'offline-wise-custom','WISE_SUBDIR':'wise_custom',
                       'WISE_CATEGORIES':'natural_science','NUM_STEPS':'70','TIMESTEP_SHIFT':'2.0'}
        wise_custom,wise_calls=run('run_t2i_accelerate.sh',wise_settings)
        wise_generators=[call for call in wise_calls if '--benchmark' in call['args']]
        assert wise_generators,('WISE custom generation missing',wise_calls)
        for call in wise_generators:
            assert call['args'][call['args'].index('--wise_subdir')+1]=='wise_custom',call
            assert call['args'][call['args'].index('--num_timesteps')+1]=='70',call
            assert float(call['args'][call['args'].index('--timestep_shift')+1])==2.0,call
        wise_eval,wise_judge=run('run_t2i_accelerate.sh',dict(wise_settings,RUN_GENERATION='0',RUN_EVAL='1'))
        assert wise_eval==wise_custom,'WISE custom scoring root diverged from generation'
        actual_scoring_calls('wise',wise_judge)
        wise_parallel,_=run('run_t2i_accelerate.sh',dict(wise_settings,GPU_LIST='0,1'))
        assert wise_parallel==wise_custom,'stable per-index WISE seeds must not split cache by GPU count'
        # Exercise lower launchers themselves with both stages disabled. This
        # catches the original eager GEN_ENV activation before any stage guard,
        # even though expensive nested scorers remain intercepted above.
        for launcher in sorted((repo/'third_party/UMM_Eval').glob('gen/*/*/launch.sh')):
            log.unlink(missing_ok=True)
            low_root=root/'disabled-launchers'/launcher.parent.name
            p=subprocess.run(['/bin/bash',str(launcher)],env=dict(env,RUN_GENERATION='0',RUN_EVAL='0',
                OUTPUT_FOLDER=str(low_root),OUTPUT_PATH=str(low_root),RESULT_ROOT_DIR=str(low_root),
                DATA_PATH=str(data),BENCHMARK_ROOT=str(dataset),ONEIG_BENCH_ROOT=str(dataset),
                TEXTCRAFTER_ROOT=str(evaluator),TEXTBENCH_ROOT=str(dataset)),
                capture_output=True,text=True,timeout=30)
            assert p.returncode==0,(str(launcher.relative_to(repo)),'disabled launcher failed',p.stdout[-600:],p.stderr[-600:])
            disabled_calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            assert not any(call['tool']=='activate' for call in disabled_calls),(str(launcher.relative_to(repo)),'eager environment activation')
            assert not any(call['tool'] in {'torchrun','accelerate','env'} for call in disabled_calls),(str(launcher.relative_to(repo)),'disabled launcher invoked a scorer/generator')
        # Changed data/scope/judge refuse old leaves without invoking generators or scorers.
        dpg_manifest=benchmark_roots['dpgbench']/'experiment_manifest.json'
        old_manifest=dpg_manifest.read_bytes()
        original_metadata=data.read_bytes()
        data.write_text(json.dumps({'changed':{'id':'input.png','prompt':'edit','edit_type':'add'}}))
        log.unlink(missing_ok=True)
        p=subprocess.run(['/bin/bash',str(repo/'scripts/eval/run_t2i_accelerate.sh')],
            env=dict(env,BENCHMARK='dpgbench',RUN_GENERATION='0',RUN_EVAL='1'),capture_output=True,text=True)
        assert p.returncode!=0 and 'new EXPERIMENT_NAME' in p.stderr,('changed metadata reused cache',p.stderr)
        assert dpg_manifest.read_bytes()==old_manifest,'rejected resume replaced original manifest'
        assert not log.exists() or not log.read_text().strip(),'rejected resume invoked scorer'
        data.write_bytes(original_metadata)
        for bench,changed_scope in [('dpgbench',{'TARGET_PROMPT_IDS':'sample'}),
                                  ('qwenimagebench',{'QJUDGER_MAX_NEW_TOKENS':'8192'})]:
            manifest=benchmark_roots[bench]/'experiment_manifest.json';before=manifest.read_bytes()
            log.unlink(missing_ok=True)
            p=subprocess.run(['/bin/bash',str(repo/'scripts/eval/run_t2i_accelerate.sh')],
                env=dict(env,BENCHMARK=bench,RUN_GENERATION='0',RUN_EVAL='1',**changed_scope),capture_output=True,text=True)
            assert p.returncode!=0 and 'new EXPERIMENT_NAME' in p.stderr,(bench,'changed scope/judge reused cache',p.stderr)
            assert manifest.read_bytes()==before,(bench,'rejected resume replaced manifest')
            assert not log.exists() or not log.read_text().strip(),(bench,'rejected resume invoked scorer')
        for wrapper,bench in cases[-4:-1]:
            manifest=benchmark_roots[bench]/'experiment_manifest.json';before=manifest.read_bytes()
            log.unlink(missing_ok=True)
            p=subprocess.run(['/bin/bash',str(repo/'scripts/eval'/wrapper)],
                env=dict(env,BENCHMARK=bench,RUN_GENERATION='0',RUN_EVAL='1',TARGET_INDICES='0'),
                capture_output=True,text=True,timeout=30)
            assert p.returncode!=0 and 'new EXPERIMENT_NAME' in p.stderr,(bench,'changed editing subset reused cache',p.stderr)
            assert manifest.read_bytes()==before,(bench,'subset rejection replaced manifest')
            assert not log.exists() or not log.read_text().strip(),(bench,'subset rejection invoked scorer')
            selected_root,selected=run(wrapper,{'BENCHMARK':bench,'RUN_GENERATION':'1','RUN_EVAL':'0',
                'EXPERIMENT_NAME':'offline-target-'+bench,'TARGET_INDICES':'0','MAX_MEMORY':'17GiB'})
            invocations=[call for call in selected if '--model-path' in call['args'] or '--model_path' in call['args']]
            assert invocations,(bench,'selected generation was not invoked')
            for call in invocations:
                assert call['args'][call['args'].index('--target_indices')+1]=='0',(bench,'editing subset was not forwarded',call)
                assert call['args'][call['args'].index('--max_memory')+1]=='17GiB',(bench,'memory override was not forwarded',call)
            selected_config=json.loads((selected_root/'experiment_manifest.json').read_text())['config']
            assert selected_config['TARGET_INDICES']=='0',(bench,'selected scope missing from manifest')
        short,calls=run('run_t2i_accelerate.sh',{'BENCHMARK':'geneval','GENEVAL_PROMPT_MODE':'short','RUN_GENERATION':'1','RUN_EVAL':'0'})
        long,_=run('run_t2i_accelerate.sh',{'BENCHMARK':'geneval','GENEVAL_PROMPT_MODE':'long','RUN_GENERATION':'1','RUN_EVAL':'0'})
        assert short!=long,'GenEval variants collided'
        for bench in ('geneval','dpgbench','qwenimagebench','wise'):
            p=subprocess.run(['/bin/bash',str(repo/'scripts/eval/run_t2i_accelerate.sh')],env=dict(env,BENCHMARK=bench,THINK_MODE='1',RUN_GENERATION='0',RUN_EVAL='0'),capture_output=True,text=True)
            assert p.returncode!=0 and 'unsupported' in p.stderr,(bench,'fake thinking accepted',p.stderr)
        # The top-level both mode must dispatch the two actual metadata protocols.
        log.unlink(missing_ok=True)
        p=subprocess.run(['/bin/bash',str(repo/'eval_t2i_go.sh')],env=dict(env,TASK='geneval',GENEVAL_PROMPT_MODE='both',RUN_GENERATION='0',RUN_EVAL='0',METADATA_FILE=''),capture_output=True,text=True)
        assert p.returncode==0,(p.stdout,p.stderr)
        calls=[json.loads(line) for line in log.read_text().splitlines()]
        assert [call['env']['GENEVAL_PROMPT_MODE'] for call in calls if call['tool']=='bash']==['short','long'],calls
        assert not any(call['tool']=='activate' for call in calls),'top-level T2I entry activated an environment before a stage'
        log.unlink(missing_ok=True)
        p=subprocess.run(['/bin/bash',str(repo/'eval_go.sh')],env=dict(env,TASK='all',RUN_GENERATION='0',RUN_EVAL='0'),capture_output=True,text=True,timeout=30)
        assert p.returncode==0,('editing top-level dispatch failed',p.stdout[-600:],p.stderr[-600:])
        calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert len([call for call in calls if call['tool']=='bash'])==4,('editing top-level dispatch must visit all four wrappers',calls)
        assert not any(call['tool']=='activate' for call in calls),'top-level editing entry activated an environment before a stage'
        log.unlink(missing_ok=True)
        p=subprocess.run(['/bin/bash',str(repo/'eval_go.sh')],
            env=dict(env,TASK='rise',RUN_GENERATION='0',RUN_EVAL='0',TARGET_INDICES='0',MAX_MEMORY='17GiB'),
            capture_output=True,text=True,timeout=30)
        assert p.returncode==0,('editing top-level selected dispatch failed',p.stdout[-600:],p.stderr[-600:])
        calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        delegated=[call for call in calls if call['tool']=='bash']
        assert len(delegated)==1 and Path(delegated[0]['args'][0]).name=='run_rise_accelerate.sh',calls
        assert delegated[0]['forwarded_env']['TARGET_INDICES']=='0' and delegated[0]['forwarded_env']['MAX_MEMORY']=='17GiB',(
            'Visual top-level RISE lost subset/memory forwarding',delegated[0])
        changed,_=run('run_t2i_accelerate.sh',{'BENCHMARK':'geneval','GENEVAL_PROMPT_MODE':'short','RUN_GENERATION':'1','RUN_EVAL':'0','NUM_STEPS':'100'})
        assert changed!=short,'steps change reused generation cache root'
        print('PASS: 13 benchmark generation/scoring roots match; generation activates bagel and every actual scoring stage activates geneval with fixed PATH; delegated lower scorers execute offline; disabled/top-level stages never activate; retired environments/config are ignored; equal settings group benchmark leaves; changed data/scope/judge refuse old cache; thinking and GenEval prompt variants remain isolated.')


class WrapperLayoutTests(unittest.TestCase):
    def test_all_active_wrappers_route_one_managed_experiment_leaf(self):
        check_wrapper_paths()


if __name__ == "__main__":
    unittest.main()
