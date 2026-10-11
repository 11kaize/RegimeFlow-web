'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const config=JSON.parse(fs.readFileSync(path.join(__dirname,'../web/models/inference-config.json')));
function setup(fetcher) {
  const warnings=[];
  const sandbox={console:{warn:(...args)=>warnings.push(args),error:()=>{},log:()=>{}},fetch:fetcher,
    location:{protocol:'http:'}, document:{getElementById:()=>null,querySelector:()=>null,querySelectorAll:()=>[]},
    setTimeout:fn=>fn()};
  sandbox.window=sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/js/prediction.js'),'utf8'),sandbox);
  return {s:sandbox,warnings};
}
function response(value) {return {ok:true,json:async()=>value,text:async()=>value};}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

test('per-species conditions and all fallback cases stay in logs',()=>{
  const {s,warnings}=setup();
  s._pwState.conditions={A:{trajectory_type:3,period:20},B:{trajectory_type:5},C:{trajectory_type:3,period:0},D:{trajectory_type:8,period:2}};
  for (const [name,expected] of [['A',[3,20]],['B',[5,0]],['C',[0,0]],['D',[0,0]],['missing',[0,0]]]) {
    const value=s.conditionForSpecies({id:'ID'},name);
    assert.deepEqual([value.pattern,value.period],expected);
  }
  assert.equal(warnings.length,4);
});

test('backend rejects synthetic/malformed results and browser failure sends raw units plus bounds',async()=>{
  let body;
  const {s}=setup(async(url,args)=>{body=JSON.parse(args.body);return response({model:'regimeflow',predictions:[1,2,3]});});
  s.RegimeFlowWeb={configReady:async()=>config,ready:async()=>true,predict:async()=>{throw Error('failed');}};
  const result=await s.getPrediction([1,2],3,20,2,[-1,1,2,4]);
  assert.deepEqual(Array.from(result),[1,2]);
  assert.deepEqual(body,{context:[1,2],prediction_length:2,traj_pattern:3,period:20,bounds:[0,4]});
  s.fetch=async()=>response({model:'fallback-synthetic',predictions:[1,2]});
  await assert.rejects(s.backendPredict([1,2],0,0,2));
});

test('multi-species overlay uses each condition and stops stale results',async()=>{
  const {s}=setup();const calls=[];let resolveFirst;let draws=0;
  Object.assign(s._pwState,{currentModel:{id:'ID',name:'Model'},spNames:['A','B'],ctxLen:2,
    fullColumns:{A:[1,2,3,4],B:[2,3,4,5]},fullTime:[0,1,2,3],conditions:{A:{trajectory_type:3,period:20},B:{trajectory_type:5,period:0}}});
  s.getDisplayName=()=> 'Model';s.drawAllSpeciesChart=()=>draws++;s.setPredicting=()=>{};
  s.getPrediction=(ctx,pat,per)=>{calls.push([pat,per]);return new Promise(resolve=>{resolveFirst=resolve;});};
  s.renderAllSpecies();
  assert.deepEqual(calls,[[3,20]]);
  s._predictionId++;
  resolveFirst([1,2]);await flush();
  assert.equal(draws,1);assert.equal(calls.length,1);
  s.getPrediction=async(ctx,pat,per)=>{calls.push([pat,per]);return [1,2];};
  s.renderAllSpecies();await flush();await flush();
  assert.deepEqual(calls.slice(-2),[[3,20],[5,0]]);
});

test('switching systems ignores an older CSV/condition response',async()=>{
  const deferred={};
  const {s}=setup(url=>new Promise(resolve=>{deferred[url]=resolve;}));
  s.getDisplayName=m=>m.name;s.updateBackButton=()=>{};s.ChartError={clear:()=>{}};
  s.ChartSkeleton={show:()=>{},ensureMin:()=>{}};s.markModelOk=()=>{};
  s.loadPathwayTrajectory({id:'old',name:'Old'});s.loadPathwayTrajectory({id:'new',name:'New'});
  deferred['/api/csv/new/New'](response('time,X\n0,2\n1,3\n2,4\n3,5\n'));
  deferred['/api/conditions/new/New'](response({X:{trajectory_type:5,period:0}}));await flush();
  deferred['/api/csv/old/Old'](response('time,X\n0,100\n1,200\n2,300\n3,400\n'));
  deferred['/api/conditions/old/Old'](response({X:{trajectory_type:3,period:12}}));await flush();
  assert.deepEqual(Array.from(s._pwState.fullColumns.X),[2,3,4,5]);
  assert.equal(s._pwState.conditions.X.trajectory_type,5);
});

test('existing species selector callback still works after initialization',()=>{
  const {s}=setup(async()=>response({}));
  function element() {return {handlers:{},style:{},dataset:{dim:'taxonomy'},classList:{add(){},remove(){}},addEventListener(name,fn){this.handlers[name]=fn;},appendChild(){}};}
  const elements=Object.fromEntries(['category-filter','pw-model-list','species-select'].map(key=>[key,element()]));
  const tab=element();
  s.document.getElementById=key=>elements[key]||null;
  s.document.querySelectorAll=key=>key==='.dim-tab'?[tab]:[];
  s.document.createElement=element;
  Object.assign(s,{BIO_MODELS_DATA:[],BIO_DOMAIN_DATA:{},BIO_PROCESSES_DATA:{},
    currentFilteredModels:()=>[],_predictChart:{},renderCategoryComparison:()=>{},updateSidebarNote:()=>{},
    addEventListener:()=>{},debounce:fn=>fn});
  let redraws=0;s.redrawPathwayChart=()=>redraws++;
  s.initPrediction();elements['species-select'].value='2';elements['species-select'].handlers.change();
  assert.equal(s._pwState.spIdx,2);assert.equal(redraws,1);
});

test('prediction before delayed browser warm-up can use the real backend',async()=>{
  const {s}=setup(async(url)=>url.includes('inference-config')?response(config):response({model:'regimeflow',predictions:[2,3]}));
  s.RegimeFlowWeb={configReady:async()=>{throw Error('not started');},ready:async()=>false,getError:()=>null};
  const values=await s.getPrediction([1,2,3,4],0,0,2,[1,2,3,4,5]);
  assert.deepEqual(Array.from(values),[2,3]);
});
