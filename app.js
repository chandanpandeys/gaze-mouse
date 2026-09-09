import{clamp,sleep,median,dp,ridge}from'./math.js';
import{FaceTracker}from'./tracker.js';

const $=id=>document.getElementById(id);
const video=$('video'),status=$('status'),diag=$('diag'),cam=$('cam'),calBtn=$('calBtn'),refineBtn=$('refineBtn'),blinkCalBtn=$('blinkCalBtn'),controlBtn=$('controlBtn'),scrollBtn=$('scrollBtn'),assist=$('assist'),cursor=$('cursor'),cal=$('cal'),dotEl=$('dot'),copyBox=$('copyBox'),ct=$('ct'),ch=$('ch'),rest=$('rest'),trainEyebrow=$('trainEyebrow'),trainActions=$('trainActions'),trainReady=$('trainReady'),trainBack=$('trainBack'),trainExit=$('trainExit'),trainRestart=$('trainRestart'),captureStop=$('captureStop'),countdown=$('countdown'),calProgress=$('calProgress'),toast=$('toast');
const tracker=new FaceTracker();

let running=false,calibrated=false,latest=null,latestAt=0,lastInfer=0,frames=0,fpsAt=performance.now(),toastTimer=null;
let modelX=null,modelY=null,trainingSamples=[],residualAnchors=[],mapNorm=null;
let sx=innerWidth/2,sy=innerHeight/2,lastTargetX=sx,lastTargetY=sy,lastStablePoint={x:sx,y:sy,at:0,velocity:999},focusEl=null,clicks=0;
let controlState='off',frozenClickPoint=null,leftWinkAt=0,leftWinkLatched=false,rightWinkAt=0,rightWinkLatched=false;
let bothClosed=false,blinkStart=0,blinkSymSum=0,blinkSymN=0,previousBlink=null,blinkProfile=null,blinkCalActive=false,blinkTrialResolve=null;
let scrollZone='center',scrollZoneSince=0,lastScrollAt=0;

let trainingDraft=null,trainingKind=null,guideAction=null,captureActive=false,captureAbort=false,restartArmUntil=0,historyGuard=false,blinkAbort=false;

const CAL_KEY='gazeMousePoseMap-v1',BLINK_KEY='gazeMouseBlink-v1',DRAFT_KEY='gazeMouseTrainingDraft-v2';
const POSE_POINTS=[
  [.09,.10],[.50,.10],[.91,.10],
  [.09,.50],[.50,.50],[.91,.50],
  [.09,.90],[.50,.90],[.91,.90]
];
const REFINE_POINTS=[];for(const y of [.10,.36,.64,.90])for(const x of [.08,.29,.50,.71,.92])REFINE_POINTS.push([x,y]);
const POSE_STAGES=[
  {name:'Steady',instruction:'Keep the phone still and look only at the tiny center dot. Relax your face; do not force your eyes open.',ms:1400},
  {name:'Left ↔ right',instruction:'Keep looking at the same dot. Move the PHONE slowly left → center → right → center. Your eyes stay on the dot.',ms:3200},
  {name:'Up ↕ down',instruction:'Keep looking at the same dot. Move the PHONE slowly upward → center → downward → center. Do not chase the phone with your eyes.',ms:3400},
  {name:'Near ↔ far',instruction:'Keep looking at the same dot. Move the PHONE slowly a little closer → normal → a little farther → normal.',ms:3200},
  {name:'Angle / tilt',instruction:'Keep looking at the same dot. Gently tilt/rotate the PHONE through a few comfortable angles, then return to normal.',ms:3000}
];
const MAX_SAVED_SAMPLES=1900;

function st(t,c=''){status.textContent=t;status.className='pill'+(c?' '+c:'')}
function msg(t,ms=2400){toast.textContent=t;toast.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>toast.classList.remove('show'),ms)}
function clearFocus(){if(focusEl){focusEl.classList.remove('focus');focusEl=null}}
function setControlState(next,announce=true){
  controlState=next;const off=next==='off',scroll=next==='scroll';
  $('controlTag').textContent='Control: '+(off?'OFF':scroll?'SCROLL':'POINTER');
  controlBtn.textContent=off?'Eye control OFF':'Eye control ON';
  scrollBtn.textContent=scroll?'Scroll mode':'Pointer mode';
  scrollBtn.disabled=off||!calibrated;
  if(off){cursor.style.display='none';clearFocus();scrollZone='center'}
  if(announce)msg(off?'Eye control off':scroll?'Scroll mode':'Pointer mode',1200);
}
function toggleControl(){if(!calibrated)return msg('Train the pose map first.');setControlState(controlState==='off'?'pointer':'off')}
function toggleScroll(){if(controlState!=='off')setControlState(controlState==='scroll'?'pointer':'scroll')}
function calibrationFrameOK(){return latest&&performance.now()-latestAt<150&&latest.blink.l<.82&&latest.blink.r<.82}

function snapTarget(x,y,velocity){
  if(!assist.checked||velocity>13)return{x,y,el:null};
  const els=[...document.querySelectorAll('.letter,button:not(:disabled),select')];let best=null,bd=27;
  for(const el of els){const r=el.getBoundingClientRect();if(!r.width||!r.height)continue;const cx=r.left+r.width/2,cy=r.top+r.height/2,d=Math.hypot(x-cx,y-cy);if(d<bd){bd=d;best={x:cx,y:cy,el}}}
  return best||{x,y,el:null};
}
function smoothCursor(tx,ty){
  const dx=tx-sx,dy=ty-sy,d=Math.hypot(dx,dy);let alpha=d<12?.045:d<38?.075:d<100?.12:d<220?.18:.23;
  let stepx=dx*alpha,stepy=dy*alpha;const step=Math.hypot(stepx,stepy),maxStep=42;
  if(step>maxStep){const k=maxStep/step;stepx*=k;stepy*=k}
  sx+=stepx;sy+=stepy;return{x:sx,y:sy,velocity:Math.hypot(sx-lastTargetX,sy-lastTargetY)};
}

function robustStats(v){const med=median(v),mad=median(v.map(x=>Math.abs(x-med)))||1;return{med,mad}}
function buildBlinkProfile(t){return{d1:robustStats(t.map(x=>x.d1)),gap:robustStats(t.map(x=>x.gap)),d2:robustStats(t.map(x=>x.d2)),total:robustStats(t.map(x=>x.total)),sym:robustStats(t.map(x=>x.sym))}}
function inProfile(v,s,min,max){return Math.abs(v-s.med)<=clamp(2.6*s.mad,min,max)}
function matchesBlinkProfile(c){return!!blinkProfile&&inProfile(c.d1,blinkProfile.d1,45,150)&&inProfile(c.gap,blinkProfile.gap,45,180)&&inProfile(c.d2,blinkProfile.d2,45,150)&&inProfile(c.total,blinkProfile.total,90,260)&&c.total<900&&c.gap>35}
function clickAt(x,y){
  clicks++;$('clicks').textContent=clicks;cursor.style.display='none';const e=document.elementFromPoint(x,y);if(controlState!=='off')cursor.style.display='block';
  const target=e?.closest('button,select,a,[role="button"],input');if(target){if(target.tagName==='SELECT')target.focus();else target.click()}
}
function handleBlinkEvent(ev){
  if(previousBlink){const gap=ev.start-previousBlink.end;if(gap>35&&gap<850){
    const c={d1:previousBlink.dur,gap,d2:ev.dur,total:ev.end-previousBlink.start,sym:(previousBlink.sym+ev.sym)/2};
    if(blinkCalActive&&blinkTrialResolve){const r=blinkTrialResolve;blinkTrialResolve=null;previousBlink=null;r(c);return}
    if(controlState==='pointer'&&calibrated&&matchesBlinkProfile(c)&&frozenClickPoint&&frozenClickPoint.velocity<18&&performance.now()-frozenClickPoint.at<1300){clickAt(frozenClickPoint.x,frozenClickPoint.y);msg('Click',600);previousBlink=null;frozenClickPoint=null;return}
  }}previousBlink=ev;
}
function processGestures(now){
  const b=latest.blink,closed=b.l>.62&&b.r>.62,leftOnly=b.l>.76&&b.r<.34,rightOnly=b.r>.76&&b.l<.34;
  if(previousBlink&&now-previousBlink.end>950){previousBlink=null;frozenClickPoint=null}
  if(closed&&!bothClosed){blinkStart=now;blinkSymSum=0;blinkSymN=0;if(!previousBlink||now-previousBlink.end>850)frozenClickPoint={...lastStablePoint}}
  if(closed){blinkSymSum+=Math.abs(b.l-b.r);blinkSymN++}
  if(!closed&&bothClosed){const dur=now-blinkStart;if(dur>45&&dur<520)handleBlinkEvent({start:blinkStart,end:now,dur,sym:blinkSymN?blinkSymSum/blinkSymN:0})}
  bothClosed=closed;if(cal.classList.contains('show'))return;
  if(leftOnly){if(!leftWinkAt)leftWinkAt=now;if(!leftWinkLatched&&now-leftWinkAt>520){leftWinkLatched=true;toggleControl()}}else{leftWinkAt=0;leftWinkLatched=false}
  if(rightOnly){if(!rightWinkAt)rightWinkAt=now;if(!rightWinkLatched&&now-rightWinkAt>520&&controlState!=='off'){rightWinkLatched=true;toggleScroll()}}else{rightWinkAt=0;rightWinkLatched=false}
}
function handleScroll(y,now){
  const yn=y/innerHeight,zone=yn<.30?'up':yn>.70?'down':'center';
  if(zone!==scrollZone){scrollZone=zone;scrollZoneSince=now;return}
  if(zone==='center'||now-scrollZoneSince<420||now-lastScrollAt<55)return;
  lastScrollAt=now;const depth=zone==='up'?clamp((.30-yn)/.30,0,1):clamp((yn-.70)/.30,0,1),dy=(zone==='up'?-1:1)*(3+15*depth);window.scrollBy(0,dy);
}

function buildMapNorm(samples){
  const n=samples[0].mapF.length,mean=Array(n).fill(0),sd=Array(n).fill(0);
  for(const s of samples)for(let i=0;i<n;i++)mean[i]+=s.mapF[i];for(let i=0;i<n;i++)mean[i]/=samples.length;
  for(const s of samples)for(let i=0;i<n;i++){const d=s.mapF[i]-mean[i];sd[i]+=d*d}for(let i=0;i<n;i++)sd[i]=Math.sqrt(sd[i]/samples.length)||1;
  return{mean,sd};
}
function normMap(f){return f.map((v,i)=>(v-mapNorm.mean[i])/mapNorm.sd[i])}
function mapDistance(a,b){let s=0;for(let i=0;i<a.length;i++){let w=1;if(i<10)w=1.35;else if(i<14)w=.9;else w=.75;const d=(a[i]-b[i])*w;s+=d*d}return Math.sqrt(s/a.length)}
function downsample(samples,max=MAX_SAVED_SAMPLES){if(samples.length<=max)return samples;const out=[];for(let i=0;i<max;i++)out.push(samples[Math.floor(i*(samples.length-1)/(max-1))]);return out}
function rebuildModel(){
  if(trainingSamples.length<120)throw new Error('not enough training samples');
  modelX=ridge(trainingSamples,'x','xF',.016);modelY=ridge(trainingSamples,'y','yF',.018);mapNorm=buildMapNorm(trainingSamples);
  const stride=Math.max(1,Math.floor(trainingSamples.length/650));residualAnchors=[];
  for(let i=0;i<trainingSamples.length;i+=stride){const s=trainingSamples[i],gx=dp(s.xF,modelX),gy=dp(s.yF,modelY);residualAnchors.push({nf:normMap(s.mapF),rx:s.x-gx,ry:s.y-gy})}
  const errors=[];for(let i=0;i<trainingSamples.length;i+=13){const s=trainingSamples[i],p=predictPoint(s);errors.push(Math.hypot((p.x-s.x)*innerWidth,(p.y-s.y)*innerHeight))}
  return Math.round(median(errors));
}
function predictPoint(sampleLike){
  let gx=dp(sampleLike.xF,modelX),gy=dp(sampleLike.yF,modelY);if(!mapNorm||!residualAnchors.length)return{x:gx,y:gy,confidence:0};
  const nf=normMap(sampleLike.mapF),nearest=residualAnchors.map(a=>({a,d:mapDistance(nf,a.nf)})).sort((u,v)=>u.d-v.d).slice(0,24);
  let rx=0,ry=0,w=0;for(const q of nearest){const wt=1/Math.pow(q.d+.18,2);rx+=q.a.rx*wt;ry+=q.a.ry*wt;w+=wt}
  const d0=nearest[0]?.d??9,confidence=clamp(1-d0/2.5,0,1),blend=.62*confidence;
  if(w){gx+=clamp(rx/w,-.16,.16)*blend;gy+=clamp(ry/w,-.16,.16)*blend}
  return{x:gx,y:gy,confidence};
}
function updateCursor(now){
  const p=predictPoint(latest),tx=clamp(p.x,.004,.996)*innerWidth,ty=clamp(p.y,.004,.996)*innerHeight,sm=smoothCursor(tx,ty),sn=snapTarget(sm.x,sm.y,sm.velocity);
  if(sn.el!==focusEl){clearFocus();focusEl=sn.el;if(focusEl?.classList.contains('letter'))focusEl.classList.add('focus')}
  lastStablePoint={x:sn.x,y:sn.y,at:performance.now(),velocity:sm.velocity};$('stableTag').textContent=`Cursor ${Math.round(sm.velocity)}px · map ${Math.round(p.confidence*100)}%`;lastTargetX=sm.x;lastTargetY=sm.y;
  if(controlState==='off'){cursor.style.display='none';clearFocus();return}
  cursor.style.left=sn.x+'px';cursor.style.top=sn.y+'px';cursor.style.display='block';if(controlState==='scroll')handleScroll(sn.y,now);
}

function saveCalibration(){
  try{localStorage.setItem(CAL_KEY,JSON.stringify({ratio:innerWidth/innerHeight,modelX,modelY,mapNorm,residualAnchors,trainingSamples:downsample(trainingSamples)}))}catch(e){console.warn('save calibration',e)}
}
function restoreCalibration(){
  try{
    const d=JSON.parse(localStorage.getItem(CAL_KEY)||'null');if(!d||Math.abs(d.ratio-innerWidth/innerHeight)>.12)return false;
    modelX=d.modelX;modelY=d.modelY;mapNorm=d.mapNorm;residualAnchors=d.residualAnchors||[];trainingSamples=d.trainingSamples||[];
    if(modelX&&modelY&&mapNorm&&residualAnchors.length){calibrated=true;$('calState').textContent='Saved';$('sampleTag').textContent=`Pose map: restored · ${trainingSamples.length} samples`;controlBtn.disabled=false;refineBtn.disabled=false;setControlState('off',false);return true}
  }catch(e){console.warn('restore calibration',e)}return false;
}
function saveBlink(){try{if(blinkProfile)localStorage.setItem(BLINK_KEY,JSON.stringify(blinkProfile))}catch{}}
function restoreBlink(){try{blinkProfile=JSON.parse(localStorage.getItem(BLINK_KEY)||'null');if(blinkProfile)$('blinkTag').textContent=`Double blink: trained · ${Math.round(blinkProfile.total.med)} ms`}catch{}}

function getDraft(){try{return JSON.parse(localStorage.getItem(DRAFT_KEY)||'null')}catch{return null}}
function compatibleDraft(d=getDraft()){return d&&Math.abs((d.ratio||0)-innerWidth/innerHeight)<=.12?d:null}
function saveDraft(){if(!trainingDraft)return;trainingDraft.updatedAt=Date.now();try{localStorage.setItem(DRAFT_KEY,JSON.stringify(trainingDraft))}catch(e){console.warn('save draft',e);msg('Training progress could not be saved; storage may be full.',3500)}}
function clearDraft(){localStorage.removeItem(DRAFT_KEY);trainingDraft=null;updateResumeUI()}
function totalStages(mode){return mode==='pose'?POSE_POINTS.length*POSE_STAGES.length:mode==='refine'?REFINE_POINTS.length:0}
function doneStages(d){return d?Object.keys(d.completed||{}).length:0}
function updateResumeUI(){
  const d=getDraft();calBtn.textContent='3D pose-map train';refineBtn.textContent='Precision refine';
  if(d){const total=totalStages(d.mode),done=doneStages(d);if(Math.abs((d.ratio||0)-innerWidth/innerHeight)>.12){$('sampleTag').textContent=`Saved ${d.mode} training kept for another orientation`;return}
    if(d.mode==='pose'){calBtn.textContent=`Resume 3D training (${done}/${total})`;$('sampleTag').textContent=`Pose training saved · ${done}/${total} stages`}
    if(d.mode==='refine'){refineBtn.textContent=`Resume refine (${done}/${total})`;$('sampleTag').textContent=`Precision training saved · ${done}/${total} dots`}
  }
}

async function initTracker(){
  try{await tracker.init(t=>diag.textContent=t,n=>$('sourceTag').textContent='Tracker source: '+n);st('Tracker ready','ok');diag.textContent='Tracker ready. Start camera.';if(running)requestAnimationFrame(loop)}
  catch(e){console.error(e);st('Tracker failed','bad');diag.textContent='Tracker error: '+(e.message||e)}
}
async function startCamera(){
  try{await tracker.startCamera(video);running=true;cam.disabled=true;cam.textContent='Camera active';if(tracker.ready)requestAnimationFrame(loop)}
  catch(e){diag.textContent='Camera error: '+e.name;msg('Allow camera permission in Chrome.',4000)}
}
function loop(now){
  if(!running||!tracker.ready)return;requestAnimationFrame(loop);if(video.readyState<2||now-lastInfer<34)return;lastInfer=now;
  try{
    latest=tracker.detect(video,now);latestAt=performance.now();$('face').textContent=latest?'Yes':'No';
    if(latest){
      $('pose').textContent=Math.round(latest.pose.yaw)+'°/'+Math.round(latest.pose.pitch)+'°';st(calibrated?'Tracking':'Face locked','ok');
      calBtn.disabled=false;blinkCalBtn.disabled=false;controlBtn.disabled=!calibrated;refineBtn.disabled=!calibrated;
      diag.textContent=calibrated?'Pose-aware gaze map active. Actions only happen while Eye control is ON.':'Face found. Start or resume pose-map training.';
      processGestures(now);if(calibrated&&!cal.classList.contains('show')&&latest.blink.l<.56&&latest.blink.r<.56)updateCursor(now);
    }else{st('Find my face…');calBtn.disabled=true;blinkCalBtn.disabled=true;controlBtn.disabled=true;refineBtn.disabled=true;scrollBtn.disabled=true;cursor.style.display='none';clearFocus()}
  }catch(e){console.warn(e)}
  frames++;if(now-fpsAt>1000){$('fps').textContent=Math.round(frames*1000/(now-fpsAt));frames=0;fpsAt=now}
}

async function enterFullscreen(){try{if(!document.fullscreenElement)await document.documentElement.requestFullscreen()}catch{}}
function pushHistoryGuard(){if(historyGuard)return;history.pushState({gazeTraining:true},document.title);historyGuard=true}
function releaseHistoryGuard(){if(!historyGuard)return;historyGuard=false;try{history.back()}catch{}}
function setGuide(title,text,meta,readyText,onReady,canBack=true){
  captureActive=false;captureAbort=false;copyBox.style.display='block';trainActions.style.display='grid';dotEl.style.display='none';captureStop.style.display='none';countdown.style.display='none';
  trainEyebrow.textContent='NOT RECORDING';trainEyebrow.classList.remove('recording');ct.textContent=title;ch.textContent=text;rest.textContent=meta||'Nothing is recording. Read this first.';
  trainReady.textContent=readyText||'I’m ready';trainBack.disabled=!canBack;guideAction=onReady;trainRestart.textContent='Start over';restartArmUntil=0;
}
function setCaptureUI(x,y){
  copyBox.style.display='none';trainActions.style.display='none';captureStop.style.display='block';dotEl.style.display='block';dotEl.style.left=x*100+'%';dotEl.style.top=y*100+'%';
}
async function runCountdown(x,y){
  setCaptureUI(x,y);captureStop.style.display='block';countdown.style.display='block';
  for(const n of [3,2,1]){if(captureAbort)return false;countdown.textContent=n;await sleep(700)}
  countdown.textContent='';countdown.style.display='none';try{navigator.vibrate?.(35)}catch{}return !captureAbort;
}
function captureSample(x,y,weight=1){
  if(!calibrationFrameOK())return null;
  return{xF:[...latest.xF],yF:[...latest.yF],mapF:[...latest.mapF],x,y,weight,cx:latest.geometry.centerX,cy:latest.geometry.centerY,sc:latest.geometry.eyeDist,yaw:latest.pose.yaw,pitch:latest.pose.pitch};
}
function stageKey(p,s=0){return`${p}:${s}`}
function previousCursor(d){let p=d.pointIndex||0,s=d.stageIndex||0;if(d.mode==='pose'){if(s>0)s--;else if(p>0){p--;s=POSE_STAGES.length-1}else return null}else{if(p>0)p--;else return null;s=0}return{p,s}}
function advanceCursor(d){if(d.mode==='pose'){if(d.stageIndex<POSE_STAGES.length-1)d.stageIndex++;else{d.pointIndex++;d.stageIndex=0}}else d.pointIndex++}
function currentTrainingFinished(d){return d.mode==='pose'?d.pointIndex>=POSE_POINTS.length:d.pointIndex>=REFINE_POINTS.length}
function trainingBack(){
  if(captureActive){captureAbort=true;return}
  if(!trainingDraft)return;const prev=previousCursor(trainingDraft);if(!prev)return msg('You are already at the first stage.');
  delete trainingDraft.completed[stageKey(prev.p,prev.s)];trainingDraft.pointIndex=prev.p;trainingDraft.stageIndex=prev.s;saveDraft();updateResumeUI();
  if(trainingDraft.mode==='pose')showPoseStageGuide('Previous stage removed. Record it again when ready.');else showRefineGuide('Previous dot removed. Record it again when ready.');
}
function restartTraining(){
  const now=Date.now();if(now>restartArmUntil){restartArmUntil=now+3500;trainRestart.textContent='Tap again to erase draft';msg('Only unfinished training will be erased. Your last completed gaze map stays saved.',3000);return}
  const mode=trainingDraft?.mode||trainingKind;if(!mode)return;trainingDraft={version:2,mode,ratio:innerWidth/innerHeight,pointIndex:0,stageIndex:0,completed:{},createdAt:Date.now(),updatedAt:Date.now()};saveDraft();updateResumeUI();
  if(mode==='pose')showPoseIntro(true);else if(mode==='refine')showRefineIntro(true);
}
async function saveAndExitTraining(){
  captureAbort=true;blinkAbort=true;saveDraft();cal.classList.remove('show');guideAction=null;captureActive=false;captureStop.style.display='none';countdown.style.display='none';calProgress.style.width='0%';
  calBtn.disabled=!latest;refineBtn.disabled=!calibrated;blinkCalBtn.disabled=!latest;updateResumeUI();releaseHistoryGuard();
  try{if(document.fullscreenElement)await document.exitFullscreen()}catch{}
  msg(trainingDraft?'Training saved. You can resume later.':'Training closed.',2600);
}
function pauseTraining(reason='Training paused.'){captureAbort=true;saveDraft();if(trainingDraft?.mode==='pose')setGuide('Training paused',reason,'Completed stages are saved. The current unfinished stage was discarded.','Resume',()=>showPoseStageGuide(),doneStages(trainingDraft)>0);else if(trainingDraft?.mode==='refine')setGuide('Training paused',reason,'Completed dots are saved. The current unfinished dot was discarded.','Resume',()=>showRefineGuide(),doneStages(trainingDraft)>0)}

function makeDraft(mode){return{version:2,mode,ratio:innerWidth/innerHeight,pointIndex:0,stageIndex:0,completed:{},createdAt:Date.now(),updatedAt:Date.now()}}
function openTraining(mode){trainingKind=mode;setControlState('off',false);cursor.style.display='none';clearFocus();cal.classList.add('show');calProgress.style.width='0%';pushHistoryGuard();enterFullscreen()}
function startPoseTraining(){
  if(!latest)return msg('Start camera and keep your full face visible.');const raw=getDraft();
  if(raw&&Math.abs((raw.ratio||0)-innerWidth/innerHeight)>.12)return msg('Saved training belongs to another orientation. Rotate back to resume it.',3800);
  if(raw&&raw.mode!=='pose')return msg('You have unfinished precision training. Resume or finish that first.',3500);
  trainingDraft=raw||makeDraft('pose');saveDraft();openTraining('pose');showPoseIntro(false);
}
function showPoseIntro(restarted=false){
  const done=doneStages(trainingDraft),total=totalStages('pose');
  setGuide(restarted?'Fresh pose training ready':done?'Resume 3D pose training':'Before 3D pose training',
    done?`Your previous progress is safe: ${done} of ${total} short stages are already recorded.`:'There are 9 screen dots. For each dot you will do 5 short phone movements. You will NEVER be asked to read while recording.',
    'For every stage: read the instruction → tap I’m ready → 3-second countdown → focus only on the dot. After recording, the app pauses before showing the next instruction.',
    done?'Continue to next instruction':'Show first instruction',()=>showPoseStageGuide(),done>0);
  calProgress.style.width=(done/total*100).toFixed(1)+'%';
}
function showPoseStageGuide(note=''){
  if(currentTrainingFinished(trainingDraft))return finalizePoseTraining();
  const p=trainingDraft.pointIndex,s=trainingDraft.stageIndex,stage=POSE_STAGES[s],done=doneStages(trainingDraft),total=totalStages('pose');
  setGuide(`Dot ${p+1} of ${POSE_POINTS.length} · ${stage.name}`,stage.instruction,
    `${note?note+' ':''}Nothing is recording now. When you tap I’m ready, find the tiny dot during the 3-second countdown. Then perform only this one movement.`,
    'I’m ready',()=>capturePoseStage(),done>0);
  calProgress.style.width=(done/total*100).toFixed(1)+'%';
}
async function capturePoseStage(){
  const p=trainingDraft.pointIndex,s=trainingDraft.stageIndex,stage=POSE_STAGES[s],[x,y]=POSE_POINTS[p];captureAbort=false;
  if(!await runCountdown(x,y)){showPoseStageGuide('Capture cancelled. No data from that attempt was kept.');return}
  captureActive=true;const local=[];let last=0,start=performance.now();
  while(performance.now()-start<stage.ms&&!captureAbort){const now=performance.now();if(now-last>50){const q=captureSample(x,y,1);if(q){local.push(q);last=now}}const frac=(performance.now()-start)/stage.ms;calProgress.style.width=((doneStages(trainingDraft)+clamp(frac,0,1))/totalStages('pose')*100).toFixed(1)+'%';await sleep(18)}
  captureActive=false;captureStop.style.display='none';dotEl.style.display='none';try{navigator.vibrate?.([35,45,35])}catch{}
  if(captureAbort){showPoseStageGuide('That attempt was discarded. Your earlier stages are still saved.');return}
  if(local.length<12){showPoseStageGuide(`Only ${local.length} usable frames were visible, so this stage was NOT saved. Keep both eyes visible and retry.`);return}
  trainingDraft.completed[stageKey(p,s)]=downsample(local,58);advanceCursor(trainingDraft);saveDraft();updateResumeUI();
  if(currentTrainingFinished(trainingDraft))return finalizePoseTraining();
  showPoseStageGuide(`Saved ${local.length} usable frames from the last stage.`);
}
async function finalizePoseTraining(){
  const old={modelX,modelY,mapNorm,residualAnchors,trainingSamples,calibrated};
  setGuide('Building your gaze map','All training stages are saved. Now the phone is fitting the model.','Do not close the page for a moment.','Please wait',()=>{},false);trainReady.disabled=true;
  try{
    const samples=[];for(let p=0;p<POSE_POINTS.length;p++)for(let s=0;s<POSE_STAGES.length;s++)samples.push(...(trainingDraft.completed[stageKey(p,s)]||[]));
    if(samples.length<420)throw new Error('too few saved training samples');trainingSamples=downsample(samples);const med=rebuildModel();calibrated=true;$('calState').textContent='Yes';$('err').textContent=med+' px';sx=lastTargetX=innerWidth/2;sy=lastTargetY=innerHeight/2;lastStablePoint={x:sx,y:sy,at:0,velocity:999};saveCalibration();localStorage.removeItem(DRAFT_KEY);trainingDraft=null;updateResumeUI();
    $('sampleTag').textContent=`Pose map: trained · ${trainingSamples.length} samples`;diag.textContent=`Pose map trained. Fit ${med}px. Precision refine can add more exact screen locations.`;controlBtn.disabled=false;refineBtn.disabled=false;setControlState('off',false);await saveAndExitTraining();msg('Pose map complete. Nothing was lost.',4000);
  }catch(e){console.error(e);modelX=old.modelX;modelY=old.modelY;mapNorm=old.mapNorm;residualAnchors=old.residualAnchors;trainingSamples=old.trainingSamples;calibrated=old.calibrated;trainReady.disabled=false;saveDraft();setGuide('Could not finish the model','Your training data is still saved. Nothing was erased.',String(e.message||e),'Return to training',()=>showPoseStageGuide(),doneStages(trainingDraft)>0)}
}

function startRefineTraining(){
  if(!calibrated||!latest)return msg('Complete or restore the main pose map first.');const raw=getDraft();
  if(raw&&Math.abs((raw.ratio||0)-innerWidth/innerHeight)>.12)return msg('Saved training belongs to another orientation. Rotate back to resume it.',3800);
  if(raw&&raw.mode!=='refine')return msg('You have unfinished pose training. Resume or finish that first.',3500);
  trainingDraft=raw||makeDraft('refine');saveDraft();openTraining('refine');showRefineIntro(false);
}
function showRefineIntro(restarted=false){
  const done=doneStages(trainingDraft),total=totalStages('refine');
  setGuide(restarted?'Fresh precision training ready':done?'Resume precision refinement':'Before precision refinement',
    done?`${done} of ${total} precision dots are already safely recorded.`:'There are 20 fixed dots. For each one, read first, then the text disappears before recording. Keep the phone normally positioned and move only your eyes.',
    'Each dot is independent. Back removes only the previous dot. Save & exit keeps everything you have completed.',done?'Continue':'Show first dot instruction',()=>showRefineGuide(),done>0);
  calProgress.style.width=(done/total*100).toFixed(1)+'%';
}
function showRefineGuide(note=''){
  if(currentTrainingFinished(trainingDraft))return finalizeRefineTraining();const p=trainingDraft.pointIndex,done=doneStages(trainingDraft),total=totalStages('refine');
  setGuide(`Precision dot ${p+1} of ${REFINE_POINTS.length}`,'Keep the phone comfortably still. During recording, look exactly at the tiny center dot and keep your head natural.',`${note?note+' ':''}Nothing is recording now. Tap I’m ready; after the countdown the instructions disappear.`,'I’m ready',()=>captureRefinePoint(),done>0);calProgress.style.width=(done/total*100).toFixed(1)+'%';
}
async function captureRefinePoint(){
  const p=trainingDraft.pointIndex,[x,y]=REFINE_POINTS[p];captureAbort=false;if(!await runCountdown(x,y)){showRefineGuide('Capture cancelled. Nothing was lost.');return}
  captureActive=true;const local=[];let last=0,start=performance.now(),ms=1250;
  while(performance.now()-start<ms&&!captureAbort){const now=performance.now();if(now-last>48){const q=captureSample(x,y,1.35);if(q){local.push(q);last=now}}calProgress.style.width=((doneStages(trainingDraft)+clamp((performance.now()-start)/ms,0,1))/totalStages('refine')*100).toFixed(1)+'%';await sleep(18)}
  captureActive=false;captureStop.style.display='none';dotEl.style.display='none';if(captureAbort){showRefineGuide('That attempt was discarded. Previous dots are still saved.');return}if(local.length<10){showRefineGuide(`Only ${local.length} usable frames were visible, so this dot was not saved. Retry.`);return}
  trainingDraft.completed[stageKey(p,0)]=downsample(local,32);advanceCursor(trainingDraft);saveDraft();updateResumeUI();if(currentTrainingFinished(trainingDraft))return finalizeRefineTraining();showRefineGuide(`Saved ${local.length} usable frames from the previous dot.`);
}
async function finalizeRefineTraining(){
  const old={modelX,modelY,mapNorm,residualAnchors,trainingSamples,calibrated};setGuide('Applying precision refinement','All precision dots are saved. Updating the existing gaze map now.','Your previous completed map remains safe until this succeeds.','Please wait',()=>{},false);trainReady.disabled=true;
  try{const added=[];for(let p=0;p<REFINE_POINTS.length;p++)added.push(...(trainingDraft.completed[stageKey(p,0)]||[]));trainingSamples=downsample([...old.trainingSamples,...added]);const med=rebuildModel();saveCalibration();localStorage.removeItem(DRAFT_KEY);trainingDraft=null;updateResumeUI();$('err').textContent=med+' px';$('sampleTag').textContent=`Pose map + precision · ${trainingSamples.length} samples`;diag.textContent=`Precision refinement added ${added.length} samples. Current fit ${med}px.`;await saveAndExitTraining();msg('Precision refinement complete.',3600)}
  catch(e){console.error(e);modelX=old.modelX;modelY=old.modelY;mapNorm=old.mapNorm;residualAnchors=old.residualAnchors;trainingSamples=old.trainingSamples;calibrated=old.calibrated;trainReady.disabled=false;saveDraft();setGuide('Could not apply refinement','Your previous gaze map and unfinished refinement are both still safe.',String(e.message||e),'Return to refinement',()=>showRefineGuide(),doneStages(trainingDraft)>0)}
}

function waitForBlinkTrial(timeoutMs=6500){return new Promise((resolve,reject)=>{let active=true;const handler=v=>{if(!active)return;active=false;clearTimeout(timer);if(blinkTrialResolve===handler)blinkTrialResolve=null;resolve(v)},timer=setTimeout(()=>{if(!active)return;active=false;if(blinkTrialResolve===handler)blinkTrialResolve=null;reject(new Error('timeout'))},timeoutMs);blinkTrialResolve=handler})}
function startBlinkTraining(){
  if(!latest)return msg('Start camera and keep both eyes visible.');trainingKind='blink';trainingDraft=null;blinkAbort=false;setControlState('off',false);cal.classList.add('show');pushHistoryGuard();dotEl.style.display='none';calProgress.style.width='0%';
  setGuide('Before double-blink training','You will perform 10 intentional fast double blinks. Read this now; nothing is recording.','After you tap Start, every trial gives a clear countdown before the blink window opens. You do not need to stare at any target.','Start blink training',()=>runBlinkTraining(),false);
}
async function runBlinkTraining(){
  blinkCalActive=true;previousBlink=null;frozenClickPoint=null;blinkCalBtn.disabled=true;const trials=[];trainActions.style.display='none';copyBox.style.display='block';
  try{
    for(let i=0;i<10;i++){
      if(blinkAbort)throw new Error('cancelled');copyBox.style.display='block';ct.textContent=`Double blink ${i+1} of 10`;ch.textContent='Get comfortable. A 3-second countdown comes next.';rest.textContent='Blink naturally now if needed; this is not recording.';await sleep(i===0?1200:700);
      copyBox.style.display='none';countdown.style.display='block';for(const n of [3,2,1]){if(blinkAbort)throw new Error('cancelled');countdown.textContent=n;await sleep(650)}countdown.textContent='BLINK';
      try{const c=await waitForBlinkTrial(5200);if(c.total<1050&&c.gap>35&&c.gap<700&&c.d1>45&&c.d2>45){trials.push(c);try{navigator.vibrate?.(40)}catch{}}else{i--;}}
      catch{if(blinkAbort)throw new Error('cancelled');i--}
      countdown.style.display='none';calProgress.style.width=(Math.max(0,i+1)/10*100).toFixed(0)+'%';
      if(i===4){copyBox.style.display='block';ct.textContent='Eye break';ch.textContent='Relax and blink normally.';rest.textContent='Five examples are safely recorded. Continuing in 3 seconds.';await sleep(3000)}
    }
    blinkProfile=buildBlinkProfile(trials);saveBlink();$('blinkTag').textContent=`Double blink: trained · ${Math.round(blinkProfile.total.med)} ms`;diag.textContent=`Double blink trained. Median total ${Math.round(blinkProfile.total.med)} ms; median gap ${Math.round(blinkProfile.gap.med)} ms.`;blinkCalActive=false;blinkTrialResolve=null;await saveAndExitTraining();msg('Double blink trained. Eye control is still OFF.',3600);
  }catch(e){blinkCalActive=false;blinkTrialResolve=null;previousBlink=null;bothClosed=false;countdown.style.display='none';if(String(e.message)!=='cancelled')console.error(e);await saveAndExitTraining();msg('Blink training stopped. Previous blink profile was not erased.',3000)}
}

for(const c of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'){const d=document.createElement('button');d.className='letter';d.textContent=c;d.onclick=()=>msg('Selected '+c,900);$('letters').appendChild(d)}
cam.onclick=startCamera;calBtn.onclick=startPoseTraining;refineBtn.onclick=startRefineTraining;blinkCalBtn.onclick=startBlinkTraining;controlBtn.onclick=toggleControl;scrollBtn.onclick=toggleScroll;
$('full').onclick=async()=>{try{document.fullscreenElement?await document.exitFullscreen():await document.documentElement.requestFullscreen()}catch{}};
trainReady.onclick=()=>{const fn=guideAction;guideAction=null;trainReady.disabled=false;fn?.()};trainBack.onclick=trainingBack;trainExit.onclick=saveAndExitTraining;trainRestart.onclick=restartTraining;captureStop.onclick=()=>{captureAbort=true;blinkAbort=true};

window.addEventListener('popstate',()=>{
  if(cal.classList.contains('show')){history.pushState({gazeTraining:true},document.title);historyGuard=true;captureAbort=true;blinkAbort=true;if(trainingDraft)saveDraft();setTimeout(()=>{if(trainingDraft)pauseTraining('The Back action paused training instead of deleting it.');else saveAndExitTraining()},0)}
});
window.addEventListener('beforeunload',()=>{if(trainingDraft)saveDraft()});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&cal.classList.contains('show')&&trainingDraft){captureAbort=true;saveDraft()}});
document.addEventListener('fullscreenchange',()=>{if(cal.classList.contains('show')&&!document.fullscreenElement&&captureActive)pauseTraining('Fullscreen was closed. Completed stages are still saved.')});
window.addEventListener('orientationchange',()=>{
  setControlState('off',false);cursor.style.display='none';captureAbort=true;if(trainingDraft)saveDraft();
  setTimeout(()=>{calibrated=false;$('calState').textContent='Paused';const restored=restoreCalibration();if(!restored){$('sampleTag').textContent='Saved gaze map kept — return to the trained orientation';diag.textContent='Orientation changed. No calibration data was deleted.'}updateResumeUI();if(cal.classList.contains('show')&&trainingDraft)pauseTraining('Orientation changed. Completed stages are saved; return to the same orientation before resuming.')},500);
});

restoreBlink();restoreCalibration();updateResumeUI();initTracker();
