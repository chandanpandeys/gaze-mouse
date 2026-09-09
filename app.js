import{clamp,sleep,median,dp,ridge}from'./math.js';
import{FaceTracker}from'./tracker.js';

const $=id=>document.getElementById(id);
const video=$('video'),status=$('status'),diag=$('diag'),cam=$('cam'),calBtn=$('calBtn'),refineBtn=$('refineBtn'),blinkCalBtn=$('blinkCalBtn'),controlBtn=$('controlBtn'),scrollBtn=$('scrollBtn'),assist=$('assist'),cursor=$('cursor'),cal=$('cal'),dotEl=$('dot'),ct=$('ct'),ch=$('ch'),rest=$('rest'),calProgress=$('calProgress'),toast=$('toast');
const tracker=new FaceTracker();

let running=false,calibrated=false,latest=null,latestAt=0,lastInfer=0,frames=0,fpsAt=performance.now(),toastTimer=null;
let modelX=null,modelY=null,trainingSamples=[],residualAnchors=[],mapNorm=null;
let sx=innerWidth/2,sy=innerHeight/2,lastTargetX=sx,lastTargetY=sy,lastStablePoint={x:sx,y:sy,at:0,velocity:999},focusEl=null,clicks=0;
let controlState='off',frozenClickPoint=null,leftWinkAt=0,leftWinkLatched=false,rightWinkAt=0,rightWinkLatched=false;
let bothClosed=false,blinkStart=0,blinkSymSum=0,blinkSymN=0,previousBlink=null,blinkProfile=null,blinkCalActive=false,blinkTrialResolve=null;
let scrollZone='center',scrollZoneSince=0,lastScrollAt=0;

const CAL_KEY='gazeMousePoseMap-v1',BLINK_KEY='gazeMouseBlink-v1';
const POSE_POINTS=[
  [.09,.10],[.50,.10],[.91,.10],
  [.09,.50],[.50,.50],[.91,.50],
  [.09,.90],[.50,.90],[.91,.90]
];
const REFINE_POINTS=[];for(const y of [.10,.36,.64,.90])for(const x of [.08,.29,.50,.71,.92])REFINE_POINTS.push([x,y]);
const POSE_STAGES=[
  ['Hold steady for a moment',900],
  ['Move the PHONE slowly LEFT ↔ RIGHT',1900],
  ['Move the PHONE slowly UP ↕ DOWN',2100],
  ['Move the PHONE slowly CLOSER ↔ FARTHER',1900],
  ['Gently change the phone ANGLE / TILT',1700]
];

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
function calibrationFrameOK(){return latest&&performance.now()-latestAt<140&&latest.blink.l<.82&&latest.blink.r<.82}
function snapTarget(x,y,velocity){
  if(!assist.checked||velocity>13)return{x,y,el:null};
  const els=[...document.querySelectorAll('.letter,button:not(:disabled),select')];let best=null,bd=27;
  for(const el of els){const r=el.getBoundingClientRect();if(!r.width||!r.height)continue;const cx=r.left+r.width/2,cy=r.top+r.height/2,d=Math.hypot(x-cx,y-cy);if(d<bd){bd=d;best={x:cx,y:cy,el}}}
  return best||{x,y,el:null};
}
function smoothCursor(tx,ty){
  const dx=tx-sx,dy=ty-sy,d=Math.hypot(dx,dy);
  let alpha=d<12?.045:d<38?.075:d<100?.12:d<220?.18:.23;
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
function rebuildModel(){
  if(trainingSamples.length<120)throw new Error('not enough training samples');
  modelX=ridge(trainingSamples,'x','xF',.016);modelY=ridge(trainingSamples,'y','yF',.018);mapNorm=buildMapNorm(trainingSamples);
  const stride=Math.max(1,Math.floor(trainingSamples.length/650));residualAnchors=[];
  for(let i=0;i<trainingSamples.length;i+=stride){const s=trainingSamples[i],gx=dp(s.xF,modelX),gy=dp(s.yF,modelY);residualAnchors.push({nf:normMap(s.mapF),rx:s.x-gx,ry:s.y-gy})}
  let errors=[];for(let i=0;i<trainingSamples.length;i+=13){const s=trainingSamples[i],p=predictPoint(s);errors.push(Math.hypot((p.x-s.x)*innerWidth,(p.y-s.y)*innerHeight))}
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
  lastStablePoint={x:sn.x,y:sn.y,at:performance.now(),velocity:sm.velocity};
  $('stableTag').textContent=`Cursor ${Math.round(sm.velocity)}px · map ${Math.round(p.confidence*100)}%`;lastTargetX=sm.x;lastTargetY=sm.y;
  if(controlState==='off'){cursor.style.display='none';clearFocus();return}
  cursor.style.left=sn.x+'px';cursor.style.top=sn.y+'px';cursor.style.display='block';if(controlState==='scroll')handleScroll(sn.y,now);
}

function saveCalibration(){
  try{localStorage.setItem(CAL_KEY,JSON.stringify({ratio:innerWidth/innerHeight,modelX,modelY,mapNorm,residualAnchors,trainingSamples:trainingSamples.slice(-1900)}))}catch(e){console.warn('save calibration',e)}
}
function restoreCalibration(){
  try{
    const d=JSON.parse(localStorage.getItem(CAL_KEY)||'null');if(!d||Math.abs(d.ratio-innerWidth/innerHeight)>.12)return;
    modelX=d.modelX;modelY=d.modelY;mapNorm=d.mapNorm;residualAnchors=d.residualAnchors||[];trainingSamples=d.trainingSamples||[];
    if(modelX&&modelY&&mapNorm&&residualAnchors.length){calibrated=true;$('calState').textContent='Saved';$('sampleTag').textContent=`Pose map: restored · ${trainingSamples.length} samples`;controlBtn.disabled=false;refineBtn.disabled=false;setControlState('off',false)}
  }catch(e){console.warn('restore calibration',e)}
}
function saveBlink(){try{if(blinkProfile)localStorage.setItem(BLINK_KEY,JSON.stringify(blinkProfile))}catch{}}
function restoreBlink(){try{blinkProfile=JSON.parse(localStorage.getItem(BLINK_KEY)||'null');if(blinkProfile)$('blinkTag').textContent=`Double blink: trained · ${Math.round(blinkProfile.total.med)} ms`}catch{}}

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
      diag.textContent=calibrated?'Pose-aware gaze map active. Actions only happen while Eye control is ON.':'Face found. Run 3D pose-map training.';
      processGestures(now);if(calibrated&&!cal.classList.contains('show')&&latest.blink.l<.56&&latest.blink.r<.56)updateCursor(now);
    }else{st('Find my face…');calBtn.disabled=true;blinkCalBtn.disabled=true;controlBtn.disabled=true;refineBtn.disabled=true;scrollBtn.disabled=true;cursor.style.display='none';clearFocus()}
  }catch(e){console.warn(e)}
  frames++;if(now-fpsAt>1000){$('fps').textContent=Math.round(frames*1000/(now-fpsAt));frames=0;fpsAt=now}
}

async function enterFullscreen(){try{if(!document.fullscreenElement)await document.documentElement.requestFullscreen()}catch{}}
function poseCoverage(samples){
  if(samples.length<2)return'coverage starting…';
  const vals=k=>samples.map(s=>s[k]),span=k=>Math.max(...vals(k))-Math.min(...vals(k));
  const cx=span('cx'),cy=span('cy'),sc=span('sc'),yaw=span('yaw'),pitch=span('pitch');
  return`coverage x ${Math.round(cx*100)} · y ${Math.round(cy*100)} · depth ${Math.round(sc*1000)} · angle ${Math.round(yaw)}°/${Math.round(pitch)}°`;
}
function captureSample(x,y,weight=1){
  if(!calibrationFrameOK())return null;
  return{xF:[...latest.xF],yF:[...latest.yF],mapF:[...latest.mapF],x,y,weight,cx:latest.geometry.centerX,cy:latest.geometry.centerY,sc:latest.geometry.eyeDist,yaw:latest.pose.yaw,pitch:latest.pose.pitch};
}
async function trainPosePoint(point,index,out){
  const[x,y]=point;dotEl.style.left=x*100+'%';dotEl.style.top=y*100+'%';dotEl.style.display='block';
  ct.textContent=`Pose-map point ${index+1} of ${POSE_POINTS.length}`;ch.textContent='Keep your eyes on the tiny center. The dot will NOT move.';rest.textContent='Settle and blink now. Then move the PHONE, not your gaze.';await sleep(1500);
  const local=[];let stageIndex=0;
  for(const[label,ms]of POSE_STAGES){
    ch.textContent=label;rest.textContent='Keep looking at the same tiny dot. Move slowly and naturally.';
    const start=performance.now();let last=0;
    while(performance.now()-start<ms){
      const now=performance.now();if(now-last>44){const s=captureSample(x,y,1);if(s){out.push(s);local.push(s);last=now}}
      rest.textContent=`${label} · ${local.length} frames · ${poseCoverage(local)}`;
      const within=(performance.now()-start)/ms,overall=(index+(stageIndex+within)/POSE_STAGES.length)/POSE_POINTS.length;calProgress.style.width=(overall*100).toFixed(1)+'%';
      await sleep(18);
    }
    stageIndex++;await sleep(180);
  }
  rest.textContent=`Point ${index+1} recorded · ${local.length} usable frames. Moving on automatically.`;await sleep(650);
}
async function calibratePoseMap(){
  if(!latest)return msg('Start camera and keep your full face visible.');
  await enterFullscreen();setControlState('off',false);calibrated=false;$('calState').textContent='No';$('err').textContent='—';clearFocus();cursor.style.display='none';
  calBtn.disabled=true;refineBtn.disabled=true;cal.classList.add('show');calProgress.style.width='0%';dotEl.style.display='block';
  try{
    const samples=[];ct.textContent='3D pose-map training';ch.textContent='Nine fixed dots. For each dot, keep looking at it while you slowly move the phone through different positions and angles.';rest.textContent='This trains screen focus across face position, distance and camera angle. Starts in 4 seconds.';dotEl.style.left='50%';dotEl.style.top='50%';await sleep(4000);
    for(let i=0;i<POSE_POINTS.length;i++){
      if(i>0&&i%3===0){ct.textContent='Eye break';ch.textContent='Relax and blink normally.';rest.textContent='Next row starts in 3 seconds.';dotEl.style.left='50%';dotEl.style.top='50%';await sleep(3000)}
      await trainPosePoint(POSE_POINTS[i],i,samples);$('sampleTag').textContent=`Pose map: ${i+1}/${POSE_POINTS.length} dots · ${samples.length} samples`;
    }
    if(samples.length<700)throw new Error('too few usable samples');
    trainingSamples=samples.slice(-1500);const med=rebuildModel();calibrated=true;$('calState').textContent='Yes';$('err').textContent=med+' px';
    sx=lastTargetX=innerWidth/2;sy=lastTargetY=innerHeight/2;lastStablePoint={x:sx,y:sy,at:0,velocity:999};controlBtn.disabled=false;refineBtn.disabled=false;setControlState('off',false);saveCalibration();
    $('sampleTag').textContent=`Pose map: trained · ${trainingSamples.length} samples`;diag.textContent=`Pose map trained across phone position/angle. Fit ${med}px. Optional Precision refine adds more screen locations.`;msg('Pose map complete. Run Precision refine for smaller targets.',4200);
  }catch(e){console.error(e);modelX=modelY=null;mapNorm=null;residualAnchors=[];diag.textContent='Pose-map training failed: '+(e.message||e);msg('Training failed. Retry with the face visible.',4000)}
  finally{cal.classList.remove('show');calBtn.disabled=!latest;refineBtn.disabled=!calibrated;calProgress.style.width='0%'}
}
async function refinePrecision(){
  if(!calibrated||!latest)return msg('Complete the 3D pose map first.');
  await enterFullscreen();setControlState('off',false);refineBtn.disabled=true;calBtn.disabled=true;cal.classList.add('show');calProgress.style.width='0%';dotEl.style.display='block';
  try{
    const added=[];ct.textContent='Precision refinement';ch.textContent='20 tiny fixed dots. Keep the phone in a normal position now — only your eyes move.';rest.textContent='Each point is short. Blink between points.';await sleep(2800);
    for(let i=0;i<REFINE_POINTS.length;i++){
      const[x,y]=REFINE_POINTS[i];dotEl.style.left=x*100+'%';dotEl.style.top=y*100+'%';ct.textContent=`Precision point ${i+1} of ${REFINE_POINTS.length}`;ch.textContent='Look exactly at the tiny center.';rest.textContent='Settle…';await sleep(650);
      const start=performance.now();let last=0,count=0;rest.textContent='Recording this exact screen location…';
      while(performance.now()-start<1050){const now=performance.now();if(now-last>44){const s=captureSample(x,y,1.35);if(s){added.push(s);last=now;count++}}calProgress.style.width=((i+(performance.now()-start)/1050)/REFINE_POINTS.length*100).toFixed(1)+'%';await sleep(18)}
      rest.textContent=`${count} frames recorded`;await sleep(260);
      if(i===9){ct.textContent='Short eye break';ch.textContent='Blink and relax for 2 seconds.';rest.textContent='Ten points remain.';await sleep(2000)}
    }
    trainingSamples=[...trainingSamples,...added].slice(-1900);const med=rebuildModel();$('err').textContent=med+' px';saveCalibration();$('sampleTag').textContent=`Pose map + precision: ${trainingSamples.length} samples`;diag.textContent=`Precision refinement added ${added.length} samples. Current fit ${med}px.`;msg('Precision refinement complete.',3600);
  }catch(e){console.error(e);msg('Precision refinement stopped. Base pose map is still kept.',3500)}
  finally{cal.classList.remove('show');refineBtn.disabled=!calibrated;calBtn.disabled=!latest;calProgress.style.width='0%'}
}

function waitForBlinkTrial(timeoutMs=6500){return new Promise((resolve,reject)=>{let active=true;const handler=v=>{if(!active)return;active=false;clearTimeout(timer);if(blinkTrialResolve===handler)blinkTrialResolve=null;resolve(v)},timer=setTimeout(()=>{if(!active)return;active=false;if(blinkTrialResolve===handler)blinkTrialResolve=null;reject(new Error('timeout'))},timeoutMs);blinkTrialResolve=handler})}
async function calibrateBlink(){
  if(!latest)return msg('Start camera and keep both eyes visible.');setControlState('off',false);blinkCalActive=true;previousBlink=null;frozenClickPoint=null;blinkCalBtn.disabled=true;cal.classList.add('show');dotEl.style.display='none';calProgress.style.width='0%';const trials=[];
  try{
    ct.textContent='Double-blink training';ch.textContent='We will record 10 fast intentional double blinks.';rest.textContent='Training begins in 3 seconds.';await sleep(3000);
    for(let i=0;i<10;i++){
      if(i===5){ct.textContent='Eye break';ch.textContent='Relax and blink normally for 3 seconds.';rest.textContent='Five examples left.';await sleep(3000)}
      let ok=false;while(!ok){ct.textContent=`Double blink ${i+1} of 10`;ch.textContent='Wait for READY, then perform one quick double blink.';rest.textContent='Do not blink early.';await sleep(900);ch.textContent='READY — double blink now';
        try{const c=await waitForBlinkTrial();if(c.total<1050&&c.gap>35&&c.gap<700&&c.d1>45&&c.d2>45){trials.push(c);ok=true;msg('Recorded',550)}else{rest.textContent='Unusual timing. Repeat this one.';await sleep(700)}}catch{rest.textContent='No clear double blink. Repeat this one.';await sleep(700)}}
      calProgress.style.width=((i+1)/10*100).toFixed(0)+'%';await sleep(420);
    }
    blinkProfile=buildBlinkProfile(trials);saveBlink();$('blinkTag').textContent=`Double blink: trained · ${Math.round(blinkProfile.total.med)} ms`;diag.textContent=`Double blink trained. Median total ${Math.round(blinkProfile.total.med)} ms; median gap ${Math.round(blinkProfile.gap.med)} ms.`;msg('Double blink trained. Eye control is still OFF.',3600);
  }catch(e){console.error(e);blinkProfile=null;$('blinkTag').textContent='Double blink: untrained';msg('Blink training stopped. Try again.',3000)}
  finally{blinkCalActive=false;blinkTrialResolve=null;previousBlink=null;bothClosed=false;cal.classList.remove('show');dotEl.style.display='block';calProgress.style.width='0%';blinkCalBtn.disabled=!latest}
}

for(const c of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'){const d=document.createElement('button');d.className='letter';d.textContent=c;d.onclick=()=>msg('Selected '+c,900);$('letters').appendChild(d)}
cam.onclick=startCamera;calBtn.onclick=calibratePoseMap;refineBtn.onclick=refinePrecision;blinkCalBtn.onclick=calibrateBlink;controlBtn.onclick=toggleControl;scrollBtn.onclick=toggleScroll;$('full').onclick=async()=>{try{document.fullscreenElement?await document.exitFullscreen():await document.documentElement.requestFullscreen()}catch{}};
window.addEventListener('orientationchange',()=>{calibrated=false;modelX=modelY=null;mapNorm=null;residualAnchors=[];$('calState').textContent='No';$('err').textContent='—';cursor.style.display='none';setControlState('off',false);msg('Orientation changed. Train the pose map again.',3200)});
restoreBlink();restoreCalibration();initTracker();
