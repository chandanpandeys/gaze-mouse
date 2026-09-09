import{avg,clamp,dist,projection,eyeVertical,eyePerp,timeout}from'./math.js';

export class FaceTracker{
  constructor(){this.landmarker=null;this.ready=false;this.stream=null;this.blink={l:0,r:0};this.pose={yaw:0,pitch:0,roll:0}}
  async init(onDiag=()=>{},onSource=()=>{}){
    const sources=[
      ['unpkg','https://unpkg.com/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs','https://unpkg.com/@mediapipe/tasks-vision@0.10.35/wasm'],
      ['esm.sh','https://esm.sh/@mediapipe/tasks-vision@0.10.35?bundle','https://unpkg.com/@mediapipe/tasks-vision@0.10.35/wasm'],
      ['jsDelivr','https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs','https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm']
    ];
    let loaded,lastErr;
    for(const [name,url,wasm] of sources){
      try{
        onDiag('Trying tracker source: '+name+'…');
        const M=await timeout(import(url),12000);
        if(M.FaceLandmarker&&M.FilesetResolver){loaded={M,wasm,name};break}
      }catch(e){lastErr=e;console.warn(name,e)}
    }
    if(!loaded)throw lastErr||new Error('all tracker sources failed');
    onSource(loaded.name);onDiag('Loading vision runtime…');
    const vision=await timeout(loaded.M.FilesetResolver.forVisionTasks(loaded.wasm),18000);
    onDiag('Loading face + eye model…');
    this.landmarker=await timeout(loaded.M.FaceLandmarker.createFromOptions(vision,{
      baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:'CPU'},
      runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true,
      minFaceDetectionConfidence:.52,minFacePresenceConfidence:.52,minTrackingConfidence:.52
    }),30000);
    this.ready=true;return loaded.name;
  }
  async startCamera(video){
    this.stream?.getTracks().forEach(t=>t.stop());
    this.stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30,max:30}}});
    video.srcObject=this.stream;await video.play();
  }
  rotationFromMatrix(m){
    try{
      const d=m?.data||m;if(!d||d.length<16)return null;
      const r00=d[0],r10=d[4],r20=d[8],r21=d[9],r22=d[10];
      const pitch=Math.asin(clamp(-r20,-1,1)),yaw=Math.atan2(r10,r00),roll=Math.atan2(r21,r22);
      return{yaw:yaw*57.2958,pitch:pitch*57.2958,roll:roll*57.2958};
    }catch{return null}
  }
  features(result){
    const a=result.faceLandmarks?.[0];if(!a||a.length<478)return null;
    const bm={};for(const c of result.faceBlendshapes?.[0]?.categories||[])bm[c.categoryName]=c.score;
    const irisL=avg(a,[468,469,470,471,472]),irisR=avg(a,[473,474,475,476,477]);
    const lOuter=a[33],lInner=a[133],rInner=a[362],rOuter=a[263];
    const lTop=avg(a,[159,158,160]),lBot=avg(a,[145,144,153]),rTop=avg(a,[386,385,387]),rBot=avg(a,[374,373,380]);
    const lWidth=dist(lOuter,lInner)||1e-5,rWidth=dist(rInner,rOuter)||1e-5;
    const lx=projection(irisL,lOuter,lInner),rx=projection(irisR,rInner,rOuter);
    const lyLid=eyeVertical(irisL,lTop,lBot),ryLid=eyeVertical(irisR,rTop,rBot);
    const lyAxis=eyePerp(irisL,lOuter,lInner)/lWidth,ryAxis=eyePerp(irisR,rInner,rOuter)/rWidth;
    const lTopShape=eyePerp(lTop,lOuter,lInner)/lWidth,lBotShape=eyePerp(lBot,lOuter,lInner)/lWidth;
    const rTopShape=eyePerp(rTop,rInner,rOuter)/rWidth,rBotShape=eyePerp(rBot,rInner,rOuter)/rWidth;
    const lOpen=dist(lTop,lBot)/lWidth,rOpen=dist(rTop,rBot)/rWidth;
    const eyeMid={x:(lOuter.x+rOuter.x)/2,y:(lOuter.y+rOuter.y)/2},eyeDist=dist(lOuter,rOuter)||1e-5;
    const nose=a[1],chin=a[152],forehead=a[10];
    const faceCenter={x:(eyeMid.x+nose.x)/2,y:(eyeMid.y+nose.y)/2};
    const faceHeight=dist(forehead,chin)||1e-5;
    const yawProxy=(nose.x-eyeMid.x)/eyeDist,pitchProxy=(nose.y-eyeMid.y)/eyeDist,rollProxy=Math.atan2(rOuter.y-lOuter.y,rOuter.x-lOuter.x);
    this.pose=this.rotationFromMatrix(result.facialTransformationMatrixes?.[0])||{yaw:yawProxy*50,pitch:pitchProxy*45,roll:rollProxy*57.2958};
    this.blink={l:bm.eyeBlinkLeft||0,r:bm.eyeBlinkRight||0};
    const lookX=(bm.eyeLookOutLeft||0)-(bm.eyeLookInLeft||0)+(bm.eyeLookInRight||0)-(bm.eyeLookOutRight||0);
    const lookDown=((bm.eyeLookDownLeft||0)+(bm.eyeLookDownRight||0))/2,lookUp=((bm.eyeLookUpLeft||0)+(bm.eyeLookUpRight||0))/2,lookY=lookDown-lookUp;
    const avgX=(lx+rx)/2,avgYAxis=(lyAxis+ryAxis)/2,avgYLid=(lyLid+ryLid)/2,open=(lOpen+rOpen)/2;
    const yaw=this.pose.yaw/45,pitch=this.pose.pitch/35,roll=this.pose.roll/30;
    const cx=(faceCenter.x-.5)*2,cy=(faceCenter.y-.5)*2,scale=(eyeDist-.22)/.12,fh=(faceHeight-.48)/.18;
    const xF=[1,avgX,lx,rx,lookX,avgX*avgX,lookX*lookX,avgX*lookX,cx,cy,scale,yaw,pitch,roll,avgX*yaw,avgX*cx,lookX*yaw,lookX*cx,avgX*scale,yaw*cx];
    const yF=[1,avgYAxis,lyAxis,ryAxis,avgYLid,lyLid,ryLid,lOpen,rOpen,lTopShape,lBotShape,rTopShape,rBotShape,lookY,lookDown,lookUp,avgYAxis*avgYAxis,lookY*lookY,avgYAxis*lookY,open*lookY,cx,cy,scale,fh,yaw,pitch,roll,avgYAxis*pitch,avgYAxis*cy,lookY*pitch,lookY*cy,avgYAxis*scale,pitch*cy];
    const mapF=[avgX,avgYAxis,lookX,lookY,lx,rx,lyAxis,ryAxis,lOpen,rOpen,cx,cy,scale,fh,yaw,pitch,roll];
    return{xF,yF,mapF,blink:this.blink,pose:this.pose,geometry:{centerX:faceCenter.x,centerY:faceCenter.y,eyeDist,faceHeight}};
  }
  detect(video,now){if(!this.ready||!this.landmarker||video.readyState<2)return null;return this.features(this.landmarker.detectForVideo(video,now))}
}
