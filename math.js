export const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export const sleep=ms=>new Promise(r=>setTimeout(r,ms));
export const timeout=(p,ms)=>Promise.race([p,new Promise((_,rej)=>setTimeout(()=>rej(new Error('timeout')),ms))]);
export const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
export function avg(a,ids){let x=0,y=0,z=0;for(const i of ids){x+=a[i].x;y+=a[i].y;z+=a[i].z||0}return{x:x/ids.length,y:y/ids.length,z:z/ids.length}}
export function projection(p,a,b){const vx=b.x-a.x,vy=b.y-a.y,d=vx*vx+vy*vy||1e-8;return((p.x-a.x)*vx+(p.y-a.y)*vy)/d}
export function eyeVertical(p,top,bottom){const d=bottom.y-top.y;return Math.abs(d)<1e-6?.5:(p.y-top.y)/d}
export function eyePerp(p,a,b){const vx=b.x-a.x,vy=b.y-a.y,w=Math.hypot(vx,vy)||1e-6;return((p.x-a.x)*(-vy)+(p.y-a.y)*vx)/w}
export function median(a){const b=[...a].sort((x,y)=>x-y),m=b.length>>1;return b.length%2?b[m]:(b[m-1]+b[m])/2}
export function dp(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
function solve(A,b){const n=b.length,M=A.map((r,i)=>[...r,b[i]]);for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;[M[c],M[p]]=[M[p],M[c]];const d=M[c][c];if(Math.abs(d)<1e-10)continue;for(let j=c;j<=n;j++)M[c][j]/=d;for(let r=0;r<n;r++){if(r===c)continue;const f=M[r][c];for(let j=c;j<=n;j++)M[r][j]-=f*M[c][j]}}return M.map(r=>Number.isFinite(r[n])?r[n]:0)}
export function ridge(samples,targetKey,featureKey,lambda){const p=samples[0][featureKey].length,A=Array.from({length:p},()=>Array(p).fill(0)),b=Array(p).fill(0);for(const s of samples){const f=s[featureKey],y=s[targetKey],edge=1+.25*Math.abs((targetKey==='x'?s.x:s.y)-.5)*2,weight=(s.weight||1)*edge;for(let i=0;i<p;i++){b[i]+=weight*f[i]*y;for(let j=0;j<p;j++)A[i][j]+=weight*f[i]*f[j]}}for(let i=1;i<p;i++)A[i][i]+=lambda;return solve(A,b)}
