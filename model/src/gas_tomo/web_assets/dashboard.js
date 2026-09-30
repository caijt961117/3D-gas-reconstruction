/* Offline viewer. Full-volume sampling and camera coordinate transforms are explicit. */
'use strict';
const D=JSON.parse(document.getElementById('gas-data').textContent), $=id=>document.getElementById(id);
window.GAS_TOMO_FRAMES=window.GAS_TOMO_FRAMES||{};
window.GAS_TOMO_FRAMES[0]=D.initial;
D.manifest.forEach((m,i)=>{if(m.embedded)window.GAS_TOMO_FRAMES[i]=m.embedded;});
const [nx,ny,nz]=D.shape, [dx,dy,dz]=D.spacing;
const X=Array.from({length:nx},(_,i)=>D.bounds[0][0]+(i+.5)*dx),Y=Array.from({length:ny},(_,i)=>D.bounds[1][0]+(i+.5)*dy),Z=Array.from({length:nz},(_,i)=>D.bounds[2][0]+(i+.5)*dz);
let state={frame:0,sx:Math.floor(nx/2),sy:Math.floor(ny/2),sz:Math.floor(nz/2),mode:'iso',probe:null,playing:false,busy:false,request:0};
let webglAvailable=false;try{const cv=document.createElement('canvas');webglAvailable=!!(cv.getContext('webgl2')||cv.getContext('webgl'));}catch(e){}
window.gasTomoWebGLAvailable=webglAvailable;
let current, colors=D.color_range.slice(), iso=D.levels.length?D.levels[Math.floor(D.levels.length/2)]:colors[1]*.25, timer=null;
const PLOT_CFG={responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:'png',filename:'gas_3d',height:900,width:1200,scale:2}};
const fmt=x=>x===null||x===undefined||!Number.isFinite(x)?'不可评价':Number(x).toPrecision(5);
function error(message){$('error').style.display=message?'block':'none';$('error').textContent=message||'';}
function decode(f){if(!f.values){let raw=atob(f.values_b64),a=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)a[i]=raw.charCodeAt(i);let dv=new DataView(a.buffer);f.values=new Float32Array(raw.length/4);for(let i=0;i<f.values.length;i++)f.values[i]=dv.getFloat32(i*4,true);}return f;}
function value(i,j,k){return current.values[(k*ny+j)*nx+i];}
function idxList(n,step){let a=[];for(let i=0;i<n;i+=step)a.push(i);if(a[a.length-1]!==n-1)a.push(n-1);return a;}
let step=1,ix,iy,iz;do{ix=idxList(nx,step);iy=idxList(ny,step);iz=idxList(nz,step);if(!D.max_voxels||ix.length*iy.length*iz.length<=D.max_voxels)break;step++;}while(step<=Math.max(nx,ny,nz));
let VX=[],VY=[],VZ=[],VI=[];for(const k of iz)for(const j of iy)for(const i of ix){VX.push(X[i]);VY.push(Y[j]);VZ.push(Z[k]);VI.push((k*ny+j)*nx+i);}
function box(){let x=[],y=[],z=[];const B=D.bounds;for(let a=0;a<3;a++){let other=[0,1,2].filter(v=>v!==a);for(let q=0;q<2;q++)for(let r=0;r<2;r++){let v=[0,0,0],w=[0,0,0];v[a]=B[a][0];w[a]=B[a][1];v[other[0]]=w[other[0]]=B[other[0]][q];v[other[1]]=w[other[1]]=B[other[1]][r];x.push(v[0],w[0],null);y.push(v[1],w[1],null);z.push(v[2],w[2],null);}}return {type:'scatter3d',mode:'lines',x,y,z,name:'重建域',line:{width:2},hoverinfo:'skip'};}
function sliceTrace(axis,n){let x=[],y=[],z=[],surfacecolor=[];
 if(axis==='z'){for(let j=0;j<ny;j++){x.push(X.slice());y.push(X.map(()=>Y[j]));z.push(X.map(()=>Z[n]));surfacecolor.push(X.map((_,i)=>value(i,j,n)));}}
 if(axis==='y'){for(let k=0;k<nz;k++){x.push(X.slice());y.push(X.map(()=>Y[n]));z.push(X.map(()=>Z[k]));surfacecolor.push(X.map((_,i)=>value(i,n,k)));}}
 if(axis==='x'){for(let k=0;k<nz;k++){x.push(Y.map(()=>X[n]));y.push(Y.slice());z.push(Y.map(()=>Z[k]));surfacecolor.push(Y.map((_,j)=>value(n,j,k)));}}
 return {type:'surface',x,y,z,surfacecolor,coloraxis:'coloraxis',opacity:.90,name:axis.toUpperCase()+'切片',showscale:false,hovertemplate:'X=%{x:.4f} m<br>Y=%{y:.4f} m<br>Z=%{z:.4f} m<br>C=%{surfacecolor:.6g}<extra></extra>'};}
function sampleProbe(p){if(!current.has_estimate)return null;const A=[X,Y,Z];let ids=[],f=[];for(let a=0;a<3;a++){if(!Number.isFinite(p[a])||p[a]<A[a][0]||p[a]>A[a][A[a].length-1])throw Error('坐标超出体素中心的插值范围。');let u=(p[a]-A[a][0])/D.spacing[a],i=Math.min(Math.floor(u),A[a].length-2);ids.push(i);f.push(u-i);}let sum=0;for(let k=0;k<2;k++)for(let j=0;j<2;j++)for(let i=0;i<2;i++)sum+=value(ids[0]+i,ids[1]+j,ids[2]+k)*(i?f[0]:1-f[0])*(j?f[1]:1-f[1])*(k?f[2]:1-f[2]);return sum;}
function project(camera,p){let C=camera.R.map((r,i)=>r.reduce((s,v,j)=>s+v*p[j],0)+camera.t[i]);if(C[2]<=0)return null;let K=camera.K,u=(K[0][0]*C[0]+K[0][1]*C[1])/C[2]+K[0][2]-camera.roi[0],v=K[1][1]*C[1]/C[2]+K[1][2]-camera.roi[1];let w=camera.roi[2],h=camera.roi[3],k=camera.rotate_k%4,a=u,b=v;if(k===1){a=v;b=w-1-u;}if(k===2){a=w-1-u;b=h-1-v;}if(k===3){a=h-1-v;b=u;}if(k%2){let t=w;w=h;h=t;}if(camera.flip_horizontal)a=w-1-a;if(camera.flip_vertical)b=h-1-b;if(a<-.5||a>=w-.5||b<-.5||b>=h-.5)return null;let [rows,cols]=(D.measurement_shapes||{})[camera.name]||D.measurement_shape;return [(a+.5)*cols/w-.5,(b+.5)*rows/h-.5];}
async function renderScene(){
 const mode=state.mode,op=Number($('opacity').value),traces=[];
 if(!webglAvailable){
   $('fallbackNote').style.display='block';$('fallbackNote').textContent='当前浏览器没有可用 WebGL：等值面显示为静态三维预览。切换“正交切片”仍可移动查看；探针仍按完整数组查询。静态预览不能旋转，其色标为导出时设置。';
   if(mode==='slices'){
     $('sceneFallback').style.display='none';$('scene').style.display='block';
     let st=[sliceTrace('x',state.sx),sliceTrace('y',state.sy),sliceTrace('z',state.sz)];
     let lt=st.map((t,i)=>({type:'heatmap',z:t.surfacecolor,coloraxis:'coloraxis',xaxis:i?'x'+(i+1):'x',yaxis:i?'y'+(i+1):'y',name:t.name}));
     let layout={margin:{l:40,r:40,t:60,b:35},coloraxis:{cmin:colors[0],cmax:colors[1],colorscale:D.colorscale,colorbar:{title:{text:D.unit}}},grid:{rows:1,columns:3,pattern:'independent'},annotations:[{text:'YZ 切片',x:.13,y:1.08,xref:'paper',yref:'paper',showarrow:false},{text:'XZ 切片',x:.5,y:1.08,xref:'paper',yref:'paper',showarrow:false},{text:'XY 切片',x:.88,y:1.08,xref:'paper',yref:'paper',showarrow:false}]};
     lt[0].x=Y;lt[0].y=Z;lt[1].x=X;lt[1].y=Z;lt[2].x=X;lt[2].y=Y;
     await Plotly.react('scene',lt,layout,PLOT_CFG);
   }else{
     $('scene').style.display='none';$('sceneFallback').style.display='block';$('sceneFallback').src='data:image/png;base64,'+current.static_png;
   }
   return;
 }

 if(current.has_estimate){
 if(mode==='iso')for(const mesh of current.meshes){traces.push({type:'mesh3d',...mesh,intensity:Array(mesh.x.length).fill(mesh.level),coloraxis:'coloraxis',opacity:op,name:'C='+fmt(mesh.level),showscale:false});}
 if(mode==='volume'||mode==='combined')traces.push({type:'volume',x:VX,y:VY,z:VZ,value:VI.map(i=>current.values[i]),isomin:Math.max(0,colors[1]*D.min_fraction),isomax:colors[1],surface:{count:18},opacity:Math.min(op,.2),coloraxis:'coloraxis',caps:{x:{show:false},y:{show:false},z:{show:false}},name:'体渲染'});
 if(mode==='custom'&&current.minimum<iso&&iso<current.peak)traces.push({type:'isosurface',x:VX,y:VY,z:VZ,value:VI.map(i=>current.values[i]),isomin:iso,isomax:iso,surface:{count:1},opacity:op,coloraxis:'coloraxis',caps:{x:{show:false},y:{show:false},z:{show:false}},name:'C='+fmt(iso)});
 if(mode==='slices'||mode==='combined')traces.push(sliceTrace('x',state.sx),sliceTrace('y',state.sy),sliceTrace('z',state.sz));
 if(mode==='points'){let ids=[];for(let i=0;i<current.values.length;i++)if(current.values[i]>=iso)ids.push(i);let stride=Math.max(1,Math.ceil(ids.length/D.point_limit));ids=ids.filter((_,i)=>i%stride===0);traces.push({type:'scatter3d',mode:'markers',x:ids.map(i=>X[i%nx]),y:ids.map(i=>Y[Math.floor(i/nx)%ny]),z:ids.map(i=>Z[Math.floor(i/(nx*ny))]),marker:{size:3,opacity:op,color:ids.map(i=>current.values[i]),coloraxis:'coloraxis'},name:'体素点云'});}
 const peakloc=current.metadata.maximum_location_m;if(peakloc)traces.push({type:'scatter3d',mode:'markers',x:[peakloc[0]],y:[peakloc[1]],z:[peakloc[2]],marker:{size:6,symbol:'diamond'},name:'峰值位置'});
 if(state.probe)traces.push({type:'scatter3d',mode:'markers',x:[state.probe[0]],y:[state.probe[1]],z:[state.probe[2]],marker:{size:7,symbol:'cross'},name:'空间探针'});
 }
 traces.push(box());let layout={margin:{l:0,r:15,t:30,b:0},showlegend:true,legend:{orientation:'h',y:1.05},uirevision:'fixed-world-camera',scene:{xaxis:{title:{text:'X (m)'},range:D.bounds[0]},yaxis:{title:{text:'Y (m)'},range:D.bounds[1]},zaxis:{title:{text:'Z (m)'},range:D.bounds[2]},aspectmode:'data',camera:{eye:{x:1.6,y:-1.8,z:1.0}}},coloraxis:{cmin:colors[0],cmax:colors[1],colorscale:D.colorscale,colorbar:{title:{text:D.unit},thickness:16}}};
 if(!current.has_estimate)layout.annotations=[{text:'有效观测不足：没有浓度估计（不是零浓度）',showarrow:false,x:.5,y:.5,xref:'paper',yref:'paper'}];
 else if((mode==='iso'&&!current.meshes.length)||(mode==='custom'&&!(current.minimum<iso&&iso<current.peak)))layout.annotations=[{text:'所选阈值没有内部等值面，请使用切片查看。',showarrow:false,x:.5,y:.1,xref:'paper',yref:'paper'}];
 await Plotly.react('scene',traces,layout,PLOT_CFG);
}
function projectionPlots(){if(!D.measurement_shape||!current.observed||!current.predicted){$('diagnostics').textContent='此文件未包含相机投影观测。';return Promise.resolve();}
 let min=Infinity,max=-Infinity,rmax=0;
 current.observed.forEach((b,i)=>{if(!current.weights||current.weights[i]>0){min=Math.min(min,b,current.predicted[i]);max=Math.max(max,b,current.predicted[i]);rmax=Math.max(rmax,Math.abs(current.predicted[i]-b));}});
 if(!Number.isFinite(min)||min===max){min=0;max=1;}if(!rmax)rmax=1;
 const names=['观测','前向投影','残差'],jobs=[];
 let offset=0;for(let c=0;c<D.cameras.length;c++){let [rows,cols]=(D.measurement_shapes||{})[D.cameras[c].name]||D.measurement_shape;let n=rows*cols;for(let kind=0;kind<3;kind++){let id='camera_'+c+'_'+kind,el=$(id);if(!el){el=document.createElement('div');el.id=id;el.className='camplot';$('diagnostics').appendChild(el);}let z=[];for(let j=0;j<rows;j++){let line=[];for(let i=0;i<cols;i++){let k=offset+j*cols+i,v=kind===0?current.observed[k]:kind===1?current.predicted[k]:current.predicted[k]-current.observed[k];line.push(current.weights&&current.weights[k]<=0?null:v);}z.push(line);}let traces=[{type:'heatmap',z,zmin:kind===2?-rmax:min,zmax:kind===2?rmax:max,colorscale:kind===2?'RdBu':D.colorscale,showscale:true,colorbar:{thickness:9},hoverongaps:false}];
 if(state.probe&&D.cameras[c]){let p=project(D.cameras[c],state.probe);if(p)traces.push({type:'scatter',mode:'markers',x:[p[0]],y:[p[1]],marker:{size:11,symbol:'x'},name:'探针',showlegend:false});}
 let name=D.cameras[c]?D.cameras[c].name:'camera '+c;
 jobs.push(Plotly.react(el,traces,{title:{text:name+' · '+names[kind],font:{size:13}},margin:{l:32,r:20,t:33,b:26},xaxis:{title:{text:'列'},range:[-.5,cols-.5]},yaxis:{title:{text:'行'},autorange:'reversed',scaleanchor:'x'},uirevision:'camera-grid'},{responsive:true,displaylogo:false}));}offset+=n;}return Promise.all(jobs);}
function updateText(){let m=current.metadata,q=m.quality||{},entry=D.manifest[state.frame];$('peak').textContent=current.has_estimate?fmt(m.maximum_concentration??current.peak):'无估计';$('integral').textContent=fmt(m.concentration_integral);$('residual').textContent=fmt(m.weighted_relative_projection_residual);$('frameLabel').textContent=`${state.frame+1}/${D.manifest.length} · ${entry.frame_id}`+(entry.timestamp_s!=null?' · t='+entry.timestamp_s.toFixed(4)+' s':' · 无时间戳');$('quality').textContent=`数据：${q.data_status||'未附状态'}\n求解：${q.solver_status||'未附状态'}`+(q.measurement_supported===false?'\n注意：无当前测量支持，可能仅为上一帧预测。':'');$('quality').className='status '+(['valid_two_view','valid_multi_view'].includes(q.data_status)?'ok':'bad');$('boundary').textContent=(m.boundary_touch_at_5pct_peak||[]).length?'边界截断提示：'+m.boundary_touch_at_5pct_peak.join(', ')+'（5%峰值阈值）':'当前记录无边界触及提示。';$('meta').textContent=JSON.stringify(m,null,2);$('resolution').textContent=`原始网格：${nx}×${ny}×${nz}；体显示网格：${ix.length}×${iy.length}×${iz.length}；切片与探针使用原始网格。`;}
async function loadFrame(index){if(window.GAS_TOMO_FRAMES[index])return decode(window.GAS_TOMO_FRAMES[index]);let item=D.manifest[index];if(item.embedded)return decode(item.embedded);await new Promise((resolve,reject)=>{let s=document.createElement('script');s.src=item.asset;s.onload=()=>{s.remove();resolve();};s.onerror=()=>{s.remove();reject(Error('无法读取序列资源。请保留 HTML 旁边的 _assets 文件夹。'));};document.head.appendChild(s);});if(!window.GAS_TOMO_FRAMES[index])throw Error('Frame asset not found.');return decode(window.GAS_TOMO_FRAMES[index]);}
async function showFrame(i){const token=++state.request;state.busy=true;try{let f=await loadFrame(i);if(token!==state.request)return;state.frame=i;current=f;$('time').value=i;updateText();await Promise.all([renderScene(),projectionPlots()]);if(state.probe)$('probeValue').textContent=current.has_estimate?'C = '+fmt(sampleProbe(state.probe))+' '+D.unit:'无有效浓度估计';error('');window.gasViewerReady=true;
 if(D.manifest[0].asset){for(const k of Object.keys(window.GAS_TOMO_FRAMES))if(Math.abs(Number(k)-i)>1)delete window.GAS_TOMO_FRAMES[k];}
 }catch(e){error(e.message);console.error(e);}finally{if(token===state.request)state.busy=false;}}
async function update(){try{await renderScene();error('');}catch(e){error(e.message);console.error(e);}}
$('unitLabel').textContent='浓度单位：'+D.unit+'。'+((D.initial.metadata.units||{}).calibrated?'已声明标定；请核对标定适用范围。':'未标定：相对响应重建。');$('levelsInfo').textContent='固定层阈值：'+D.levels.map(fmt).join(' / ')+'（全序列统一）';$('cmin').value=colors[0];$('cmax').value=colors[1];$('isoValue').value=iso;$('isoSlider').value=colors[1]>0?iso/colors[1]*1000:0;$('time').max=D.manifest.length-1;
[['x',nx,X],['y',ny,Y],['z',nz,Z]].forEach(([axis,n,centers])=>{let slider=$('s'+axis);slider.max=n-1;slider.value=state['s'+axis];$('q'+axis).value=centers[Math.floor(n/2)].toFixed(6);function change(){state['s'+axis]=Number(slider.value);$(axis+'pos').textContent=centers[state['s'+axis]].toFixed(4);if(current){if(!['slices','combined'].includes(state.mode)){state.mode='slices';$('mode').value='slices';}update();}}slider.oninput=change;$(axis+'pos').textContent=centers[state['s'+axis]].toFixed(4);});
$('mode').onchange=()=>{state.mode=$('mode').value;update();};$('opacity').oninput=()=>{$('opacityValue').textContent=Number($('opacity').value).toFixed(2);};$('opacity').onchange=update;$('opacityValue').textContent=$('opacity').value;
function setIso(v){if(!Number.isFinite(v)||v<0){error('阈值必须为有限非负数。');return;}iso=v;$('isoValue').value=v;$('isoSlider').value=colors[1]>0?v/colors[1]*1000:0;state.mode='custom';$('mode').value='custom';update();}
$('isoValue').onchange=()=>setIso(Number($('isoValue').value));$('isoSlider').oninput=()=>{$('isoValue').value=Number($('isoSlider').value)*colors[1]/1000;};$('isoSlider').onchange=()=>setIso(Number($('isoSlider').value)*colors[1]/1000);
$('applyColor').onclick=()=>{let a=Number($('cmin').value),b=Number($('cmax').value);if(!Number.isFinite(a)||!Number.isFinite(b)||a>=b){error('色标范围必须为递增有限数值。');return;}colors=[a,b];update();};
$('query').onclick=async()=>{try{let p=['x','y','z'].map(a=>Number($('q'+a).value));let v=sampleProbe(p);if(v===null){$('probeValue').textContent='无有效浓度估计';return;}state.probe=p;$('probeValue').textContent='C = '+fmt(v)+' '+D.unit;await Promise.all([renderScene(),projectionPlots()]);error('');}catch(e){error(e.message);}};
$('time').oninput=()=>showFrame(Number($('time').value));$('prev').onclick=()=>showFrame(Math.max(0,state.frame-1));$('next').onclick=()=>showFrame(Math.min(D.manifest.length-1,state.frame+1));
function stop(){state.playing=false;if(timer)clearTimeout(timer);$('play').textContent='播放';}
async function tick(){if(!state.playing)return;if(state.frame>=D.manifest.length-1){stop();return;}await showFrame(state.frame+1);if(state.playing)timer=setTimeout(tick,Math.max(100,Number($('interval').value)||500));}
$('play').onclick=()=>{if(state.playing){stop();return;}if(D.manifest.length<2)return;state.playing=true;$('play').textContent='暂停';if(state.frame===D.manifest.length-1)showFrame(0).then(tick);else tick();};
if(D.manifest.length===1)$('play').disabled=true;
if(!webglAvailable){for(const option of $('mode').options)if(!['iso','slices'].includes(option.value))option.disabled=true;$('isoValue').disabled=true;$('isoSlider').disabled=true;}
window.gasTomoDebug={sampleProbe,project,showFrame,state,config:D,webglAvailable};
if(typeof Plotly==='undefined')error('Plotly 未加载：请检查离线资源或网络设置。');else showFrame(0);
