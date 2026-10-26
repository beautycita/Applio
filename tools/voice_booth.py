"""Voice booth: record voiceover takes in a browser on another tailnet machine.

The talent opens the page on her own laptop (mic plugged in there), records,
and every take lands here as a WAV. The browser only allows the mic over
HTTPS, so the page is published through `tailscale serve`:

    .venv/bin/python tools/voice_booth.py --out assets/recordings/comercial
    tailscale serve --bg --set-path /voz http://127.0.0.1:8765
    -> https://<this-machine>.<tailnet>.ts.net/voz/

The script is <out>/guion.json (a list of {id, titulo, texto, nota} windows,
~12-15 s each) or plain guion.txt; edits show up within a few seconds. Tailnet-only: nothing here is public.
"""

import argparse
import datetime
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cabina de voz</title>
<style>
:root{--bg:#1b1418;--panel:#271d23;--fg:#f6eef2;--dim:#b7a3ad;--pink:#ff5fa2;--ok:#5fd38a;--bad:#ff4d4d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px;display:grid;gap:16px}
h1{margin:0;font-size:22px}h1 span{color:var(--pink)}
section{background:var(--panel);border-radius:12px;padding:16px;display:grid;gap:12px}
label{color:var(--dim);font-size:13px;text-transform:uppercase;letter-spacing:.06em}
select,input{width:100%;padding:10px;border-radius:8px;border:1px solid #4a3a43;background:#120d10;color:var(--fg);font-size:16px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{border:0;border-radius:999px;padding:14px 28px;font-size:18px;font-weight:700;cursor:pointer}
#rec{background:var(--pink);color:#fff;min-width:200px}#rec.on{background:var(--bad);animation:p 1s infinite}
@keyframes p{50%{opacity:.7}}
button.sec{background:#3a2c34;color:var(--fg);font-size:15px;padding:10px 18px}
#meter{height:22px;background:#120d10;border-radius:6px;overflow:hidden;position:relative}
#bar{height:100%;width:0;background:var(--ok);transition:width .05s}
#peak{position:absolute;top:0;bottom:0;width:3px;background:#fff}
#win{background:#120d10;border:2px solid var(--pink);border-radius:12px;padding:18px;display:grid;gap:12px}
#guion{white-space:pre-wrap;font-size:30px;line-height:1.45}
#nota{color:#f2c94c;font-size:15px;font-style:italic}
#segt{color:var(--pink);font-size:18px}
#room{background:#120d10;border-radius:10px;padding:12px;display:grid;gap:10px}
#chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.chip{border-radius:8px;padding:8px 10px;background:#271d23;border-left:5px solid #555}
.chip b{display:block;font-size:13px;color:var(--dim);font-weight:600}.chip span{font-size:17px;font-weight:700}
.chip.g{border-color:var(--ok)}.chip.y{border-color:#f2c94c}.chip.r{border-color:var(--bad)}
#tips div{padding:6px 0 6px 10px;border-left:3px solid var(--pink);margin:4px 0;font-size:16px}
#tips div b{color:var(--pink)}
#time{font:600 28px ui-monospace,monospace}
#takes div{display:flex;gap:10px;align-items:center;padding:6px 0;border-bottom:1px solid #3a2c34;flex-wrap:wrap}
#takes audio{height:36px}#fout a{color:var(--pink);font-weight:600}#fout div{padding:4px 0}#fout .warn{color:#f2c94c}.dim{color:var(--dim)}#msg{min-height:1.4em}
</style></head><body><main>
<h1>Cabina de voz <span>BeautyCita</span></h1>
<section>
  <div><label>Micrófono</label><select id="mic"></select><div id="mickind" style="margin-top:6px;font-size:15px"></div></div>
  <details id="lav" open><summary style="cursor:pointer;color:var(--pink);font-weight:700">Cómo poner el micrófono de solapa</summary>
        <div style="display:grid;gap:4px;margin-top:8px;font-size:15px">
      <div>• Al centro del pecho, a <b>un palmo de la boca</b> (15–20 cm), con la cabeza del micrófono hacia arriba.</div>
      <div>• Por fuera de la ropa o asomado del escote: si queda bajo la tela, la voz sale opaca y se oye el roce.</div>
      <div>• Nada que lo toque: collares, aretes largos, cabello, la mano. Cada roce se graba.</div>
      <div>• Haz un pequeño lazo con el cable debajo del clip y pégalo con cinta a la ropa: así el cable no jala ni hace ruido al moverte.</div>
      <div>• Al leer, gira la cabeza lo menos posible; el micrófono se queda donde está, tu boca se mueve.</div>
    </div></details>
  <div><label>Nombre de la toma</label><input id="name" value="toma"></div>
  <div><label>Nivel (verde bien · amarillo fuerte · rojo se satura)</label><div id="meter"><div id="bar"></div><div id="peak"></div></div></div>
  <div id="room">
    <div class="row" style="justify-content:space-between"><label>Calidad de la sala (en vivo)</label>
      <button class="sec" id="rreset" style="padding:6px 14px;font-size:13px">↻ Medir de nuevo</button></div>
    <div id="chips"></div>
    <div id="tips"></div>
    <div id="rprev" class="dim" style="font-size:13px"></div>
  </div>
  <div id="win">
    <div class="row" style="justify-content:space-between"><b id="segt"></b><span id="segn" class="dim"></span></div>
    <div id="guion" class="dim">(sin guion todavía)</div>
    <div id="nota"></div>
    <div class="row"><button class="sec" id="prev">◀ Anterior</button><button class="sec" id="next">Siguiente ▶</button><span id="segs" class="dim"></span></div>
  </div>
  <div class="row"><button id="rec">● Grabar</button><span id="time">0:00</span><span id="msg" class="dim"></span></div>
</section>
<section id="flow"><label>Auto-flow: limpiar · ordenar ventanas · voz · pulir</label>
  <div class="row"><select id="voz" style="width:auto"><option value="US_Female_1">Voz: US_Female_1 (pop femenina)</option><option value="none">Voz: la original de Quetzaly</option></select>
  <button id="go" class="sec" style="background:var(--pink);color:#fff">▶ Procesar todas las tomas</button></div>
  <div id="fmsg" class="dim">Usa todas las tomas grabadas; si una ventana se leyó varias veces, gana la más reciente. Tarda 3–5 min.</div>
  <div id="fout"></div>
</section>
<section><label>Tomas guardadas</label><div id="takes" class="dim">Ninguna todavía.</div></section>
</main>
<script>
const $=id=>document.getElementById(id);
let ctx,src,node,stream,chunks=[],recording=false,t0=0,peakHold=0;
function msg(t){$('msg').textContent=t}
async function openMic(id){
  if(stream)stream.getTracks().forEach(t=>t.stop());
  // Raw voice: no browser echo-cancel/denoise/auto-gain; we clean up afterwards.
  stream=await navigator.mediaDevices.getUserMedia({audio:{deviceId:id?{exact:id}:undefined,
    echoCancellation:false,noiseSuppression:false,autoGainControl:false,channelCount:1}});
  if(!ctx){ctx=new AudioContext({sampleRate:48000});
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([`
      class T extends AudioWorkletProcessor{process(i){if(i[0]&&i[0][0])this.port.postMessage(i[0][0].slice());return true}}
      registerProcessor('tap',T)`],{type:'text/javascript'})));}
  if(src)src.disconnect();
  stream.getAudioTracks()[0].onended=()=>{
    if(recording)$('rec').onclick();   // stop and save what was recorded so far
    $('mickind').innerHTML='<span style="color:var(--bad);font-weight:700">⚠ Se desconectó el micrófono. Vuelve a conectarlo; la página lo toma sola.</span>';
  };
  src=ctx.createMediaStreamSource(stream);
  if(!node){node=new AudioWorkletNode(ctx,'tap');node.port.onmessage=e=>onBlock(e.data);}
  src.connect(node);
  if(!an){an=ctx.createAnalyser();an.fftSize=4096;an.smoothingTimeConstant=0;fbuf=new Float32Array(an.frequencyBinCount)}
  src.connect(an);
  const devs=await listMics();
  if(id)store.set('voz.mic',id);
  else{const best=pickMic(devs);const cur=stream.getAudioTracks()[0].getSettings().deviceId;
    if(best&&best!==cur)return openMic(best);}
}
// Far-field mics (the Owl, laptop arrays, webcams) record the room; the USB
// lavalier records the voice. Prefer the lavalier and remember the choice.
const FAR=/owl|array|matriz|webcam|c[aá]mara|camera|realtek|intel|internal|interno|integrad|built-in|predeterminado|default|comunicaciones|communications/i;
const store={get:k=>{try{return localStorage.getItem(k)}catch(e){return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}}};
function micLabel(){return stream?(stream.getAudioTracks()[0].label||''):''}
function isLav(){return !FAR.test(micLabel())}
async function listMics(){
  const cur=stream&&stream.getAudioTracks()[0].getSettings().deviceId;
  const devs=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audioinput');
  $('mic').innerHTML=devs.map(d=>`<option value="${d.deviceId}" ${d.deviceId===cur?'selected':''}>${d.label||'Micrófono'}</option>`).join('');
  const lab=micLabel();
  $('mickind').innerHTML=!stream?'':isLav()
    ?`<span style="color:var(--ok)">✓ Grabando con: ${lab.replace(/</g,'&lt;')}</span>`
    :`<span style="color:var(--bad);font-weight:700">⚠ Esto parece un micrófono lejano (${lab.replace(/</g,'&lt;')}). Conecta el de solapa y elígelo en la lista.</span>`;
  return devs;
}
function pickMic(devs){
  const saved=store.get('voz.mic');
  if(saved&&devs.some(d=>d.deviceId===saved))return saved;
  const lav=devs.find(d=>d.label&&d.deviceId!=='default'&&d.deviceId!=='communications'&&!FAR.test(d.label));
  return lav?lav.deviceId:null;
}
navigator.mediaDevices.addEventListener('devicechange',async()=>{
  const dead=stream&&stream.getAudioTracks()[0].readyState==='ended';
  const devs=dead?(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audioinput'):await listMics();
  // the mic came back, or a lavalier was plugged in after a far mic opened
  if(!recording&&stream&&(dead||!isLav())){const id=pickMic(devs);
    if(id||dead)openMic(id||undefined).then(()=>msg('Micrófono listo.')).catch(e=>msg(e.message))}
});
// ---- Room analysis: same math as the offline check that measured Q's first takes
// (echo ~0.8 s, noise -57 dBFS, normal tone, no hum). 20 ms frames.
let floorC,an,fbuf,acc=[],tick=0,frames=[],spec=null,specN=0,qspec=null,qN=0,decays=[],clips=0,prevSnap=null;
const FR=960;
function roomReset(){if(frames.length>250)prevSnap=roomStats();frames=[];spec=null;specN=0;qspec=null;qN=0;decays=[];clips=0;renderRoom()}
function pct(a,p){if(!a.length)return NaN;const s=[...a].sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(p*s.length))]}
function med(a){return pct(a,0.5)}
function frameIn(dbv){
  frames.push(dbv);if(frames.length>1500)frames.shift();            // last 30 s
  tick++;const n=frames.length;if(n<40)return;
  if(tick%25===0||floorC===undefined)floorC=pct(frames,0.1);const floor=floorC;
  // decay after a word ends: slope of the fall from -5 to -25 dB below the peak
  const i=n-21,d=frames;
  if(d[i]>floor+25&&d[i+1]<d[i]-1&&d[i-1]<=d[i]&&d[i-2]<=d[i-1]+3&&d[i+19]<d[i]-20){
    const xs=[],ys=[];for(let k=0;k<20;k++){if(d[i+k]<=d[i]-5&&d[i+k]>=d[i]-25){xs.push(k*0.02);ys.push(d[i+k])}}
    if(xs.length>=3){const mx=xs.reduce((a,b)=>a+b)/xs.length,my=ys.reduce((a,b)=>a+b)/ys.length;
      let nu=0,de=0;xs.forEach((x,j)=>{nu+=(x-mx)*(ys[j]-my);de+=(x-mx)**2});decays.push(nu/de);if(decays.length>400)decays.shift()}}
  if(an&&tick%5===0){an.getFloatFrequencyData(fbuf);
    const lin=Array.from(fbuf,v=>Math.pow(10,v/10));
    if(dbv>floor+15){if(!spec)spec=new Float64Array(lin.length);lin.forEach((v,j)=>spec[j]+=v);specN++}
    else if(dbv<floor+3){if(!qspec)qspec=new Float64Array(lin.length);lin.forEach((v,j)=>qspec[j]+=v);qN++}}
  if(tick%50===0)renderRoom();                                        // once a second
}
function band(sp,lo,hi){const hz=ctx.sampleRate/an.fftSize;let s=0,c=0;for(let j=Math.ceil(lo/hz);j<hi/hz;j++){s+=sp[j];c++}return 10*Math.log10(s/c+1e-20)}
function roomStats(){
  const st={};const sp=frames.filter(v=>v>pct(frames,0.1)+15);
  st.floor=pct(frames,0.1);st.voice=sp.length>50?med(sp):NaN;st.clips=clips;
  st.echo=decays.length>=12?-60/med(decays):NaN;
  if(specN>=20){st.low=band(spec,100,300)-band(spec,300,2000);st.pres=band(spec,2000,5000)-band(spec,300,2000)}
  if(qN>=20){const hz=ctx.sampleRate/an.fftSize;let worst=-99;
    for(const h of [60,120,180]){const j=Math.round(h/hz);let nb=0;for(const k of [-8,-7,-6,-5,-4,4,5,6,7,8])nb+=qspec[j+k];nb/=10;
      worst=Math.max(worst,10*Math.log10(Math.max(qspec[j-1],qspec[j],qspec[j+1])/nb))}st.hum=worst}
  return st;
}
function renderRoom(){
  const st=roomStats(),chips=[],tips=[];
  const chip=(name,val,c)=>chips.push(`<div class="chip ${c}"><b>${name}</b><span>${val}</span></div>`);
  if(frames.length<250){$('chips').innerHTML='';$('tips').innerHTML='<div class="dim" style="border:0">Habla normal unos 10 segundos (lee la ventana) para medir la sala…</div>';return}
  // echo
  if(isNaN(st.echo))chip('Eco','midiendo…','');
  else{const e=st.echo,c=e<0.35?'g':e<0.6?'y':'r';chip('Eco',e.toFixed(2)+' s',c);
    if(c!=='g')tips.push([c==='r'?3:2,`<b>Eco${c==='r'?' alto':''}:</b> la voz rebota en paredes duras (por eso suena hueca o metálica). Pon cobijas, edredones o almohadas en las paredes más cercanas y detrás de ti; una toalla o cobija sobre la mesa también ayuda. Lo más rápido: graba dentro de un clóset con ropa colgada, o en la recámara en vez de un cuarto vacío.`]);
    if(e>=0.5)tips.push([2,isLav()?'<b>Sube el micrófono:</b> que quede a un palmo de la boca (15–20 cm), al centro del pecho, no en el cuello de la blusa ni de lado. Con el de solapa bien puesto el eco casi desaparece.':'<b>Usa el micrófono de solapa:</b> estás grabando con un micrófono lejano, por eso se oye el cuarto. Conecta el de solapa y elígelo arriba.'])}
  // noise
  {const f=st.floor,c=f<-60?'g':f<-50?'y':'r';chip('Ruido de fondo',f.toFixed(0)+' dB',c);
    if(c!=='g')tips.push([c==='r'?3:1,`<b>Ruido de fondo${c==='r'?' alto':''}:</b> apaga ventilador, aire acondicionado y refrigerador cercano si puedes; cierra puertas y ventanas; aleja la laptop del micrófono y pon el celular en silencio.`])}
  // level
  {const v=st.voice;if(isNaN(v))chip('Volumen de voz','—','');else{
    const c=st.clips>0?'r':v<-34?'y':v>-12?'y':'g';chip('Volumen de voz',v.toFixed(0)+' dB'+(st.clips?' · satura':''),c);
    if(st.clips)tips.push([3,'<b>Se satura:</b> aléjate un poco o baja el volumen del micrófono en Windows (Configuración › Sonido › Entrada).']);
    else if(v<-34)tips.push([2,isLav()?'<b>Voz baja:</b> los de solapa USB suelen venir bajos. Sube el volumen de entrada en Windows (Configuración › Sonido › Entrada › el micrófono de solapa) hasta que la barra quede verde al hablar normal.':'<b>Voz baja:</b> acércate al micrófono o sube el volumen de entrada en Windows (Configuración › Sonido › Entrada).'])}}
  // tone
  if(st.low!==undefined){let c='g',t='normal';
    if(st.low<0){c='y';t='delgada'}else if(st.low>10){c='y';t='retumba'}else if(st.pres<-20){c='y';t='opaca'}
    chip('Tono',t,c);
    if(t==='delgada')tips.push([2,'<b>Voz delgada:</b> estás lejos o de lado al micrófono. Háblale de frente y más cerca.']);
    if(t==='retumba')tips.push([2,'<b>Retumba:</b> aléjate de las esquinas del cuarto y del micrófono unos centímetros.']);
    if(t==='opaca')tips.push([1,isLav()?'<b>Voz opaca:</b> el micrófono quedó bajo la ropa o mirando hacia abajo. Sácalo por fuera de la tela con la cabeza hacia arriba.':'<b>Voz opaca:</b> que nada tape el micrófono (tela, laptop, mano) y háblale de frente.'])}
  // hum
  if(st.hum!==undefined&&st.hum>6){chip('Zumbido','sí','r');tips.push([3,'<b>Zumbido eléctrico:</b> desconecta el cargador de la laptop mientras grabas y aleja el cable del micrófono de extensiones y cables de corriente.'])}
  if(!tips.length)tips.push([0,'<b>¡Sala lista!</b> Todo en verde: graba con confianza.']);
  tips.sort((a,b)=>b[0]-a[0]);
  $('chips').innerHTML=chips.join('');$('tips').innerHTML=tips.map(t=>'<div>'+t[1]+'</div>').join('');
  if(prevSnap&&!isNaN(prevSnap.echo)&&!isNaN(st.echo))$('rprev').textContent=`Antes: eco ${prevSnap.echo.toFixed(2)} s, ruido ${prevSnap.floor.toFixed(0)} dB → ahora: eco ${st.echo.toFixed(2)} s, ruido ${st.floor.toFixed(0)} dB`;
}
$('rreset').onclick=roomReset;
function onBlock(b){
  for(const v of b){if(Math.abs(v)>0.98)clips++;acc.push(v*v);
    if(acc.length>=FR){frameIn(10*Math.log10(acc.reduce((a,c)=>a+c,0)/acc.length+1e-12));acc=[]}}
  let pk=0;for(const s of b){const a=Math.abs(s);if(a>pk)pk=a}
  peakHold=Math.max(pk,peakHold*0.97);
  const db=20*Math.log10(Math.max(pk,1e-6)),w=Math.max(0,Math.min(100,(db+60)/60*100));
  $('bar').style.width=w+'%';$('bar').style.background=db>-1?'var(--bad)':db>-6?'#f2c94c':'var(--ok)';
  $('peak').style.left=Math.max(0,Math.min(99,(20*Math.log10(Math.max(peakHold,1e-6))+60)/60*100))+'%';
  if(recording){chunks.push(b);const s=Math.floor((performance.now()-t0)/1000);$('time').textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0')}
}
function wav(blocks,rate){
  const n=blocks.reduce((a,b)=>a+b.length,0),buf=new ArrayBuffer(44+n*3),v=new DataView(buf);
  const w=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))};
  w(0,'RIFF');v.setUint32(4,36+n*3,true);w(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);
  v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*3,true);v.setUint16(32,3,true);v.setUint16(34,24,true);
  w(36,'data');v.setUint32(40,n*3,true);let o=44;
  for(const b of blocks)for(const s of b){let x=Math.round(Math.max(-1,Math.min(1,s))*8388607);
    v.setUint8(o,x&255);v.setUint8(o+1,(x>>8)&255);v.setUint8(o+2,(x>>16)&255);o+=3}
  return new Blob([buf],{type:'audio/wav'});
}
$('rec').onclick=async()=>{
  if(!ctx){try{await openMic()}catch(e){msg('No se pudo abrir el micrófono: '+e.message);return}}
  await ctx.resume();
  if(!recording){chunks=[];t0=performance.now();recording=true;$('rec').textContent='■ Detener';$('rec').classList.add('on');msg('Grabando…');return}
  recording=false;$('rec').textContent='● Grabar';$('rec').classList.remove('on');
  saveTake(wav(chunks,ctx.sampleRate),$('name').value||'toma');
};
async function saveTake(blob,name){
  msg('Guardando…');
  try{const r=await fetch('/voz/save?name='+encodeURIComponent(name),{method:'POST',body:blob});
    const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);msg('Guardada: '+j.file);loadTakes()}
  catch(e){
    // keep the take in memory: one click retries it, nothing has to be re-read
    $('msg').innerHTML='';const b=document.createElement('button');b.className='sec';b.textContent='↻ Reintentar guardar';
    b.onclick=()=>saveTake(blob,name);$('msg').append('NO se guardó ('+e.message+'). ',b)}
}
window.addEventListener('beforeunload',e=>{if(recording){e.preventDefault();e.returnValue=''}});
$('mic').onchange=e=>{if(recording){msg('Detén la toma antes de cambiar de micrófono.');listMics();return}
  openMic(e.target.value).catch(err=>msg(err.message))};
async function loadTakes(){
  const j=await (await fetch('/voz/takes')).json();
  $('takes').innerHTML=j.length?j.map(f=>`<div><b>${f.file}</b><span class="dim">${f.seconds}s</span><audio controls preload="none" src="/voz/take/${encodeURIComponent(f.file)}"></audio></div>`).join(''):'Ninguna todavía.';
}
let segs=[],cur=0,sig='';
function showSeg(){if(!segs.length)return;const g=segs[cur];
  $('segt').textContent=g.titulo;$('segn').textContent=(cur+1)+' de '+segs.length;
  $('guion').textContent=g.texto;$('guion').classList.remove('dim');$('nota').textContent=g.nota?('Dirección: '+g.nota):'';
  const w=g.texto.split(/\s+/).length;$('segs').textContent='≈ '+Math.round(w/2.5)+' s';
  if(!recording)$('name').value=g.id;}
$('prev').onclick=()=>{if(cur>0){cur--;showSeg()}};$('next').onclick=()=>{if(cur<segs.length-1){cur++;showSeg()}};
async function loadScript(){try{const r=await fetch('/voz/guion');if(!r.ok)return;const t=await r.text();if(t===sig)return;sig=t;
  try{segs=JSON.parse(t)}catch(e){segs=t.trim()?[{id:'toma',titulo:'Guion',texto:t}]:[]}
  if(cur>=segs.length)cur=0;showSeg()}catch(e){}}

async function flowState(){
  const j=await (await fetch('/voz/estado')).json();
  $('go').disabled=j.running;$('go').textContent=j.running?'⏳ Procesando…':'▶ Procesar todas las tomas';
  if(j.msg)$('fmsg').textContent=j.msg;
  if(j.error)$('fmsg').innerHTML='<span style="color:var(--bad)">Falló: '+j.error.replace(/</g,'&lt;')+'</span>';
  const r=j.result;
  if(r){const base='/voz/final/'+r.carpeta+'/';
    const link=(f,t)=>`<a href="${base}${f}" download>${t}</a>`;
    let h=`<div><b>Resultado ${r.carpeta}</b> · ${r.duracion}s · ${r.lufs} LUFS · voz: ${r.voz||"original"}</div>`;
    h+=`<div>${link('voz-completa.mp3','⬇ Voz completa MP3')} &nbsp; ${link('voz-completa.wav','⬇ WAV estudio')} &nbsp; ${link('BeautyCita-voz-'+r.carpeta+'.zip','⬇ Todo (zip)')}</div>`;
    h+=`<div><audio controls preload="none" src="${base}voz-completa.mp3"></audio></div>`;
    h+=r.encontradas.map(e=>`<div>${link(e.id+'.wav','⬇ '+e.id)} <span class="dim">${e.titulo} · de ${e.toma} (${e.desde}s)</span></div>`).join('');
    if(r.faltan.length)h+=`<div class="warn">No encontré: ${r.faltan.join(', ')} — vuelve a grabar esas ventanas y procesa otra vez.</div>`;
    $('fout').innerHTML=h;}
  if(j.running)setTimeout(flowState,2000);
}
$('go').onclick=async()=>{const r=await fetch('/voz/procesar?voz='+$('voz').value,{method:'POST'});
  if(!r.ok){$('fmsg').textContent='No se pudo iniciar: '+(await r.text())}flowState()};
flowState();
openMic().catch(e=>msg('Toca Grabar para permitir el micrófono.'));
loadTakes();loadScript();setInterval(loadScript,4000);
</script></body></html>"""


FLOW = {"running": False, "msg": "", "result": None, "error": None}
AUTOFLOW_PY = os.path.expanduser("~/.local/share/remifasol-autoflow/venv/bin/python")


def run_flow(out, voice):
    """Runs tools/voice_autoflow.py and mirrors its `@@` progress lines into FLOW."""
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FLOW.update(running=True, msg="Iniciando…", result=None, error=None)
    try:
        proc = subprocess.Popen([AUTOFLOW_PY, os.path.join(here, "tools", "voice_autoflow.py"), out, "--voz", voice],
                                cwd=here, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in proc.stdout:
            tail = (tail + [line.strip()])[-5:]
            if line.startswith("@@ "):
                ev = json.loads(line[3:])
                FLOW["msg"] = ev.get("msg", "")
                if ev.get("step") == "listo":
                    FLOW["result"] = ev["result"]
                if ev.get("step") == "error":
                    FLOW["error"] = ev.get("msg")
        if proc.wait() != 0 and not FLOW["error"]:
            FLOW["error"] = " | ".join(tail)[-400:]
    except Exception as exc:
        FLOW["error"] = str(exc)
    finally:
        FLOW["running"] = False


def latest_result(out):
    final = os.path.join(out, "final")
    if not os.path.isdir(final):
        return None
    for d in sorted(os.listdir(final), reverse=True):
        p = os.path.join(final, d, "resultado.json")
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return None


class Booth(BaseHTTPRequestHandler):
    out = "."

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _route(self):
        # works whether or not `tailscale serve` strips the /voz prefix
        path = self.path.split("?", 1)[0]
        return re.sub(r"^/voz", "", path) or "/"

    def do_GET(self):
        route = self._route()
        if route == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if route == "/guion":
            text = ""
            for f in ("guion.json", "guion.txt"):
                p = os.path.join(self.out, f)
                if os.path.exists(p):
                    text = open(p, encoding="utf-8").read()
                    break
            return self._send(200, text, "text/plain; charset=utf-8")
        if route == "/takes":
            takes = []
            for f in sorted(os.listdir(self.out)):
                if f.endswith(".wav"):
                    size = os.path.getsize(os.path.join(self.out, f))
                    takes.append({"file": f, "seconds": round((size - 44) / (48000 * 3), 1)})
            return self._send(200, json.dumps(takes[::-1]))
        if route == "/estado":
            state = dict(FLOW)
            if not state["running"] and not state["result"] and not state["error"]:
                state["result"] = latest_result(self.out)
            return self._send(200, json.dumps(state, ensure_ascii=False))
        if route.startswith("/final/"):
            parts = route.split("/")
            if len(parts) == 4 and re.fullmatch(r"\d{8}-\d{6}", parts[2]) and re.fullmatch(r"[\w.-]+", parts[3]):
                p = os.path.join(self.out, "final", parts[2], parts[3])
                if os.path.isfile(p):
                    ctype = {"wav": "audio/wav", "mp3": "audio/mpeg", "zip": "application/zip"}.get(p.rsplit(".", 1)[-1], "application/json")
                    data = open(p, "rb").read()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Content-Disposition", f'inline; filename="{parts[3]}"')
                    self.end_headers()
                    return self.wfile.write(data)
        if route.startswith("/take/"):
            name = os.path.basename(route[len("/take/"):].replace("%20", " "))
            p = os.path.join(self.out, name)
            if name.endswith(".wav") and os.path.isfile(p):
                return self._send(200, open(p, "rb").read(), "audio/wav")
        self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if self._route() == "/procesar":
            import threading
            if FLOW["running"]:
                return self._send(409, '{"error":"ya está procesando"}')
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            voice = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("voz", "US_Female_1")
            if voice != "none" and not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", re.sub(r"[^\w-]", "", voice))):
                return self._send(400, '{"error":"no existe ese modelo de voz"}')
            threading.Thread(target=run_flow, args=(self.out, re.sub(r"[^\w-]", "", voice) or "none"), daemon=True).start()
            return self._send(202, '{"started":true}')
        if self._route() != "/save":
            return self._send(404, '{"error":"not found"}')
        length = int(self.headers.get("Content-Length", 0))
        if not 44 < length < 500 * 1024 * 1024:
            return self._send(400, '{"error":"empty take"}')
        data = self.rfile.read(length)
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return self._send(400, '{"error":"not a wav"}')
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        raw = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("name", "toma")
        from urllib.parse import unquote
        stem = re.sub(r"[^\w-]+", "-", unquote(raw)).strip("-")[:40] or "toma"
        stamp = datetime.datetime.now().strftime("%H%M%S")
        name = f"{stem}-{stamp}.wav"
        with open(os.path.join(self.out, name), "wb") as fh:
            fh.write(data)
        print(f"saved {name} ({length // 1024} KB)", flush=True)
        self._send(200, json.dumps({"file": name}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join("assets", "recordings", "voz"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Booth.out = os.path.abspath(args.out)
    print(f"booth on 127.0.0.1:{args.port}, takes -> {Booth.out}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Booth).serve_forever()


if __name__ == "__main__":
    main()
