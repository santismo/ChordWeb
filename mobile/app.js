const $=id=>document.getElementById(id);
const DB={keys:[],data:{},stats:{sheets:0,rows:0,used:0,chords:0}};
let CURRENT=[];
let activeSound=0;
let audioCtx=null;

const noteToSemi={C:0,'B#':0,'C#':1,Db:1,D:2,'D#':3,Eb:3,E:4,Fb:4,F:5,'E#':5,'F#':6,Gb:6,G:7,'G#':8,Ab:8,A:9,'A#':10,Bb:10,B:11,Cb:11};
const CANON=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
const romanRe=/^[b♭]?(?:I|II|III|IV|V|VI|VII)°?$/i;
const qualities={maj7:[0,4,7,11],min7:[0,3,7,10],dom7:[0,4,7,10],dim7:[0,3,6,9],min7b5:[0,3,6,10],maj9:[0,4,7,11,14],min9:[0,3,7,10,14],dom9:[0,4,7,10,14],six:[0,4,7,9],min6:[0,3,7,9]};
const intentHints={
  any:[],
  jazz:['ii','v','tritone','altered','dominant','lydian','melodic','minor','dorian','mixolydian','backdoor','maj7','dom7','min7'],
  gospel:['iv','bVII','♭VII','plagal','sus','6','9','13','major','mixolydian','secondary'],
  neo:['maj9','min9','m9','sus','11','13','lydian','dorian','slash','add9','min7','maj7'],
  dream:['lydian','maj7','maj9','add9','6','ionian','dorian','sus','parallel'],
  dark:['phrygian','locrian','minor','aeolian','dim','ø','min7b5','bII','♭II','bVI','♭VI'],
  modal:['dorian','phrygian','lydian','mixolydian','aeolian','locrian','ionian'],
  weird:['tritone','chromatic','mediant','altered','locrian','dim','whole','aug','bII','♭II','side'],
  clean:['ionian','major','maj7','min7','v','ii','vi']
};
const sounds=[
  {name:'Soft Piano',type:'triangle',attack:.012,release:.9,filter:2400,detune:0,chorus:.003},
  {name:'Toy Keys',type:'square',attack:.004,release:.55,filter:1800,detune:7,chorus:.002},
  {name:'Warm Pad',type:'sawtooth',attack:.08,release:1.4,filter:900,detune:-5,chorus:.006},
  {name:'Glass Organ',type:'sine',attack:.02,release:1.0,filter:3200,detune:12,chorus:.004},
  {name:'Retro Poly',type:'sawtooth',attack:.01,release:.7,filter:1500,detune:9,chorus:.005},
  {name:'Music Box',type:'triangle',attack:.002,release:.65,filter:4200,detune:0,chorus:.001},
  {name:'Muted Pluck',type:'square',attack:.001,release:.38,filter:1100,detune:-8,chorus:.001},
  {name:'Dream Sample-ish',type:'sine',attack:.04,release:1.2,filter:1300,detune:18,chorus:.008}
];
function clean(s){return (s||'').replace(/\s+/g,' ').trim()}
function displaySubLabel(s){return clean(s).replace(/^All\s+/i,'')}
function esc(s){return clean(s).replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]))}
function parseLabel(label){
  const parts=clean(label).replace(/♯/g,'#').replace(/♭/g,'b').split(/\s+/).filter(Boolean);
  if(!parts.length)return ['C','maj7'];
  const qTok=parts.find(p=>/^(maj9|m9|min9|dom9|maj7|dom7|min7b5|min7|dim7|m7b5|ø7|7|6|m6)$/i.test(p));
  let q=null;
  if(qTok){const x=qTok.toLowerCase();q=x==='m7b5'||x==='ø7'?'min7b5':x==='7'?'dom7':x==='6'?'six':x==='m6'?'min6':x==='m9'?'min9':x;}
  let rootTokens=parts;
  if(romanRe.test(parts[parts.length-1]||''))rootTokens=parts.slice(0,-1);
  let root=rootTokens.filter(t=>!t.match(/^(maj9|m9|min9|dom9|maj7|dom7|min7b5|min7|dim7|m7b5|ø7|7|6|m6)$/i)).join('');
  root=root.match(/[A-G](?:#|b)?/)?.[0]||'C';
  if(!q){q=rootTokens.join('').endsWith('m')?'min7':'maj7';root=root.replace(/m$/,'');}
  return [root,q];
}
function chordSemis(label){const [r,q]=parseLabel(label);const base=noteToSemi[r]??0;return (qualities[q]||qualities.maj7).map(i=>base+i)}
function midiName(n){return CANON[((n%12)+12)%12]+(Math.floor(n/12)-1)}
function voiceChord(label){
  const pcs=chordSemis(label).map(n=>((n%12)+12)%12);
  const [root,q]=parseLabel(label);const rootPc=noteToSemi[root]??0;
  let notes=pcs.map((pc,i)=>({pc,role:i===0?'R':i===1?'3':i===2?'5':i===3?'7':'T'}));
  const voiced=[];
  notes.forEach((o,i)=>{let n=48+o.pc;while(n<52)n+=12;while(n>76)n-=12;if(i&&n<=voiced[voiced.length-1].midi)n+=12;voiced.push({...o,midi:n});});
  if(voiced[0].pc!==rootPc)voiced.unshift({pc:rootPc,role:'R',midi:36+rootPc});
  while(voiced[0].midi>45)voiced[0].midi-=12;
  const compact=voiced.slice(0,5).sort((a,b)=>a.midi-b.midi);
  const fingers=fingeringFor(compact,q);
  return compact.map((n,i)=>({...n,finger:fingers[i]||''}));
}
function fingeringFor(notes,q){
  if(notes.length<=3)return ['5','1','3'];
  if(notes.length===4)return ['5','1','2','5'];
  if(q.includes('9'))return ['5','1','2','3','5'];
  return ['5','1','2','3','5'];
}
function ingestXLSX(wb){
  const data={};let rows=0,used=0,chords=0;
  (wb.SheetNames||[]).forEach(sheet=>{const ws=wb.Sheets[sheet];if(!ws)return;const arr=XLSX.utils.sheet_to_json(ws,{header:1,raw:true,blankrows:false});if(!arr||arr.length<2)return;const key=clean(sheet);let lastBase='',lastSub='';rows+=arr.length-1;for(let r=1;r<arr.length;r++){const row=arr[r]||[];const baseCell=clean(row[1]);if(baseCell)lastBase=baseCell;const subCell=clean(row[2]);if(subCell)lastSub=subCell;const family=clean(row[3]);if(!lastBase||!lastSub)continue;const degrees=[];for(let c=4;c<600;c++){const v=clean(row[c]);if(v)degrees.push(v)}if(!degrees.length)continue;used++;chords+=degrees.length;(((data[key]||={})[lastBase]||={})[lastSub]||=[]).push({family,degrees});}});
  DB.keys=Object.keys(data).sort();DB.data=data;DB.stats={sheets:(wb.SheetNames||[]).length,rows,used,chords};
}
async function loadBuffer(ab){const wb=XLSX.read(ab,{type:'array'});ingestXLSX(wb);localStorage.setItem('cwMobileHasLoaded','1');populateSelectors();$('loaderPanel').classList.add('hidden');$('dataStatus').textContent=`${DB.keys.length} keys · ${DB.stats.chords} chords from data`;generate();}
async function loadURL(url){const res=await fetch(url);if(!res.ok)throw new Error(`${res.status} ${res.statusText}`);await loadBuffer(await res.arrayBuffer());}
async function tryAutoload(){const q=new URLSearchParams(location.search).get('xlsx');const tries=[q,'../data.xlsx','../ChordWeb.xlsx','../chordweb.xlsx'].filter(Boolean);for(const u of tries){try{await loadURL(u);$('dataUrl').value=u;return true}catch(e){}}return false;}
function populateSelectors(){const roots=$('rootSel');roots.innerHTML='<option>Any Key</option>'+DB.keys.map(k=>`<option>${esc(k)}</option>`).join('');refreshBaseSub();}
function refreshBaseSub(){const root=$('rootSel').value;const base=$('baseSel');const sub=$('subSel');let bases=[];if(root&&root!=='Any Key')bases=Object.keys(DB.data[root]||{});else DB.keys.forEach(k=>bases.push(...Object.keys(DB.data[k]||{})));bases=[...new Set(bases)].sort();base.innerHTML='<option>Any Base</option>'+bases.map(b=>`<option>${esc(b)}</option>`).join('');sub.innerHTML='<option>Any Sub</option>';refreshSubs();}
function refreshSubs(){const root=$('rootSel').value,base=$('baseSel').value;let subs=[];const keys=root==='Any Key'?DB.keys:[root];keys.forEach(k=>{const bases=base==='Any Base'?Object.keys(DB.data[k]||{}):[base];bases.forEach(b=>subs.push(...Object.keys((DB.data[k]||{})[b]||{})))});subs=[...new Set(subs)].sort((a,b)=>displaySubLabel(a).localeCompare(displaySubLabel(b)));$('subSel').innerHTML='<option>Any Sub</option>'+subs.map(s=>`<option value="${esc(s)}">${esc(displaySubLabel(s))}</option>`).join('');}
function pool(){const root=$('rootSel').value,base=$('baseSel').value,sub=$('subSel').value,intent=$('intentSel').value;const keys=root==='Any Key'?DB.keys:[root];const out=[];keys.forEach(k=>{const bases=base==='Any Base'?Object.keys(DB.data[k]||{}):[base];bases.forEach(b=>{const subs=sub==='Any Sub'?Object.keys((DB.data[k]||{})[b]||{}):[sub];subs.forEach(s=>{((DB.data[k]||{})[b]||{})[s]?.forEach(row=>row.degrees.forEach(label=>out.push({label:clean(label),key:k,base:b,sub:s,family:row.family})))})})});return weightedIntent(out,intent);}
function weightedIntent(items,intent){if(intent==='any')return items;const hints=intentHints[intent]||[];const scored=items.map(o=>{const hay=`${o.label} ${o.base} ${o.sub} ${o.family}`.toLowerCase();let score=1;hints.forEach(h=>{if(hay.includes(h.toLowerCase()))score+=3});return {...o,score}});const good=scored.filter(o=>o.score>1);return good.length>=6?good:scored;}
function pickWeighted(arr){const total=arr.reduce((a,o)=>a+(o.score||1),0);let r=Math.random()*total;for(const o of arr){r-=o.score||1;if(r<=0)return o}return arr[arr.length-1];}
function generate(){const p=pool();if(!p.length){$('cards').innerHTML='<article class="card"><div class="chordName">No chords</div><p class="meta">Try a wider Key/Base/Sub choice or load data.</p></article>';return;}const seen=new Set();CURRENT=[];let guard=0;while(CURRENT.length<6&&guard++<200){const x=pickWeighted(p);const k=x.label+'|'+x.key+'|'+x.base;if(!seen.has(k)){seen.add(k);CURRENT.push(x)}}renderCards();}
function renderCards(){const cards=$('cards');cards.innerHTML=CURRENT.map((c,i)=>{const v=voiceChord(c.label);return `<article class="card" data-i="${i}"><div class="cardTop"><div><div class="chordName">${esc(c.label)}</div><div class="meta">${esc(c.key)} · ${esc(c.base)}<br>${esc(displaySubLabel(c.sub))}${c.family?' · '+esc(c.family):''}</div></div><button class="playBtn" data-play="${i}">▶</button></div>${keyboardSVG(v)}<div class="voicing">${v.map(n=>`<div class="tone"><b>${midiName(n.midi)}</b><span>${n.role} · finger ${n.finger}</span></div>`).join('')}</div><div class="fingers"><div><b>LH</b> ${leftHand(v)}</div><div><b>RH</b> ${rightHand(v)}</div></div><div class="explain">Fingerings are suggested mobile piano finger numbers generated from the loaded chord label and voicing.</div></article>`}).join('');cards.querySelectorAll('[data-play]').forEach(b=>b.addEventListener('click',()=>playChord(CURRENT[+b.dataset.play])));}
function leftHand(v){return v.slice(0,1).map(n=>`${n.finger} on ${midiName(n.midi)}`).join(', ')}
function rightHand(v){return v.slice(1).map(n=>`${n.finger} on ${midiName(n.midi)}`).join(', ')}
function keyboardSVG(v){const start=48,white=[0,2,4,5,7,9,11],black=[1,3,6,8,10],W=28,H=122;let whiteIndex={};let x=0;let html=`<svg class="keyboard" viewBox="0 0 ${W*15} ${H}" role="img" aria-label="Piano fingering">`;for(let n=start;n<=72;n++){const pc=n%12;if(white.includes(pc)){whiteIndex[n]=x;html+=`<rect class="whiteKey" x="${x*W}" y="0" width="${W}" height="${H}" rx="3"/>`;x++;}}for(let n=start;n<=72;n++){const pc=n%12;if(black.includes(pc)){let prev=n-1;while(!whiteIndex.hasOwnProperty(prev))prev--;const bx=whiteIndex[prev]*W+W*.64;html+=`<rect class="blackKey" x="${bx}" y="0" width="${W*.68}" height="${H*.62}" rx="3"/>`;}}v.forEach(n=>{let midi=n.midi;while(midi<start)midi+=12;while(midi>72)midi-=12;const pc=midi%12;let cx,cy;if(white.includes(pc)){cx=whiteIndex[midi]*W+W/2;cy=H-31}else{let prev=midi-1;while(!whiteIndex.hasOwnProperty(prev))prev--;cx=whiteIndex[prev]*W+W*.98;cy=42}html+=`<circle class="noteMark ${n.role==='R'?'root':''}" cx="${cx}" cy="${cy}" r="12"/><text class="noteText" x="${cx}" y="${cy}">${n.role}</text><circle class="fingerDot" cx="${cx}" cy="${cy-24}" r="10"/><text class="fingerText" x="${cx}" y="${cy-24}">${n.finger}</text>`;});return html+'</svg>';}
function ensureAudio(){if(!audioCtx)audioCtx=new (window.AudioContext||window.webkitAudioContext)();if(audioCtx.state==='suspended')audioCtx.resume();return audioCtx;}
function freq(m){return 440*Math.pow(2,(m-69)/12)}
function playChord(chord,when=0,dur=1.25){const ctx=ensureAudio();const s=sounds[activeSound];const gain=ctx.createGain();gain.gain.value=+$('volume').value/100*.22;const filter=ctx.createBiquadFilter();filter.type='lowpass';filter.frequency.value=s.filter;filter.connect(gain).connect(ctx.destination);voiceChord(chord.label).forEach((n,i)=>{const o=ctx.createOscillator();const g=ctx.createGain();o.type=s.type;o.detune.value=s.detune+(i-2)*2;o.frequency.value=freq(n.midi);g.gain.setValueAtTime(0,ctx.currentTime+when);g.gain.linearRampToValueAtTime(1/(i?1.6:2.4),ctx.currentTime+when+s.attack);g.gain.exponentialRampToValueAtTime(.0001,ctx.currentTime+when+dur+s.release);o.connect(g).connect(filter);o.start(ctx.currentTime+when);o.stop(ctx.currentTime+when+dur+s.release+.05);if(s.chorus){const o2=ctx.createOscillator();const g2=ctx.createGain();o2.type=s.type;o2.detune.value=s.detune+9;o2.frequency.value=freq(n.midi);g2.gain.value=s.chorus;o2.connect(g2).connect(filter);o2.start(ctx.currentTime+when);o2.stop(ctx.currentTime+when+dur+s.release+.05);}})}
function playAll(){CURRENT.forEach((c,i)=>playChord(c,i*.68,.68))}
function setSound(i){activeSound=(i+sounds.length)%sounds.length;$('soundBtn').textContent='Sound: '+sounds[activeSound].name;}
function copyText(){const txt=CURRENT.map((c,i)=>`${i+1}. ${c.label} — ${c.key} / ${c.base} / ${displaySubLabel(c.sub)}`).join('\n');navigator.clipboard?.writeText(txt);$('copyBtn').textContent='Copied';setTimeout(()=>$('copyBtn').textContent='Copy',900)}
function scrollCard(dir){$('cards').scrollBy({left:dir*innerWidth*.9,behavior:'smooth'})}
$('fileInput').addEventListener('change',async e=>{const f=e.target.files[0];if(f)await loadBuffer(await f.arrayBuffer())});
$('loadUrlBtn').addEventListener('click',async()=>{try{await loadURL($('dataUrl').value)}catch(e){$('dataStatus').textContent='Load failed: '+e.message}});
$('rootSel').addEventListener('change',()=>{refreshBaseSub();generate()});$('baseSel').addEventListener('change',()=>{refreshSubs();generate()});$('subSel').addEventListener('change',generate);$('intentSel').addEventListener('change',generate);
$('generateBtn').addEventListener('click',generate);$('regenBtn').addEventListener('click',generate);$('playAllBtn').addEventListener('click',playAll);$('copyBtn').addEventListener('click',copyText);$('prevBtn').addEventListener('click',()=>scrollCard(-1));$('nextBtn').addEventListener('click',()=>scrollCard(1));$('soundBtn').addEventListener('click',()=>setSound(activeSound+1));$('randomSoundBtn').addEventListener('click',()=>setSound(Math.floor(Math.random()*sounds.length)));
setSound(0);tryAutoload();
