// ===========================================================================
// tabs/graph.js — Canvas "spiral galaxy" renderer for the brain graph.
// ---------------------------------------------------------------------------
// Canvas 2D — no WebGL context to exhaust.
//
// Four fundamentals, laid out in JS on load: memory AGE, COMMUNITY, memory
// KIND, and CONNECTEDNESS (degree). A SHAPE (see SHAPES below) decides which
// of them owns position; kind always keeps color and degree always keeps size,
// so switching shape re-arranges the room without changing what anything is.
// Default `galaxy`: radius = age (oldest in the bright core, recent growth out
// on the arms), knots = community. Glow = RECALL HEAT (recency + volume).
// Depth = the brain's main TOPIC axis when the payload carries one (see
// queries/graph.py): each knot sits at its topic's depth, packed as tight as
// the topic is coherent.
//
// The picture grows with the brain. A young brain has no history to spread
// along the spiral, so it is laid out by its connections — a constellation,
// its strongest memories named, its threads drawn — and the galaxy takes the
// shape over as the brain gains size and age span (see `maturity`). Every
// such generosity comes from the rendering, never from invented data, and
// tapers to nothing by production size.
//
// Live layer: a node lights up when the brain actually touched it — a recall
// recognized it, an encode wrote it or linked a new memory to it, or the
// operator clicked it. NOTHING lights on a timer. An idle brain looks idle;
// the slow global `breath` and the camera's drift are the only motion that
// isn't caused by something real.
// Each activation wave is tinted by the STREAM that caused it (shared palette
// with lib/sessions.js), so several sessions thinking at once stay tellable
// apart. A pinned recall or encode lights its memories where they live, with
// their names and the edges among them; hovering a star names it and draws
// its connections.
//
// Lifecycle contract (drives app.js / live.js / recall_panel.js):
//   init() activate() deactivate() destroy() resize()
//   onGraphSearch() onGraphSearchKey() onGraphRefresh()
//   previewRecallOnGraph() clearRecallPreview() pinRecallToGraph() clearHighlight()
// ===========================================================================

import { api } from '/static/lib/api.js';
import bus from '/static/lib/bus.js';
import { escapeHtml } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';
import { STREAM_RGB, sessionHueIndex } from '/static/lib/sessions.js';

// ── aspect palette ──
// Colour slot (HUE / sprite index) per aspect. Which aspect a node type or a
// relation belongs to comes from the live taxonomy, in the /api/graph3d
// payload's `aspects`; an aspect without a slot draws in the default one.
const ASPECT_SLOT={identity_bearing:0,episodic_anchor:1,active_thread:2,lesson_insight:3,generic_relation:4,noise:5,correction_improvement:6,extension_refinement:7,explanation_causation:8,dependency_flow:9,contradiction_conflict:10,validation_evidence:11,hierarchical_structure:12,temporal_sequence:13,survivor_lineage:14,wisdom:15};
const DEF_NODE_ASP=ASPECT_SLOT.lesson_insight, DEF_REL_ASP=ASPECT_SLOT.generic_relation,
      NOISE_ASP=ASPECT_SLOT.noise, GENERIC_ASP=ASPECT_SLOT.generic_relation;

const HUE=[[255,205,110],[255,150,170],[95,208,230],[110,222,165],[110,130,175],[80,92,120],
  [255,110,95],[150,220,150],[185,140,255],[245,190,110],[245,110,180],[130,230,200],
  [140,165,240],[120,182,242],[150,150,160],[255,233,176]];
const NODE_FAMS=[0,3,1,2,6,15,5];
const NF_LABEL={0:'identity',3:'lessons',1:'moments',2:'open threads',6:'corrections',15:'wisdom',5:'scaffolding'};
const KIND_NAME={0:'identity',1:'moment',2:'open thread',3:'lesson',5:'scaffolding',6:'correction',15:'wisdom'};

// ── module state ──
let G=null;                 // laid-out galaxy { N, X,Y,Z, A,D,H, DEG, TITLE, TYPE, COMM, AGE, IDS, idIndex, adj, adjA, edges }
let raw=null;               // last /api/graph3d payload (search fallback, depth-axis captions)
let cv=null, ctx=null, host=null, tip=null, legend=null;
let raf=0, _loadGen=0, _activateTimer=null, _ro=null;
let W=0,H2=0,DPR=1,MIN=0, yaw=0, pitch=0.62, roll=0, zoom=1, panx=0, pany=0, t0=0;
let spin=true, linesMode='auto', colorMode=0, hoverIdx=-1;
let needSort=true, _winWired=false, breath=0, wob=0;
let act=null, front=[];
let _searchQuery='';
const _highlightTier=new Map();     // id -> tier (persistent spotlight; pin/preview)
let _pinnedEventId=null, _previewSnapshot=null, _pendingPin=null;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

// Growth-aware rendering, all set by buildGalaxy from the graph itself.
let _grow=1;                // 0 for a newborn (≤20 memories) → 1 from GROWN on (log scale)
let _boost=1;               // star size multiplier: a small brain's stars get the room there is
let _lineGain=0;            // how bright a young brain's threads draw (0 once the web gets dense)
let _labels=[];             // a young brain's strongest memories, named
let _halos=[], _haloGain=0; // one soft cloud per community while communities are few
const FOCAL=4.2, CX=0.5, CY=0.5, FIT=0.66;
let _zEnds=null, _focal=FOCAL;  // depth-axis ends (when the payload carries depth), perspective
const GROWN=3000;
// Spotlight tiers in precedence order (an id listed twice takes the later
// tier): colour, and whether its memories are named.
const TIERS=[
  {name:'returned',   rgb:[130,190,255]},
  {name:'activation', rgb:[80,255,150]},
  {name:'used',       rgb:[255,255,255], named:true},
  {name:'revised',    rgb:[196,168,255], named:true},
  {name:'created',    rgb:[255,226,150], named:true},
];
const TIER_IX=Object.fromEntries(TIERS.map((t,k)=>[t.name,k]));
const clamp01=v=>Math.max(0,Math.min(1,v));

// ── sprites ──
function sprite(r,g,b){const s=48,c=document.createElement('canvas');c.width=c.height=s;
  const x=c.getContext('2d'),gd=x.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);
  gd.addColorStop(0,`rgba(${Math.min(r+55,255)},${Math.min(g+55,255)},${Math.min(b+55,255)},1)`);
  gd.addColorStop(.18,`rgba(${r},${g},${b},.9)`);gd.addColorStop(.44,`rgba(${r},${g},${b},.18)`);
  gd.addColorStop(1,`rgba(${r},${g},${b},0)`);x.fillStyle=gd;x.fillRect(0,0,s,s);return c;}
let SPR=null, WHITE=null, RED=null;
function ensureSprites(){ if(SPR)return; SPR=HUE.map(h=>sprite(h[0],h[1],h[2])); WHITE=sprite(255,244,214); RED=sprite(255,104,88); }

// ── helpers ──
const NOW = Date.now();
function daysAgo(ts){ if(!ts) return 9999; const s=String(ts).replace(' ','T').replace('Z','+00:00');
  const d=Date.parse(s); if(isNaN(d)) return 9999; return Math.max(0,(NOW-d)/86400000); }
function hstr(s){ let h=2166136261>>>0; s=String(s); for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);} return (h>>>0)/4294967296; }

// ── shapes ────────────────────────────────────────────────────────────────
// Four ways to stand in the same room. The graph has exactly four
// fundamentals to spend — AGE, COMMUNITY, DEGREE, KIND — and a layout is a
// decision about which one gets the strongest visual channel (position).
// Kind always keeps color, so a shape swap never changes what a dot IS.
//
//   galaxy        radius=age, clumps=community   → history: how the brain grew
//   constellations islands=community              → structure: what belongs together
//   strata        rows=age, columns=kind          → chronology: what formed when
//   core          radius=1/degree                 → gravity: what everything hangs off
//
// Each writes X/Y/Z in place and nothing else. Adding a fifth is one entry.
const DISC=0.26;
const SHAPE_LABEL={galaxy:'galaxy', constellations:'islands', strata:'strata', core:'core'};
// Camera each shape wants. A layout that encodes meaning in a flat axis has to
// be SEEN flat — strata's whole point is reading time down the y-axis, and the
// default three-quarter tilt (with roll drifting) destroys exactly that. The
// others are volumetric and read better in motion. Applied on shape change
// only, so the operator's own rotation is never yanked mid-look.
// Each shape also says which growth layers it takes: `relax` (a young brain
// laid out by its connections) and `puffCap` (semantic depth, with how far a
// loose knot may puff out; absent = its axes already carry meaning).
const SHAPE_VIEW={
  galaxy:         {pitch:0.62, roll:0,    spin:true,  relax:true, puffCap:Infinity},
  constellations: {pitch:0.62, roll:0,    spin:true,  puffCap:1},
  strata:         {pitch:0,    roll:0,    spin:false},
  core:           {pitch:0.35, roll:0,    spin:true},
};
const SHAPE_ORDER=['galaxy','constellations','strata','core'];
const SHAPE_KEY='dashboard.graphShape';
let _shape='galaxy';
try { const sv=localStorage.getItem(SHAPE_KEY); if(sv&&SHAPE_ORDER.includes(sv)) _shape=sv; } catch(e){}

const SHAPES={

  // Spiral arms wound by age, each community a knot riding an arm. The
  // original — history as the organizing principle. A young galaxy has few
  // knots and a short history, so its knots take the room there is (`room`)
  // and its arms are as long as it has grown (`arm`).
  galaxy({N,X,Y,Z,AGE,DEG,COMM,cmap,aget,room,arm}){
    const WIND=2.75;
    function spiral(t,armIx,jr,ja,jz){ const rad=0.24+2.4*t*arm, th=armIx*Math.PI+t*WIND*2*Math.PI+(ja-0.5)*(0.13+0.2*t);
      const x=rad*Math.cos(th), y=rad*Math.sin(th), px=-Math.sin(th),py=Math.cos(th), w=(jr-0.5)*(0.05+0.18*t);
      return [x+px*w, y+py*w, jz*DISC*(0.5+0.55*t)*rad]; }
    // community centroids on the spiral (by median member age)
    const cids=[...cmap.keys()], cc={}, cx=[],cy=[];
    for(const c of cids){ const mem=cmap.get(c); const med=mem.map(i=>AGE[i]).sort((a,b)=>a-b)[mem.length>>1];
      const p=spiral(aget(med), hstr(c+'m')<0.5?0:1, hstr(c+'r'), hstr(c+'a'), (hstr(c+'z')-0.5)*2);
      cc[c]=[p[0],p[1],p[2]]; cx.push(p[0]); cy.push(p[1]); }
    // local in-plane repulsion + spring home → distinct knots
    const hx=cx.slice(), hy=cy.slice(), n=cids.length;
    for(let it=0;it<45;it++){ for(let i=0;i<n;i++){ let fx=0,fy=0;
      for(let j=0;j<n;j++){ if(i===j)continue; const dx=cx[i]-cx[j],dy=cy[i]-cy[j],d2=dx*dx+dy*dy;
        if(d2<0.09&&d2>1e-9){ const inv=1/(d2+2e-3); fx+=dx*inv; fy+=dy*inv; } }
      cx[i]+=Math.max(-0.03,Math.min(0.03,fx*0.0016))-(cx[i]-hx[i])*0.1;
      cy[i]+=Math.max(-0.03,Math.min(0.03,fy*0.0016))-(cy[i]-hy[i])*0.1; } }
    for(let k=0;k<n;k++){ cc[cids[k]][0]=cx[k]; cc[cids[k]][1]=cy[k]; }
    const rnd=mulberry(12345);
    for(const c of cids){ const mem=cmap.get(c), ctr=cc[c], ball=(0.03+0.075*Math.log1p(mem.length))*room;
      for(const i of mem){ let dx=randn(rnd),dy=randn(rnd),dz=randn(rnd)*DISC; const L=Math.hypot(dx,dy,dz)||1;
        const cp=0.16*(DEG[i]/(DEG[i]+15)); const rr=ball*(0.35+0.65*rnd());
        X[i]=ctr[0]*(1-cp)+dx/L*rr; Y[i]=ctr[1]*(1-cp)+dy/L*rr; Z[i]=ctr[2]*(1-cp)+dz/L*rr; } }
    // free nodes ride the arms as faint dust
    for(let i=0;i<N;i++){ if(COMM[i]!=null) continue; const p=spiral(aget(AGE[i]), rnd()<0.5?0:1, rnd(), rnd(), randn(rnd)*0.8);
      X[i]=p[0]*1.06; Y[i]=p[1]*1.06; Z[i]=p[2]*1.06; }
  },

  // Communities as separated islands on a phyllotaxis spiral (sunflower
  // packing — even spacing at any count), sized by membership. Age is dropped
  // from position entirely, so what you see is pure belonging: how many
  // narratives there are, how big, and how much loose material floats between.
  constellations({N,X,Y,Z,DEG,COMM,cmap}){
    const cids=[...cmap.keys()], n=cids.length, rnd=mulberry(9871);
    const GA=Math.PI*(3-Math.sqrt(5));   // golden angle
    const DISC_R=2.25;
    // Island radius is derived from the ACTUAL spacing, not a fixed constant.
    // Sunflower packing puts n points ~DISC_R/sqrt(n) apart, so a fixed ball
    // size that reads well at 30 communities merges into a single cloud at
    // 726 (this brain). Tying the ball to the spacing keeps islands separate
    // at any count, and keeps the size ratio between a 160-member narrative
    // and a 6-member one visible.
    const spacing=DISC_R/Math.sqrt(Math.max(1,n));
    let maxMem=1; for(const c of cids) maxMem=Math.max(maxMem,cmap.get(c).length);
    const lm=Math.log1p(maxMem);
    const cc={};
    // Largest islands outward, where the ring has the most room for them.
    const order=cids.slice().sort((a,b)=>cmap.get(a).length-cmap.get(b).length);
    order.forEach((c,k)=>{
      const t=n>1?(k+0.5)/n:0.5, rad=0.22+(DISC_R-0.22)*Math.sqrt(t), th=k*GA;
      cc[c]=[rad*Math.cos(th), rad*Math.sin(th), (hstr(c+'z')-0.5)*DISC*1.1*rad];
    });
    for(const c of cids){ const mem=cmap.get(c), ctr=cc[c];
      const ball=spacing*(0.20+0.30*(Math.log1p(mem.length)/lm));
      for(const i of mem){ let dx=randn(rnd),dy=randn(rnd),dz=randn(rnd)*0.7; const L=Math.hypot(dx,dy,dz)||1;
        // Hubs pull toward their island's center; leaves sit on the rim.
        const cp=0.55*(DEG[i]/(DEG[i]+8)), rr=ball*(1-cp)*(0.35+0.65*rnd());
        X[i]=ctr[0]+dx/L*rr; Y[i]=ctr[1]+dy/L*rr; Z[i]=ctr[2]+dz/L*rr; } }
    // Unplaced memories form the halo — visibly OUTSIDE every narrative, and
    // clear of the outermost island, which is the honest picture of how much
    // the brain holds that no community has claimed.
    for(let i=0;i<N;i++){ if(COMM[i]!=null) continue;
      const th=rnd()*6.2832, rad=DISC_R+0.35+0.5*rnd();
      X[i]=rad*Math.cos(th); Y[i]=rad*Math.sin(th); Z[i]=randn(rnd)*DISC*0.8; }
  },

  // Time as an axis you can read: newest at the top, oldest at the bottom,
  // communities as vertical lanes. The one shape where "when did this happen"
  // is answerable by eye rather than by hover.
  strata({N,X,Y,Z,AGE,DEG,A,aget}){
    const rnd=mulberry(4242);
    // Lanes are KINDS, not communities. Community was the obvious choice and
    // the wrong one: this brain has 726 of them, which is 726 lanes across
    // four units of width — sub-pixel columns, unreadable. There are only
    // seven kinds, they're the same seven the legend already names, and
    // "which kinds of memory did I form, and when" is the question a
    // time axis can actually answer.
    const lanes=NODE_FAMS.length;
    const laneOf=new Map(); NODE_FAMS.forEach((a,k)=>laneOf.set(a,k));
    const laneX=(k)=>(lanes>1?(k/(lanes-1))*4-2:0);
    for(let i=0;i<N;i++){
      const t=aget(AGE[i]);                       // 1 = oldest, 0 = newest
      const k=laneOf.has(A[i])?laneOf.get(A[i]):lanes-1;
      // Hubs sit centered in their column, leaves spread — so the lane still
      // reads as a lane while showing where its weight is.
      const spread=0.30*(1-0.55*(DEG[i]/(DEG[i]+10)));
      X[i]=laneX(k)+randn(rnd)*spread;
      Y[i]=2.2-4.4*(1-t)+randn(rnd)*0.03;
      Z[i]=randn(rnd)*DISC*0.5;
    }
  },

  // Connectedness as gravity: the most-linked memories collapse to the core,
  // orphans drift to the shell. Answers "what is everything hanging off?" —
  // the question the age spiral can't.
  core({N,X,Y,Z,DEG,A}){
    const rnd=mulberry(777);
    let maxDeg=0; for(let i=0;i<N;i++) maxDeg=Math.max(maxDeg,DEG[i]);
    const norm=Math.max(1,Math.sqrt(maxDeg));
    for(let i=0;i<N;i++){
      const pull=Math.sqrt(DEG[i])/norm;          // 1 = biggest hub
      const rad=0.16+2.3*Math.pow(1-pull,1.35);
      // Fibonacci sphere, offset per aspect so kinds band rather than mix —
      // color still reads at the shell where the dots are densest.
      const k=i+0.5, phi=Math.acos(1-2*k/N), th=Math.PI*(1+Math.sqrt(5))*k+A[i]*0.9;
      X[i]=rad*Math.sin(phi)*Math.cos(th)+randn(rnd)*0.02;
      Y[i]=rad*Math.sin(phi)*Math.sin(th)+randn(rnd)*0.02;
      Z[i]=rad*Math.cos(phi)*(0.55+0.45*DISC)+randn(rnd)*0.02;
    }
  },
};

// Mean position of a set of nodes.
function centroidOf(mem,X,Y,Z){ let cx=0,cy=0,cz=0; for(const i of mem){cx+=X[i];cy+=Y[i];cz+=Z[i];} const n=mem.length||1; return [cx/n,cy/n,cz/n]; }
// Largest in-plane radius of the layout.
function discRadius(N,X,Y){ let R=1e-3; for(let i=0;i<N;i++) R=Math.max(R,Math.hypot(X[i],Y[i])); return R; }

// How much history the brain has to show: its size AND its age span. A brain
// that is small, or large but all written in a day, can't say much with a
// spiral; its connections are its structure.
function maturity(N, spanDays){
  const t=Math.min(clamp01(Math.log(Math.max(N,1)/800)/Math.log(4000/800)), clamp01((spanDays-3)/27));
  return t*t*(3-2*t);
}

// One relaxation from the galaxy positions: springs on real edges, local
// repulsion, cohesion to the node's community, and an anchor back to its
// galaxy position whose strength is maturity. m=0 → a constellation shaped by
// connections; m→1 → the galaxy exactly. Repulsion runs on a 2D grid, so the
// cost stays linear in N.
const RELAX_MAX=6000;
function relaxLayout({N,X,Y,Z,cmap,E,m}){
  const AX=X.slice(),AY=Y.slice(),AZ=Z.slice();
  const R=discRadius(N,AX,AY), sp=R*1.5/Math.sqrt(Math.max(N,4)), cut2=4*sp*sp, cell=2*sp;
  const ka=0.03+3*m*m, grav=0.15*sp/R*(1-m), ks=(1-m)*(1-m), kc=0.25*(1-m);
  const ed=[]; for(const [a,b] of E) ed.push(a,b);
  const cidx=new Int32Array(N).fill(-1), cl=[...cmap.keys()]; cl.forEach((c,k)=>{ for(const i of cmap.get(c)) cidx[i]=k; });
  const K=N<=1000?110:N<=3000?70:35, fx=new Float32Array(N),fy=new Float32Array(N),fz=new Float32Array(N);
  const ccx=new Float32Array(cl.length),ccy=new Float32Array(cl.length),ccz=new Float32Array(cl.length),ccn=new Float32Array(cl.length);
  for(let it=0;it<K;it++){ const temp=sp*(0.35*(1-it/K)+0.02); fx.fill(0);fy.fill(0);fz.fill(0);
    const grid=new Map(); for(let i=0;i<N;i++){ const k=Math.floor(X[i]/cell)+','+Math.floor(Y[i]/cell); (grid.get(k)||grid.set(k,[]).get(k)).push(i); }
    for(const [k,cellA] of grid){ const [gx,gy]=k.split(',').map(Number);
      for(let ox=-1;ox<=1;ox++) for(let oy=-1;oy<=1;oy++){ const nbC=grid.get((gx+ox)+','+(gy+oy)); if(!nbC) continue;
        for(const i of cellA) for(const j of nbC){ if(j<=i) continue; const dx=X[i]-X[j],dy=Y[i]-Y[j],dz=Z[i]-Z[j],d2=dx*dx+dy*dy+dz*dz;
          if(d2>cut2||d2<1e-12) continue; const f=sp*sp/(d2+0.02*sp*sp); fx[i]+=dx*f;fy[i]+=dy*f;fz[i]+=dz*f; fx[j]-=dx*f;fy[j]-=dy*f;fz[j]-=dz*f; } } }
    for(let e=0;e<ed.length;e+=2){ const a=ed[e],b=ed[e+1],dx=X[b]-X[a],dy=Y[b]-Y[a],dz=Z[b]-Z[a],d=Math.hypot(dx,dy,dz)+1e-9,f=Math.min(d/sp,3)*ks;
      fx[a]+=dx*f;fy[a]+=dy*f;fz[a]+=dz*f; fx[b]-=dx*f;fy[b]-=dy*f;fz[b]-=dz*f; }
    ccx.fill(0);ccy.fill(0);ccz.fill(0);ccn.fill(0); for(let i=0;i<N;i++){ const k=cidx[i]; if(k<0)continue; ccx[k]+=X[i];ccy[k]+=Y[i];ccz[k]+=Z[i];ccn[k]++; }
    for(let i=0;i<N;i++){ const k=cidx[i]; if(k>=0&&ccn[k]>1){ fx[i]+=(ccx[k]/ccn[k]-X[i])*kc; fy[i]+=(ccy[k]/ccn[k]-Y[i])*kc; fz[i]+=(ccz[k]/ccn[k]-Z[i])*kc; }
      fx[i]+=(AX[i]-X[i])*ka-X[i]*grav; fy[i]+=(AY[i]-Y[i])*ka-Y[i]*grav; fz[i]+=(AZ[i]-Z[i])*ka*2-Z[i]*grav*2;
      const L=Math.hypot(fx[i],fy[i],fz[i]); if(L>temp){ fx[i]*=temp/L;fy[i]*=temp/L;fz[i]*=temp/L; }
      X[i]+=fx[i];Y[i]+=fy[i];Z[i]+=fz[i]; } }
}

// Depth carries the brain's main topic axis: each knot moves to its topic's
// depth (its members keep a little of their own), and a coherent topic packs
// tight while a grab-bag puffs out — up to `puffCap`, so islands that must
// stay separate still do. Free memories take their own depth.
function applyDepth({N,X,Y,Z,COMM,cmap,DEPTH,coh,puffCap}){
  const DZ=0.62*discRadius(N,X,Y);
  for(const [c,mem] of cmap){ const [cx,cy,cz]=centroidOf(mem,X,Y,Z);
    const k=coh[c]?.coherence??0.5, puff=Math.min(puffCap,0.72+0.85*(1-k));
    for(const i of mem){ const dz=Z[i]-cz; X[i]=cx+(X[i]-cx)*puff; Y[i]=cy+(Y[i]-cy)*puff; Z[i]=DZ*DEPTH[i]+dz*0.45*puff; } }
  for(let i=0;i<N;i++){ if(COMM[i]==null) Z[i]=DZ*DEPTH[i]+Z[i]*0.3; }
}

// ── galaxy layout (JS port of build_mind.py) ──
function buildGalaxy(data){
  const nodes=data.nodes||[], edgesRaw=data.edges||[];
  const {node_types:typeAsp={}, edge_relations:relAsp={}}=data.aspects||{};
  const N=nodes.length;
  const IDS=new Array(N), idIndex=new Map();
  for(let i=0;i<N;i++){ IDS[i]=nodes[i].id; idIndex.set(nodes[i].id,i); }
  const A=new Int32Array(N), DEG=new Float32Array(N), AGE=new Float64Array(N), DEPTH=new Float32Array(N);
  const TITLE=new Array(N), TYPE=new Array(N), COMM=new Array(N);
  const acc=new Float64Array(N), rec=new Float64Array(N);
  let hasDepth=false;
  for(let i=0;i<N;i++){ const n=nodes[i];
    A[i]=ASPECT_SLOT[typeAsp[n.type]]??DEF_NODE_ASP;
    TITLE[i]=n.name||n.id; TYPE[i]=n.type||''; COMM[i]=n.community||null;
    acc[i]=n.access_count||1; AGE[i]=daysAgo(n.created_at);
    rec[i]=Math.exp(-daysAgo(n.last_accessed||n.created_at)/45);
    if(typeof n.depth==='number'){ DEPTH[i]=n.depth; hasDepth=true; }
  }
  // Degree counts connections. The payload carries real ones only — noise
  // bookkeeping such as community membership stays server-side (membership
  // arrives as each node's `community`), so no community node poses as the
  // biggest hub in the brain.
  const E=[]; for(const e of edgesRaw){ const a=idIndex.get(e.source), b=idIndex.get(e.target);
    if(a==null||b==null||a===b) continue;
    DEG[a]++; DEG[b]++;
    E.push([a,b,ASPECT_SLOT[relAsp[e.relation]]??DEF_REL_ASP]); }
  // heat
  let maxAcc=1; for(let i=0;i<N;i++) maxAcc=Math.max(maxAcc,acc[i]); const la=Math.log1p(maxAcc);
  const H=new Float32Array(N), D=new Float32Array(N), L=new Float32Array(N);
  // Degree normalizes against THIS graph's max (clamped 3.2–13.4): D drives
  // size and baseline brightness, so a fixed divisor renders a fresh brain
  // tiny and dim. The clamp keeps a big brain's look and stops a 2-edge brain
  // rendering every node maximal.
  let maxDeg=0; for(let i=0;i<N;i++) maxDeg=Math.max(maxDeg,DEG[i]);
  const degNorm=Math.max(3.2,Math.min(13.4,Math.sqrt(maxDeg)||1));
  // A small brain has less TOTAL light on screen — fewer dots means the whole
  // field reads dim even with per-dot brightness fixed. Its stars get the room
  // the canvas has for them (_boost), tapering on a log scale to 1 at GROWN,
  // so the picture brightens as the brain grows instead of staying flat.
  _grow=clamp01(Math.log(Math.max(N,20)/20)/Math.log(GROWN/20));
  _boost=Math.max(1,Math.min(3.6,Math.pow(GROWN/Math.max(N,1),0.27)));
  for(let i=0;i<N;i++){ H[i]=Math.max(0,Math.min(1, 0.35*Math.log1p(acc[i])/la + 0.65*rec[i]));
    D[i]=Math.min(1, Math.sqrt(DEG[i])/degNorm);
    // baseline presence — every memory is here, lit by WHAT IT IS (kind+reach+warmth),
    // not by whether it's firing. this is the "I don't go dark" resting state.
    L[i]=Math.min(0.95, 0.30 + 0.42*H[i] + 0.34*D[i]); }

  // communities — a community's own node belongs inside its knot, not out on
  // the arms: it is that knot's summary, and as a lone outlier it would set
  // the frame for the whole picture.
  const cmap=new Map();
  for(let i=0;i<N;i++){ const c=COMM[i]; if(c==null) continue; (cmap.get(c)||cmap.set(c,[]).get(c)).push(i); }
  for(let i=0;i<N;i++){ if(COMM[i]==null && cmap.has(IDS[i])){ COMM[i]=IDS[i]; cmap.get(IDS[i]).push(i); } }
  // Radius carries AGE: oldest in the bright core, newest out on the arms.
  const ages=Array.from(AGE).filter(a=>a<9999).sort((a,b)=>a-b);
  const lo=ages.length?ages[Math.floor(ages.length*0.03)]:0, hi=ages.length?ages[Math.floor(ages.length*0.99)]:100;
  // A brain born today has NO age span: hi-lo collapses to ~0, every node maps
  // to the same t, and the spiral degenerates into a single ring. Fall back to
  // ordinal position (rank among all ages) when the span is under a day —
  // still age, just ordered rather than scaled, which is the only thing a
  // one-day-old brain can honestly say about its own history.
  const ageRank=new Map();
  if(hi-lo<1 && ages.length){ for(let k=0;k<ages.length;k++) if(!ageRank.has(ages[k])) ageRank.set(ages[k],k/Math.max(1,ages.length-1)); }
  const aget=ageRank.size
    ? (x=>1-(ageRank.has(x)?ageRank.get(x):0.5))
    : (x=>{ x=Math.max(lo,Math.min(hi,x)); return Math.max(0,Math.min(1,1-(x-lo)/Math.max(1e-6,hi-lo))); });
  const X=new Float32Array(N), Y=new Float32Array(N), Z=new Float32Array(N);
  // The shape is a CHOICE over the same four fundamentals — age, community,
  // degree, kind. Each layout below decides which of them owns radius, which
  // owns clustering, and which is left to color and size. Nothing downstream
  // (projection, picking, activation, highlight) knows which one ran.
  const shape=SHAPES[_shape]?_shape:'galaxy';
  SHAPES[shape]({N,X,Y,Z,AGE,DEG,COMM,A,cmap,aget,room:1+1.5*(1-_grow),arm:0.45+0.55*_grow});
  const spec=SHAPE_VIEW[shape];
  if(spec.relax){ const m=maturity(N,hi-lo); if(m<0.999 && N<=RELAX_MAX) relaxLayout({N,X,Y,Z,cmap,E,m}); }
  const depthOn=hasDepth && spec.puffCap!=null;
  if(depthOn) applyDepth({N,X,Y,Z,COMM,cmap,DEPTH,coh:data.communities||{},puffCap:spec.puffCap});
  _focal=depthOn?3.1:FOCAL;
  // recenter + normalize. The frame fits the brain's body (98th percentile
  // radius), not its single outermost dot.
  let mx=0,my=0,mz=0; for(let i=0;i<N;i++){mx+=X[i];my+=Y[i];mz+=Z[i];} mx/=N||1;my/=N||1;mz/=N||1;
  const RR=new Float32Array(N); let mr=1e-3;
  for(let i=0;i<N;i++){X[i]-=mx;Y[i]-=my;Z[i]-=mz;RR[i]=Math.hypot(X[i],Y[i],Z[i]);mr=Math.max(mr,RR[i]);}
  if(N>8){ RR.sort(); mr=Math.max(1e-3,RR[Math.floor(0.98*(N-1))]); }
  for(let i=0;i<N;i++){X[i]/=mr;Y[i]/=mr;Z[i]/=mr;}
  _zEnds=null; if(depthOn&&N>8){ const zs=Float32Array.from(Z).sort(); _zEnds=[zs[Math.floor(0.97*(N-1))],zs[Math.floor(0.03*(N-1))]]; }
  // adjacency (with each connection's aspect) + drawn-edge subset
  const adj=Array.from({length:N},()=>[]), adjA=Array.from({length:N},()=>[]);
  const draw=[]; let ei=0; const every=E.length<1500;
  for(const [a,b,r] of E){
    adj[a].push(b); adj[b].push(a); adjA[a].push(r); adjA[b].push(r);
    const bridge=COMM[a]!==COMM[b]; ei++;
    // bridges always drawn; once the web is dense, intra-community sampled (generic 1-in-4, other 1-in-2)
    if(!bridge && !every){ if(r===GENERIC_ASP){ if(ei%4) continue; } else if(ei%2) continue; }
    draw.push([a,b,r,bridge?1:0]); }
  // A young brain's structure IS its connections: its threads draw bright,
  // fading out as the web gets dense enough to read as a hairball.
  _lineGain=Math.pow(clamp01((5000-draw.length)/4200),2);
  // The strongest memories of a young brain, named — by reach + warmth.
  const k=Math.round(10*Math.pow(1-_grow,1.6));
  _labels=k>0?[...Array(N).keys()].filter(i=>TYPE[i]!=='community').sort((i,j)=>(D[j]+0.5*H[j])-(D[i]+0.5*H[i])).slice(0,k):[];
  // One soft cloud per community, at its members' centroid, sized by their
  // spread and coloured by its most common kind; clouds fade as communities
  // crowd, so overlapping gas never turns to fog.
  _halos=[]; _haloGain=clamp01((160-cmap.size)/130);
  if(_haloGain>0) for(const mem of cmap.values()){ if(mem.length<3) continue; const [cx,cy,cz]=centroidOf(mem,X,Y,Z); let rr=0; for(const i of mem) rr+=(X[i]-cx)**2+(Y[i]-cy)**2+(Z[i]-cz)**2;
    const kc=new Map(); for(const i of mem) kc.set(A[i],(kc.get(A[i])||0)+1); let ka=DEF_NODE_ASP,kn=-1; for(const [kk,v] of kc) if(v>kn&&kk!==NOISE_ASP){ka=kk;kn=v;}
    _halos.push([cx,cy,cz,Math.sqrt(rr/mem.length),ka,mem.length]); }
  return { N, X,Y,Z, A, D, H, L, DEG, TITLE, TYPE, COMM, AGE, IDS, idIndex, adj, adjA, edges:draw,
           C:cmap.size, E_total:E.length };
}
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function randn(R){let u=0,v=0;while(!u)u=R();while(!v)v=R();return Math.sqrt(-2*Math.log(u))*Math.cos(6.2831853*v);}

// ── projection + render ──
let sx,sy,sp,sz,order,fired,hop,tender,streamOf;
function allocScreen(N){ sx=new Float32Array(N);sy=new Float32Array(N);sp=new Float32Array(N);sz=new Float32Array(N);
  order=new Int32Array(N); for(let i=0;i<N;i++)order[i]=i; act=new Float32Array(N); fired=new Uint8Array(N);
  hop=new Int8Array(N); hop.fill(-1); tender=new Float32Array(N);
  // Which stream of thought lit this node. Several streams think at once, and
  // an untinted bloom makes three simultaneous sessions look like one. Index
  // into the SAME palette lib/sessions.js gives the feed, so a wave in the
  // galaxy is visibly the stream whose activity is scrolling beside it.
  // 255 = "no stream" (nothing lit it, or the event carried no session).
  streamOf=new Uint8Array(N); streamOf.fill(255);
  front=[]; _tierAt=new Int8Array(N).fill(-1); _spotDirty=true;
  _hov=null; _hovIdx=-1; _hovShow=-1; _hovT=0; _titleT=0; _tipFor=-1;
  hoverIdx=-1; }   // a rebuilt graph re-indexes its stars; an old index names the wrong one, or none
// The camera drifts in a slow tilt-and-turn while spinning, so the depth is
// always visible as parallax; the operator's drag adds on top.
function viewAngles(){ return [Math.max(-1.5,Math.min(1.5,pitch+0.22*Math.sin(wob*0.21))), yaw+0.42*Math.sin(wob*0.13)]; }
// The frame's camera, set by project() and read by projPt() for the clouds
// and captions drawn after it.
const cam={cr:1,sr:0,cy:1,syw:0,cp:1,spp:0,sc:1};
function project(){ const g=G, [P,Yw]=viewAngles();
  cam.cr=Math.cos(roll); cam.sr=Math.sin(roll); cam.cy=Math.cos(Yw); cam.syw=Math.sin(Yw); cam.cp=Math.cos(P); cam.spp=Math.sin(P); cam.sc=(FIT*MIN)*zoom;
  const {cr,sr,cy,syw,cp,spp,sc}=cam;
  for(let i=0;i<g.N;i++){ let x=g.X[i],y=g.Y[i],z=g.Z[i];
    let xr=x*cr-y*sr, yr=x*sr+y*cr; let rx=xr*cy-z*syw, rz=xr*syw+z*cy; let ry=yr*cp-rz*spp; rz=yr*spp+rz*cp;
    const pe=_focal/(_focal+rz); sx[i]=W*CX+panx+rx*sc*pe; sy[i]=H2*CY+pany+ry*sc*pe; sp[i]=pe; sz[i]=rz; } }
function projPt(x,y,z){ const {cr,sr,cy,syw,cp,spp,sc}=cam;
  let xr=x*cr-y*sr, yr=x*sr+y*cr; let rx=xr*cy-z*syw, rz=xr*syw+z*cy; let ry=yr*cp-rz*spp; rz=yr*spp+rz*cp;
  const pe=_focal/(_focal+rz); return [W*CX+panx+rx*sc*pe, H2*CY+pany+ry*sc*pe, pe]; }
// A gently bowed edge from a to b, on a Path2D or the context itself.
function bow(p,ax,ay,bx,by,k){ const mx=(ax+bx)/2,my=(ay+by)/2; p.moveTo(ax,ay); p.quadraticCurveTo(mx+(ay-by)*k,my+(bx-ax)*k,bx,by); }
// A star's core radius before perspective, activation and emphasis.
const coreR=i=>1.3+2.2*(1-_grow)+G.D[i]*9;

function matches(i){ if(!_searchQuery) return true; const q=_searchQuery;
  return (G.TITLE[i]||'').toLowerCase().includes(q) || (G.TYPE[i]||'').toLowerCase().includes(q); }

const MAXHOP=4;   // a recall lights a bounded neighbourhood, not the whole brain
function seed(i,hue){ if(i<0||i>=G.N)return; act[i]=Math.max(act[i],1); fired[i]=1; hop[i]=0;
  if(hue!=null&&hue>=0) streamOf[i]=hue;
  if(front.indexOf(i)<0)front.push(i); }
// A recall spreads like a signal firing through a network: a lit node CHARGES its
// fresh neighbours up over ~0.7s until they ignite and pass it on, out to MAXHOP
// then it stops and the whole thing fades. Each node fires once per wave (fired[]
// refractory → no feedback, no flash-all), then cools and re-arms. dt-scaled ⇒
// frame-rate independent; the front travels ~1-2 hops/s, blooms ~2s, fades by ~4s.
function stepActivation(dt){ if(!front.length)return; const nf=[], add=new Map(), hm=new Map(), sm=new Map();
  const decay=Math.pow(0.5,dt);
  for(const i of front){ const a=act[i], nb=G.adj[i], hi=hop[i];
    if(a>0.12 && hi<MAXHOP && nb.length){ const cap=Math.min(nb.length,5), h=hi+1;
      for(let k=0;k<cap;k++){ const j=nb[k]; if(!fired[j]){ add.set(j,(add.get(j)||0)+a*4.0*dt/cap);
        if(!hm.has(j)||hm.get(j)>h) hm.set(j,h);
        // The wave carries its stream outward — a spreading thought stays the
        // color of whoever is thinking it.
        if(streamOf[i]!==255) sm.set(j,streamOf[i]); } } }
    if(G.A[i]===6 && a>0.35) tender[i]=1;    // where I was wrong stays tender
    act[i]*=decay;
    if(act[i]>0.03) nf.push(i); else { fired[i]=0; hop[i]=-1; streamOf[i]=255; }   // cooled → re-armable next wave
  }
  for(const [j,v] of add){ act[j]=Math.min(1,act[j]+v); if(hop[j]<0) hop[j]=hm.get(j);
    if(sm.has(j)) streamOf[j]=sm.get(j);
    if(act[j]>0.6) fired[j]=1; if(nf.indexOf(j)<0)nf.push(j); }
  front=nf.slice(0,2500); }

// ── named networks: a pinned recall/encode, and the hovered star ──
// The spotlight (pin or preview) names its memories — the picks, or what was
// written; a real recall often picks one, so what it activated and the
// strongest of what it returned fill in — draws the edges among the lit set
// and short threads from the named ones to their other neighbours. Nothing
// moves: the lit set shows where it lives, named at once, so going card to
// card never sends stars or names flying.
// _tierAt is the spotlight resolved to node indices (-1 = not lit; else an
// index into TIERS), which is what the render passes read; _spotDirty marks
// _highlightTier changed since it was resolved.
let _tierAt=null, _spotDirty=true, _net=[], _named=[];
const byStrength=(a,b)=>G.D[b]+G.H[b]-G.D[a]-G.H[a];
function syncSpotlight(){
  if(_spotDirty){ _spotDirty=false; _net=[]; _named=[]; _tierAt.fill(-1);
    const lit=[]; for(const [id,t] of _highlightTier){ const i=G.idIndex.get(id); if(i==null) continue; const k=TIER_IX[t]; lit.push(i); _tierAt[i]=k; }
    const set=new Set(lit);
    _named=lit.filter(i=>TIERS[_tierAt[i]].named);
    for(const k of [TIER_IX.activation,TIER_IX.returned]){ if(_named.length>=7) break;
      _named=_named.concat(lit.filter(i=>_tierAt[i]===k).sort(byStrength).slice(0,7-_named.length)); }
    _named=_named.slice(0,9);
    const seen=new Set();
    for(const i of lit) for(const j of G.adj[i]) if(set.has(j)){ const k=i<j?i+','+j:j+','+i; if(!seen.has(k)){ seen.add(k); _net.push([i,j,1]); } }
    for(const i of _named){ let n=0; for(const j of G.adj[i]){ if(set.has(j)) continue; _net.push([i,j,0]); if(++n>=10) break; } } } }
// Hover: the star stays put (so it never slides from under the cursor); its
// connections and strongest neighbours come forward and the rest steps back.
let _hov=null, _hovIdx=-1, _hovShow=-1, _hovT=0, _hovNamed=[];
// Titles wait for the pointer to settle: sweeping the sky shows the ring and
// the connections at once, but names — the tooltip, the star's and its
// neighbours' — only after HOVER_TITLE_MS of holding still. Jitter under
// STILL_PX doesn't count as moving.
const HOVER_TITLE_MS=1500, STILL_PX=4;
let _stillAt=0, _stillX=0, _stillY=0, _titleT=0, _tipFor=-1;
function syncTitles(dt,now){ const still=hoverIdx>=0 && now-_stillAt>=HOVER_TITLE_MS;
  _titleT=still?Math.min(1,_titleT+dt*4):0;
  if(still && _tipFor!==hoverIdx){ _tipFor=hoverIdx; showTip(hoverIdx,_stillX,_stillY); } }
function _hideTitles(){ _titleT=0; _tipFor=-1; if(tip)tip.style.opacity='0'; }
function syncHover(dt){ const want=(!_highlightTier.size && hoverIdx>=0 && hoverIdx<G.N)?hoverIdx:-1;
  if(want!==_hovIdx){ _hovIdx=want; if(want>=0){ _hovShow=want; _hov=new Uint8Array(G.N); _hov[want]=1; for(const j of G.adj[want]) _hov[j]=1;
    _hovNamed=[...new Set(G.adj[want])].sort(byStrength).slice(0,7); } }
  _hovT=_hovIdx>=0?Math.min(1,_hovT+dt*6):Math.max(0,_hovT-dt*4); if(_hovT<=0){ _hov=null; _hovShow=-1; } }
// One label: kept on-canvas, nudged down past labels already placed, on a dark
// plate when it has to read over a busy field.
function label(i,text,font,color,alpha,boxes,plate){ ctx.font=font; const w=ctx.measureText(text).width, pad=coreR(i)*0.6*sp[i]*_boost+6;
  let x=(sx[i]+pad+w>W-6)?sx[i]-pad-w:sx[i]+pad, y=sy[i];
  for(let q=0;q<4&&boxes.some(b=>x<b[2]&&x+w>b[0]&&Math.abs(y-b[1])<14);q++) y+=14;
  if(!plate&&boxes.some(b=>x<b[2]&&x+w>b[0]&&Math.abs(y-b[1])<12)) return; boxes.push([x,y,x+w]);
  ctx.globalAlpha=alpha; if(plate){ ctx.fillStyle='rgba(6,8,16,.74)'; ctx.fillRect(x-4,y-8,w+8,16); }
  ctx.fillStyle=color; ctx.fillText(text,x,y); }
const cut=(t,n)=>(t||'').length>n?t.slice(0,n-1)+'…':(t||'');
const lighten=(h,d)=>`rgb(${Math.min(255,h[0]+d)},${Math.min(255,h[1]+d)},${Math.min(255,h[2]+d)})`;

function frame(now){ if(!G||!ctx){raf=0;return;}
  const dt=Math.min((now-t0)/1000,.05); t0=now;
  breath+=dt; const br=0.9+0.1*Math.sin(breath*0.9);   // a slow living breath — always present
  if(!reduce){ const tc=Math.pow(0.9993,dt*60); for(let i=0;i<G.N;i++) if(tender[i]>0.004) tender[i]*=tc; }
  if(spin&&!reduce){ roll+=dt*0.075; wob+=dt; needSort=true; }
  stepActivation(dt);
  // NOTHING seeds activation here. A node lights up when — and only when —
  // the brain actually touched it: a recall recognized it, or an encode wrote
  // it, so "something just happened" is never decorative. An idle brain
  // should LOOK idle: the breath below is the resting state, and it is honest.
  syncSpotlight(); syncHover(dt); syncTitles(dt,now);
  project();
  ctx.clearRect(0,0,W,H2);
  ctx.globalCompositeOperation='lighter';
  const spot=_highlightTier.size>0, hovering=_hov&&_hovT>0.1, naming=spot&&_named.length>0;
  const g=G, young=1-_grow;
  // community clouds, under everything
  if(_haloGain>0){ const sc=cam.sc, thin=Math.min(1,3/Math.sqrt(Math.max(1,_halos.length)));
    for(const [x,y,z,rad,ka,m] of _halos){ const [px,py,pe]=projPt(x,y,z), s=Math.min(MIN*0.32,Math.max(18,rad*sc*pe*2.1));
      ctx.globalAlpha=Math.min(0.22,0.03+0.018*Math.log1p(m))*_haloGain*thin*br*(spot?0.3:1); ctx.drawImage(SPR[ka],px-s,py-s,s*2,s*2); } }
  // lines: 'auto' draws a young brain's threads; 'on' draws them at any size; 'off' never
  if(linesMode==='on' || (linesMode==='auto' && _lineGain>0)){ const paths=[],bp=[], bucket=_lineGain>0;
    for(let e=0;e<g.edges.length;e++){ const a=g.edges[e][0],b=g.edges[e][1],r=g.edges[e][2];
      const ax=sx[a],ay=sy[a],bx=sx[b],by=sy[b];
      // short threads carry a young brain's structure; long bridges fade
      const lk=bucket?Math.min(2,Math.floor(Math.hypot(bx-ax,by-ay)/(MIN*0.14))):0;
      const t=g.edges[e][3]?bp:paths, key=r*3+lk; bow(t[key]||(t[key]=new Path2D()),ax,ay,bx,by,0.06); }
    const LK=[1,0.4,0.12], sd=spot?0.18:1, lw=0.6+0.6*_lineGain;
    ctx.globalAlpha=1;
    for(const [set,a,w] of [[paths,(0.05+0.20*_lineGain)*sd,lw],[bp,(0.13+0.22*_lineGain)*sd,lw+0.2]])
      set.forEach((p,key)=>{ const h=HUE[key/3|0]; ctx.strokeStyle=`rgba(${h[0]},${h[1]},${h[2]},${a*LK[key%3]})`; ctx.lineWidth=w; ctx.stroke(p); }); }

  if(needSort){ order.sort((i,j)=>sz[j]-sz[i]); needSort=false; }   // re-sort only when the view rotated
  // PASS 1 — the present body: every memory glowing by what it IS (kind+reach+warmth),
  // breathing. not heat-gated — nothing goes dark. a thought adds warmth on top.
  for(let k=0;k<g.N;k++){ const i=order[k],ac=act[i];
    const dim = (_searchQuery&&!matches(i)) || (spot&&_tierAt[i]<0);
    const hd=(_hov&&!_hov[i])?1-0.7*_hovT:1;
    const pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7));
    const s=(1.5+2.5*young+g.D[i]*10+ac*ac*24)*pe*1.7*_boost;
    ctx.globalAlpha=Math.min(0.4,(0.03+0.13*g.L[i])*br*(0.45+0.55*depth)+ac*0.34)*(dim?0.16:1)*hd;
    ctx.drawImage(ac>0.2?WHITE:SPR[g.A[i]],sx[i]-s,sy[i]-s,s*2,s*2); }
  // firing bloom — the crest of a thought, in the color of the stream thinking it
  for(let x=0;x<front.length;x++){const i=front[x],ac=act[i];if(ac<0.12)continue;const pe=sp[i],s=(3+ac*15)*pe;
    ctx.globalAlpha=Math.min(0.55,ac*0.65);
    ctx.drawImage(g.A[i]===6?RED:streamSprite(streamOf[i]),sx[i]-s,sy[i]-s,s*2,s*2);}
  // tenderness — where I was wrong stays a little warm, always
  for(let i=0;i<g.N;i++){ const td=tender[i]; if(td<0.02)continue; const pe=sp[i],s=(2+g.D[i]*3)*pe;
    ctx.globalAlpha=Math.min(0.2,td*0.14); ctx.drawImage(RED,sx[i]-s,sy[i]-s,s*2,s*2); }
  // PASS 2 — crisp cores: the structure, always here, source-over so it never blows out
  ctx.globalCompositeOperation='source-over';
  for(let k=0;k<g.N;k++){ const i=order[k],pe=sp[i],depth=Math.max(0,Math.min(1,(pe-0.5)/0.7)),heat=g.H[i],hh=heat*heat,ac=act[i];
    const dim=(_searchQuery&&!matches(i))||(spot&&_tierAt[i]<0);
    const tk=spot?_tierAt[i]:-1;
    const hb=(_hov&&_hov[i])?1+0.5*_hovT:1;
    const s=(coreR(i)+ac*4)*0.6*pe*(tk>=0?1.5:1)*_boost*hb;
    let spr,al;
    if(ac>0.15){spr=g.A[i]===6?RED:streamSprite(streamOf[i]);al=(0.4+0.55*ac)*(0.45+0.55*depth);}
    else if(tk>=0){ spr=tierSprite(TIERS[tk].rgb); al=(0.55+0.4*depth);}
    else{spr=colorMode===0?SPR[g.A[i]]:(heat>0.62?WHITE:SPR[3]);
      al=(colorMode===0?(0.14+0.22*young+0.6*g.L[i])*br:(0.12+0.8*hh))*(0.45+0.55*depth);}
    if(dim && tk<0 && ac<0.15){ al*=0.16; }
    if(_hov) al*=_hov[i]?1+0.3*_hovT:1-0.7*_hovT;
    ctx.globalAlpha=Math.min(0.96,al); ctx.drawImage(spr,sx[i]-s,sy[i]-s,s*2,s*2); }
  ctx.textBaseline='middle';
  // a young brain names its strongest memories — until something else is being named
  if(_labels.length && !naming && !hovering){ const boxes=[];
    for(const i of _labels){ const depth=clamp01((sp[i]-0.5)/0.7);
      label(i,cut(g.TITLE[i],38),'10.5px ui-monospace,Menlo,monospace',lighten(HUE[g.A[i]],40),(0.35+0.5*depth)*(spot?0.3:1),boxes,false); } }
  // what each end of depth means — the brain's own topics at the poles. Only
  // while the operator turns the galaxy (and a moment after): that's when depth
  // shows as parallax; at rest the captions read as stray titles.
  _axisT=now-_turnedAt<AXIS_LINGER_MS?Math.min(1,_axisT+dt*4):Math.max(0,_axisT-dt*1.5);
  if(_axisT>0 && _zEnds && raw && raw.depth_axis && !spot && !hovering){ const ax=raw.depth_axis, short=a=>(a||[]).slice(0,2).map(t=>cut(String(t).split(/[:—(;,]/)[0].trim(),32)).join(' · ');
    const up=short(ax.up), dn=short(ax.down);
    if(up && up!==dn){ const [ux,uy]=projPt(0,0,_zEnds[0]), [dx,dy]=projPt(0,0,_zEnds[1]);
      ctx.globalAlpha=0.28*_axisT; ctx.strokeStyle='rgb(150,165,200)'; ctx.lineWidth=1; ctx.setLineDash([2,5]); ctx.beginPath(); ctx.moveTo(ux,uy); ctx.lineTo(dx,dy); ctx.stroke(); ctx.setLineDash([]);
      ctx.font='10px ui-monospace,Menlo,monospace'; ctx.globalAlpha=0.62*_axisT; ctx.fillStyle='rgb(200,210,235)';
      for(const [t,x,y] of [['▲ '+up,ux,uy],['▼ '+dn,dx,dy]]) ctx.fillText(t,Math.min(W-ctx.measureText(t).width-8,x+8),y); } }
  // the pinned recall's named network
  if(naming){ const boxes=[]; ctx.globalCompositeOperation='lighter';
    for(const [i,j,strong] of _net){ const c=TIERS[_tierAt[i]].rgb;
      // threads to memories outside the recall reach out a little way, not across the sky
      const f=strong?1:Math.min(1,MIN*0.07/(Math.hypot(sx[j]-sx[i],sy[j]-sy[i])+1e-6)), ex=sx[i]+(sx[j]-sx[i])*f, ey=sy[i]+(sy[j]-sy[i])*f;
      ctx.globalAlpha=strong?0.62:0.3; ctx.strokeStyle=`rgb(${c[0]},${c[1]},${c[2]})`; ctx.lineWidth=strong?1.3:0.7;
      ctx.beginPath(); bow(ctx,sx[i],sy[i],ex,ey,0.08); ctx.stroke(); }
    ctx.globalCompositeOperation='source-over';
    for(const i of _named){ const c=TIERS[_tierAt[i]].rgb;
      label(i,cut(g.TITLE[i],54),'600 11.5px ui-monospace,Menlo,monospace',`rgb(${c[0]},${c[1]},${c[2]})`,0.92,boxes,true); } }
  // the hovered star: its name, its connections coloured by relation, its strongest neighbours named
  if(_hovShow>=0){ const i=_hovShow, nb=g.adj[i], ra=g.adjA[i], boxes=[]; ctx.globalCompositeOperation='lighter';
    for(let k=0;k<nb.length&&k<60;k++){ const j=nb[k], h=HUE[ra[k]]||HUE[DEF_REL_ASP]; ctx.globalAlpha=0.55*_hovT; ctx.strokeStyle=`rgb(${h[0]},${h[1]},${h[2]})`; ctx.lineWidth=1.2;
      ctx.beginPath(); bow(ctx,sx[i],sy[i],sx[j],sy[j],0.1); ctx.stroke(); }
    ctx.globalCompositeOperation='source-over';
    if(_titleT>0){ const a=_hovT*_titleT;
      label(i,cut(g.TITLE[i],70),'600 12.5px ui-monospace,Menlo,monospace',lighten(HUE[g.A[i]],50),a,boxes,true);
      for(const j of _hovNamed) label(j,cut(g.TITLE[j],44),'10.5px ui-monospace,Menlo,monospace',lighten(HUE[g.A[j]],30),0.85*a,boxes,true); } }
  // hover ring
  if(hoverIdx>=0 && hoverIdx<g.N){ const i=hoverIdx;
    ctx.globalAlpha=0.9;ctx.strokeStyle='rgba(255,255,255,.9)';ctx.lineWidth=1.4;
    ctx.beginPath();ctx.arc(sx[i],sy[i],7+g.D[i]*10,0,6.2832);ctx.stroke(); }
  ctx.globalAlpha=1;
  raf=requestAnimationFrame(frame); }

const _tierSprites={};
function tierSprite(c){ const k=c.join(','); if(!_tierSprites[k])_tierSprites[k]=sprite(c[0],c[1],c[2]); return _tierSprites[k]; }
// Firing color per stream. 255 (untagged) falls back to the neutral white
// flash, so a recall with no session — an internal one — still reads as fired.
const _streamSprites={};
function streamSprite(h){ if(h===255||h>=STREAM_RGB.length) return WHITE;
  if(!_streamSprites[h]){ const c=STREAM_RGB[h]; _streamSprites[h]=sprite(c[0],c[1],c[2]); }
  return _streamSprites[h]; }

// ── picking + tooltip ──
function nearest(mx,my){ if(!G)return-1; let best=-1,bd=18*18;
  for(let i=0;i<G.N;i++){ if(sp[i]<0.4)continue; const dx=sx[i]-mx,dy=sy[i]-my,d=dx*dx+dy*dy;
    if(d<bd){bd=d;best=i;} } return best; }
function ago(days){ if(days>=9999)return'unknown'; if(days<1)return'today'; if(days<30)return Math.round(days)+'d ago'; return Math.round(days/30)+'mo ago'; }
function recallWord(h){ return h>0.6?'recalled recently':h>0.3?'recalled this month':'resting'; }
function showTip(i,mx,my){ if(!tip)return; const h=HUE[G.A[i]],name=KIND_NAME[G.A[i]]||G.TYPE[i]||'memory';
  tip.querySelector('.gt-title').textContent=G.TITLE[i];
  tip.querySelector('.gt-meta').innerHTML=`<span class="gt-k" style="color:rgb(${h[0]},${h[1]},${h[2]});border-color:rgb(${h[0]},${h[1]},${h[2]})">${escapeHtml(name)}</span>`+
    `<span>${G.DEG[i]} connection${G.DEG[i]===1?'':'s'}</span><span>·</span><span>created ${ago(G.AGE[i])}</span><span>·</span><span>${recallWord(G.H[i])}</span>`;
  let x=mx+16,y=my+16; if(x>W-300)x=mx-290; if(y>H2-80)y=my-70;
  tip.style.left=x+'px'; tip.style.top=y+'px'; tip.style.opacity='1'; }

// ── DOM scaffold inside #graph-3d ──
const LINES_ORDER=['auto','on','off'];
function buildScaffold(){
  host=document.getElementById('graph-3d'); if(!host) return false;
  host.innerHTML='';
  host.style.position='relative';
  cv=document.createElement('canvas'); cv.style.cssText='position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab;background:radial-gradient(140% 110% at 52% 44%,#0a0f24 0%,#06080f 44%,#04050c 100%)';
  host.appendChild(cv); ctx=cv.getContext('2d');
  tip=document.createElement('div'); tip.className='gt';
  tip.style.cssText='position:absolute;z-index:5;pointer-events:none;opacity:0;transition:opacity .12s;max-width:290px;padding:9px 11px;border-radius:9px;background:rgba(6,8,16,.94);border:1px solid #1a2138;box-shadow:0 10px 30px -12px #000';
  tip.innerHTML='<div class="gt-title" style="font:600 12.5px system-ui;color:#eef2fb;line-height:1.35;margin-bottom:6px"></div><div class="gt-meta" style="font:10.5px ui-monospace,Menlo,monospace;color:#67718e;display:flex;gap:6px;flex-wrap:wrap;align-items:center"></div>';
  host.appendChild(tip);
  legend=document.createElement('div');
  legend.style.cssText='position:absolute;left:10px;bottom:8px;z-index:4;display:flex;flex-wrap:wrap;gap:4px 8px;max-width:72%;pointer-events:none;font:10px ui-monospace,Menlo,monospace;color:#8791ab';
  legend.innerHTML=NODE_FAMS.map(a=>{const h=HUE[a];return `<span style="display:flex;align-items:center;gap:4px"><span style="width:7px;height:7px;border-radius:50%;background:rgb(${h[0]},${h[1]},${h[2]});box-shadow:0 0 5px rgb(${h[0]},${h[1]},${h[2]})"></span>${NF_LABEL[a]}</span>`;}).join('');
  host.appendChild(legend);
  // toggles overlay the canvas, banded ABOVE the legend so they never collide
  const ctl=document.createElement('div');
  ctl.style.cssText='position:absolute;right:10px;bottom:34px;z-index:4;display:flex;gap:6px';
  ctl.innerHTML=`<button data-g="shape" class="gbtn" title="How the graph is laid out — the same memories, arranged by a different fundamental">shape: ${SHAPE_LABEL[_shape]}</button><button data-g="color" class="gbtn">color: kind</button><button data-g="lines" class="gbtn" title="auto: a young brain's threads, fading as it grows · on: always · off: never">lines: ${linesMode}</button><button data-g="spin" class="gbtn">spin: on</button>`;
  host.appendChild(ctl);
  ctl.querySelectorAll('.gbtn').forEach(b=>{ b.style.cssText='border:1px solid #1a2138;background:rgba(8,10,20,.72);color:#c9d3e6;font:10px ui-monospace,Menlo,monospace;letter-spacing:.06em;text-transform:uppercase;border-radius:7px;padding:6px 9px;cursor:pointer;white-space:nowrap;backdrop-filter:blur(6px)';
    b.onclick=()=>{ const k=b.dataset.g;
      if(k==='shape'){ setShape(SHAPE_ORDER[(SHAPE_ORDER.indexOf(_shape)+1)%SHAPE_ORDER.length]); }
      if(k==='color'){colorMode^=1;b.textContent='color: '+(colorMode?'heat':'kind');}
      if(k==='lines'){linesMode=LINES_ORDER[(LINES_ORDER.indexOf(linesMode)+1)%LINES_ORDER.length];b.textContent='lines: '+linesMode;}
      if(k==='spin'){spin=!spin;b.textContent='spin: '+(spin?'on':'off');} }; });
  wireCanvas();
  return true;
}
let drag=false,lx=0,ly=0,sh=false,moved=false;
const AXIS_LINGER_MS=1200;
let _turnedAt=-Infinity, _axisT=0;   // last drag-rotate, and the depth captions' fade
// window-level pointer handlers: bound ONCE (in init) — they reference the
// current module cv/G/tip, which are reassigned per mount. Binding per mount
// would add a pair on every Refresh.
function _onWinMove(e){ if(!G)return;
  if(drag){const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;
    if(moved){ hoverIdx=-1; _hideTitles(); }   // the sky turned under the pointer; the next move re-picks
    if(sh){panx+=dx;pany+=dy;}else{yaw+=dx*0.005;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.005));needSort=true;_turnedAt=performance.now();}return;}
  if(!cv)return; const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  if(mx<0||my<0||mx>r.width||my>r.height){hoverIdx=-1;_hideTitles();return;}
  const i=nearest(mx,my);
  if(i!==hoverIdx||Math.hypot(mx-_stillX,my-_stillY)>STILL_PX){ _stillAt=performance.now(); _stillX=mx; _stillY=my; _hideTitles(); }
  hoverIdx=i; cv.style.cursor=i>=0?'pointer':'grab'; }
function _onWinUp(e){ if(!drag)return; drag=false; if(cv)cv.style.cursor='grab';
  if(!moved&&G){const r=cv.getBoundingClientRect(),i=nearest(e.clientX-r.left,e.clientY-r.top);
    if(i>=0){ seed(i); loadNodeDetail(G.IDS[i]); }} }
function wireWindowOnce(){ if(_winWired)return; _winWired=true;
  window.addEventListener('mousemove',_onWinMove); window.addEventListener('mouseup',_onWinUp);
  // Leaving the window straight from the canvas sends no outside move; without this the
  // still clock keeps counting and names appear after the pointer is gone.
  window.addEventListener('mouseout',e=>{ if(!e.relatedTarget){ hoverIdx=-1; _hideTitles(); } }); }
// canvas-scoped handlers re-bind per mount — the canvas is discarded on destroy,
// so these listeners are GC'd with it (no leak).
function wireCanvas(){
  cv.addEventListener('mousedown',e=>{drag=true;moved=false;lx=e.clientX;ly=e.clientY;sh=e.shiftKey;cv.style.cursor='grabbing';});
  cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.4,Math.min(6,zoom*(e.deltaY<0?1.12:.892)));},{passive:false});
}

function sizeCanvas(){ if(!cv||!host) return; DPR=Math.min(devicePixelRatio||1,2);
  W=host.clientWidth||800; H2=host.clientHeight||600; MIN=Math.min(W,H2);
  cv.width=W*DPR; cv.height=H2*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }

// ── load ──
export async function loadGraph3D(){
  const gen=_loadGen;
  if(!buildScaffold()) return;
  try {
    raw = await api.graph3d();
    if(gen!==_loadGen) return;
    if(!raw.nodes || !raw.nodes.length){ _pendingPin=null; host.innerHTML='<div class="graph-error"><div class="graph-error-title">No graph data</div></div>'; return; }
    ensureSprites();
    G = buildGalaxy(raw);
    allocScreen(G.N); needSort=true;
    _applyShapeView();
    sizeCanvas();
    t0=performance.now();
    if(!raf) raf=requestAnimationFrame(frame);
    _updateMatchCount();
    if(_pendingPin){ const ev=_pendingPin; _pendingPin=null; pinRecallToGraph(ev); }
  } catch(e){ console.error('[graph] galaxy load failed:', e); _pendingPin=null;
    if(host) host.innerHTML='<div class="graph-error"><div class="graph-error-title">Graph unavailable</div><div class="graph-error-msg">'+escapeHtml(String(e&&e.message||e))+'</div></div>'; }
}

// ── shape ──
/** Switch layout. Relays out from the payload already in memory — no refetch,
 *  and the highlight/activation state survives, so a shape change is a change
 *  of viewpoint on the same moment rather than a reset. */
export function setShape(name){
  if(!SHAPES[name]||name===_shape) return;
  _shape=name;
  try { localStorage.setItem(SHAPE_KEY,_shape); } catch(e){}
  const btn=host&&host.querySelector('.gbtn[data-g="shape"]');
  if(btn) btn.textContent='shape: '+SHAPE_LABEL[_shape];
  _applyShapeView();
  if(raw) _rebuild();
}

/** Re-lay G out from `raw`, keeping what is lit: rebuilding reallocates the
 *  activation arrays, so carry the live values — and the stream each wave
 *  belongs to — across by node id. */
function _rebuild(){
  const carried=[];
  if(G){ for(let i=0;i<G.N;i++) if(act[i]>0.02||tender[i]>0.02) carried.push([G.IDS[i],act[i],tender[i],streamOf[i]]); }
  G=buildGalaxy(raw);
  allocScreen(G.N); needSort=true;
  for(const [id,a,td,st] of carried){ const i=G.idIndex.get(id); if(i!=null){ act[i]=a; tender[i]=td; streamOf[i]=st; if(a>0.12) front.push(i); } }
  sizeCanvas();
}

/** Fetch a fresh payload and swap it in under the current view: the old graph
 *  keeps drawing while it loads, and the camera, the lit waves and the
 *  spotlight carry across — no blank canvas, no reset angle. Only the newest
 *  request swaps; a teardown in the meantime drops it. Any pending pin lands
 *  after, on the fresh graph or (if the fetch failed) the one still shown. */
let _swapGen=0;
async function _reloadInPlace(){
  const gen=_loadGen, mine=++_swapGen;
  let fresh=null;
  try { fresh=await api.graph3d(); } catch(e){ console.error('[graph] in-place reload failed:', e); }
  if(gen!==_loadGen || mine!==_swapGen || !G) return;
  if(fresh && fresh.nodes && fresh.nodes.length){ raw=fresh; _rebuild(); _updateMatchCount(); }
  if(_pendingPin){ const ev=_pendingPin; _pendingPin=null; pinRecallToGraph(ev); }
}

/** Point the camera the way the current shape wants to be seen. Called on
 *  every load and every shape switch — a persisted `strata` must come back
 *  flat on reload, not only when you switch into it. */
function _applyShapeView(){
  const view=SHAPE_VIEW[_shape];
  if(!view) return;
  pitch=view.pitch; roll=view.roll; yaw=0; wob=0; spin=view.spin; needSort=true;
  const sb=host&&host.querySelector('.gbtn[data-g="spin"]');
  if(sb) sb.textContent='spin: '+(spin?'on':'off');
}

// ── search ──
function _updateMatchCount(){ const el=document.getElementById('graph-search-count'); if(!el)return;
  if(!_searchQuery){el.textContent='';return;}
  let n=0; if(G){ for(let i=0;i<G.N;i++) if(matches(i)) n++; }
  else if(raw?.nodes){ const q=_searchQuery; n=raw.nodes.filter(x=>(x.name||'').toLowerCase().includes(q)||(x.type||'').toLowerCase().includes(q)).length; }
  el.textContent=n+' match'+(n===1?'':'es'); }
export function setSearchQuery(q){ _searchQuery=(q||'').toLowerCase().trim(); _updateMatchCount(); }
export function onGraphSearch(){ const i=document.getElementById('graph-search'); setSearchQuery(i?i.value:''); }
export function onGraphSearchKey(event){ if(event&&event.key==='Enter') onGraphSearch(); }

// ── recall highlight (persistent spotlight for pin/preview) + live bloom ──
// A graph event is a recall (returned / activation / used ids) or an encode
// run (created / revised ids, plus the linked ids it wove them into).
function _applyEvent(event){ if(!event)return; _spotDirty=true;
  for(const t of TIERS) for(const id of event[t.name+'_ids']||[]) _highlightTier.set(id,t.name); }
// Seed ONLY the judge's picks — or what an encode wrote, or a few activation
// seeds — and let stepActivation carry the wave outward. Seeding the whole
// recall set at once made it all flash together instead of spreading. A live
// run's new memories aren't in the loaded graph yet, so while any is missing
// the memories it linked them to stand in for it.
function _bloomEvent(event){ if(!event||!G)return;
  const hue=sessionHueIndex(event.session_id||'');
  const wrote=[...(event.created_ids||[]),...(event.revised_ids||[])];
  const touched=wrote.every(id=>G.idIndex.has(id)) ? wrote : [...wrote,...(event.linked_ids||[])];
  const src=(event.used_ids&&event.used_ids.length) ? event.used_ids
          : touched.length ? touched
          : (event.activation_ids||[]).slice(0,3);
  for(const id of src){ const i=G.idIndex.get(id); if(i!=null) seed(i,hue); } }
// Live "watch it think": live.js publishes each recall or encode once, as it
// lands. It only blooms — a pinned spotlight stays put.
function _onActivity({event}){ _bloomEvent(event); }
export function previewRecallOnGraph(event){ if(!G||!event)return; if(!_previewSnapshot) _saveSnapshot();
  _highlightTier.clear(); _applyEvent(event); }
export function clearRecallPreview(){ if(_previewSnapshot) _restoreSnapshot(); }
function _saveSnapshot(){ _previewSnapshot={tier:new Map(_highlightTier),pinned:_pinnedEventId}; }
function _restoreSnapshot(){ if(!_previewSnapshot)return; _highlightTier.clear(); _spotDirty=true;
  for(const [k,v] of _previewSnapshot.tier)_highlightTier.set(k,v); _pinnedEventId=_previewSnapshot.pinned; _previewSnapshot=null; }
// Events already reloaded for. A card can list memories the graph will never
// hold (archived since the run), so each event gets one reload, ever.
const _reloadedFor=new Set();
export function pinRecallToGraph(event){ if(!event)return;
  _pendingPin=null;
  // A memory written moments ago isn't in the graph loaded before it: fetch
  // once in place, then pin.
  const written=[...(event.created_ids||[]),...(event.revised_ids||[])];
  if(G && !_reloadedFor.has(event.id) && written.some(id=>!G.idIndex.has(id))){
    _reloadedFor.add(event.id); _pendingPin=event; _reloadInPlace(); return; }
  _previewSnapshot=null;
  _pinnedEventId=event.id||null; _highlightTier.clear(); _applyEvent(event); _bloomEvent(event);
  bus.publish('graph:pinned',{eventId:_pinnedEventId}); }
/** Drop the spotlight without touching the layout or refetching. Split out of
 *  onGraphRefresh so a caller that only wants to stop highlighting (closing the
 *  recall panel) doesn't pay a full destroy + reload. With an `eventId`, only
 *  when that event is what holds the graph (pinned, or pending its reload) —
 *  collapsing a card releases the graph only if that card pinned it. */
export function clearHighlight(eventId){
  if(eventId!==undefined && eventId!==_pinnedEventId && eventId!==(_pendingPin&&_pendingPin.id)) return;
  _highlightTier.clear(); _spotDirty=true; _pinnedEventId=null; _pendingPin=null;
  _previewSnapshot=null; bus.publish('graph:pinned',{eventId:null}); }
export function onGraphRefresh(){ clearHighlight(); destroy(); loadGraph3D(); }

// ── lifecycle ──
export function init(){
  bus.subscribe('activity:new', _onActivity);
  wireWindowOnce();
  const h=document.getElementById('graph-3d');
  if(h && 'ResizeObserver' in window){ _ro=new ResizeObserver(()=>requestAnimationFrame(resize)); _ro.observe(h); }
}
export function resize(){ if(!cv||!host) return; sizeCanvas(); needSort=true; }
export function activate(){
  if(_activateTimer) clearTimeout(_activateTimer);
  _activateTimer=setTimeout(()=>{ _activateTimer=null;
    if(!G) loadGraph3D(); else { sizeCanvas(); if(!raf){t0=performance.now();raf=requestAnimationFrame(frame);} }
    // No opening bloom: replaying the last recall as a live firing would
    // make every tab-switch look like the brain had just thought something.
  }, 300);
}
export function deactivate(){ /* canvas keeps animating; cheap */ }
export function destroy(){
  if(_activateTimer){clearTimeout(_activateTimer);_activateTimer=null;}
  _loadGen++;
  if(raf){cancelAnimationFrame(raf);raf=0;}
  G=null; raw=null;
  const c=document.getElementById('graph-3d'); if(c){c.innerHTML='';} cv=null; ctx=null; host=null; tip=null;
}
