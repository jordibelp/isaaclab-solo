// Editable shot phases. Source defaults stay immutable; all distances are metres.
function createShotCurriculum(){
  const initial=()=>D.shot_half_extents_yz.map((extent,i)=>({extent:[...extent],flight:[...D.shot_flight_time_ranges_s[i]]}));
  let phases=initial();
  const overlays=new THREE.Group();scene.add(overlays);
  const fields=['spread-y','spread-z','flight-min','flight-max'];
  const selected=()=>phases[+$('level').value];
  const fmt=n=>Number.isInteger(n)?n.toFixed(1):String(n);

  function updateConfig(){
    const tuple=(key)=>'(\n'+phases.map(p=>'    ('+p[key].map(fmt).join(', ')+'),').join('\n')+'\n)';
    $('curriculum-config').value=`shot_half_extents_yz = ${tuple('extent')}\nshot_flight_time_ranges_s = ${tuple('flight')}\nshot_level = ${$('level').value}`;
    $('copy-status').textContent='';
  }
  function menu(index){
    $('level').replaceChildren(...phases.map((p,i)=>new Option(`${i} · ±${p.extent.map(fmt).join(' / ')} m`,i)));
    $('level').value=index;
    $('phase-remove').disabled=phases.length===1;
  }
  function loadFields(){
    const p=selected();
    fields.forEach((id,i)=>$(id).value=[...p.extent,...p.flight][i]);
    $('phase-error').textContent='';
    for(const id of ['sample','phase-add','phases-copy'])$(id).disabled=false;
    updateConfig();
  }
  function rectangle(extent,x,color,filled,phase,plane){
    const [y,z]=D.target_center_yz,[hy,hz]=extent;
    const points=[[x,y-hy,z-hz],[x,y+hy,z-hz],[x,y+hy,z+hz],[x,y-hy,z+hz]].map(V);
    const geometry=new THREE.BufferGeometry().setFromPoints(points);
    const line=new THREE.LineLoop(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity:filled?1:.45}));
    line.userData={phase,plane};overlays.add(line);
    if(filled){
      const fill=geometry.clone();fill.setIndex([0,1,2,0,2,3]);
      const mesh=new THREE.Mesh(fill,new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide,transparent:true,opacity:.18,depthWrite:false}));
      mesh.userData={phase,plane};overlays.add(mesh);
    }
  }
  function updateOverlay(){
    for(const child of [...overlays.children]){overlays.remove(child);child.geometry.dispose();child.material.dispose();}
    const index=+$('level').value,p=selected();
    // Offset from the solid goal's front face only to avoid depth fighting.
    const goalX=D.goal_center[0]+D.goal_size[0]/2+.001;
    if($('spread-goal').checked){
      if($('spread-all').checked)phases.forEach((phase,i)=>{if(i!==index)rectangle(phase.extent,goalX,0xb28cff,false,i,'goal');});
      rectangle(p.extent,goalX+.001,0xb28cff,true,index,'goal');
    }
    if($('aim').checked)rectangle(p.extent,D.keeper_plane_x,0xff9960,false,index,'keeper');
    const outside=p.extent.some((v,i)=>Math.abs(D.target_center_yz[i]-D.goal_center[i+1])+v>D.goal_size[i+1]/2);
    $('spread-summary').textContent=`Phase ${index} of ${phases.length-1} · full area ${fmt(2*p.extent[0])} × ${fmt(2*p.extent[1])} m (Y × Z). Flight ${p.flight.map(fmt).join('–')} s to keeper plane. Centre Y/Z: ${D.target_center_yz.map(fmt).join(' / ')} m.${outside?' The projected area extends outside the goal.':''}`;
  }
  function edit(){
    const values=fields.map(id=>$(id).valueAsNumber);
    const valid=fields.every(id=>$(id).checkValidity())&&values.every(Number.isFinite)&&values[2]>0&&values[3]>=values[2];
    $('phase-error').textContent=valid?'':'Use non-negative spreads and 0 < flight min ≤ flight max. Last valid values remain active.';
    for(const id of ['sample','phase-add','phases-copy'])$(id).disabled=!valid;
    if(!valid)return;
    const index=+$('level').value;
    phases[index]={extent:values.slice(0,2),flight:values.slice(2)};
    menu(index);updateConfig();sample();
  }
  fields.forEach(id=>$(id).oninput=edit);
  $('level').onchange=()=>{loadFields();sample();};
  $('phase-add').onclick=()=>{const p=selected();phases.push({extent:[...p.extent],flight:[...p.flight]});menu(phases.length-1);loadFields();sample();};
  $('phase-remove').onclick=()=>{const index=+$('level').value;phases.splice(index,1);menu(Math.min(index,phases.length-1));loadFields();sample();};
  $('phases-reset').onclick=()=>{phases=initial();menu(D.shot_level);loadFields();sample();};
  for(const id of ['spread-goal','spread-all'])$(id).onchange=()=>{updateOverlay();render();};
  $('phases-copy').onclick=async()=>{
    try{await navigator.clipboard.writeText($('curriculum-config').value);$('copy-status').textContent='Copied.';}
    catch{const field=$('curriculum-config');field.focus();field.select();$('copy-status').textContent=document.execCommand('copy')?'Copied.':'Text selected. Press Ctrl+C to copy.';}
  };
  menu(D.shot_level);loadFields();
  return {selected,updateOverlay,snapshot:()=>({selected:+$('level').value,phases:structuredClone(phases),overlays:overlays.children.map(o=>({...o.userData,filled:o.isMesh===true,vertices:Array.from(o.geometry.attributes.position.array)}))})};
}
