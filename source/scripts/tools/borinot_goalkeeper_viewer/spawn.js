function createSpawnPreview(){
  const markers=new THREE.Group();scene.add(markers);
  let locations=[],active=false,lastLevel=null;
  const points=[];
  D.reset_grid_z_offsets_m.forEach((z,row)=>D.reset_grid_y_offsets_m.forEach((y,column)=>points.push({
    index:points.length,position:[D.robot_home[0],D.robot_home[1]+y,D.robot_home[2]+z],row,column
  })));
  function update(){
    const level=+$('level').value;
    if(level===lastLevel)return;
    lastLevel=level;active=D.reset_grid_enabled&&level>=D.reset_grid_start_level;
    locations=active?points:[{index:-1,position:[...D.robot_home]}];
    for(const child of [...markers.children]){markers.remove(child);child.geometry.dispose();child.material.dispose();}
    const zero=active?locations.findIndex(p=>p.position.every((v,i)=>Math.abs(v-D.robot_home[i])<1e-9)):0;
    $('spawn-location').replaceChildren(...locations.map((p,i)=>new Option(`${p.index<0?'Centre':p.index+' · '+(p.row+1)+'/'+(p.column+1)}`,i)));
    $('spawn-location').value=Math.max(0,zero);
    for(const point of locations){
      const [x,y,z]=point.position;
      const cross=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([
        V([x,y-.06,z]),V([x,y+.06,z]),V([x,y,z-.06]),V([x,y,z+.06])
      ]),new THREE.LineBasicMaterial({color:0x69dfcf,depthTest:false}));
      cross.userData={index:point.index,position:point.position};markers.add(cross);
    }
    markers.visible=$('spawn-markers').checked;
    $('spawn-summary').textContent=`Phase ${level}: ${active?'uniform random selection from '+points.length+' locations':'centre only'}. Grid starts at phase ${D.reset_grid_start_level}${D.reset_grid_enabled?'':'; grid disabled'}. Home: ${D.robot_home.join(', ')} m. Reset noise: XY ±${D.reset_xy_noise_m} m; Z ±${D.reset_z_noise_m} m.`;
  }
  function place(point,noise){
    restore();
    ['x','y','z'].forEach((key,i)=>state[key]=point.position[i]+(noise?(2*Math.random()-1)*(i===2?D.reset_z_noise_m:D.reset_xy_noise_m):0));
    sync();updatePose();
  }
  $('spawn-location').onchange=()=>place(locations[+$('spawn-location').value],false);
  $('spawn-sample').onclick=()=>{const i=Math.floor(Math.random()*locations.length);$('spawn-location').value=i;place(locations[i],true);};
  $('spawn-markers').onchange=()=>{markers.visible=$('spawn-markers').checked;render();};
  $('spawn-config').textContent=`reset_grid_enabled = ${D.reset_grid_enabled?'True':'False'}\nreset_grid_start_level = ${D.reset_grid_start_level}\nreset_grid_y_offsets_m = (${D.reset_grid_y_offsets_m.join(', ')},)\nreset_grid_z_offsets_m = (${D.reset_grid_z_offsets_m.join(', ')},)`;
  return {update,snapshot:()=>({active,locations:structuredClone(locations),visible:markers.visible,markers:markers.children.map(m=>({index:m.userData.index,vertices:Array.from(m.geometry.attributes.position.array)}))})};
}
