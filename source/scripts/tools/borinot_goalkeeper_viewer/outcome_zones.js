// Ball-centre outcome geometry. These guides do not simulate contact or score saves.
function createOutcomeZones(){
  const half=floorSide/2, escape=D.outcome_rules.escape_bounds_m;
  const height=Math.max(escape.z+.5,D.goal_center[2]+D.goal_size[2]/2+.5);
  const returnedX=D.keeper_plane_x+D.clear_distance_m;
  const goalX=D.goal_center[0]+D.goal_size[0]/2+D.ball_radius_m;
  const yMin=D.goal_center[1]-D.goal_size[1]/2-D.ball_radius_m;
  const yMax=D.goal_center[1]+D.goal_size[1]/2+D.ball_radius_m;
  const zMin=D.goal_center[2]-D.goal_size[2]/2-D.ball_radius_m;
  const zMax=D.goal_center[2]+D.goal_size[2]/2+D.ball_radius_m;
  const zones={};
  const fmt=n=>Number(n.toFixed(7)).toString();

  function zone(key,title,color,description,checked){
    const group=new THREE.Group();group.name=key;group.visible=checked;scene.add(group);zones[key]=group;
    const row=document.createElement('div');row.className='row';
    const input=document.createElement('input');input.type='checkbox';input.id='zone-'+key;input.checked=checked;
    const caption=document.createElement('label');caption.htmlFor=input.id;caption.textContent=title;
    caption.style.color='#'+color.toString(16).padStart(6,'0');row.append(input,caption);
    const hint=document.createElement('p');hint.className='hint';hint.textContent=description;
    $('outcome-controls').append(row,hint);
    input.onchange=()=>{group.visible=input.checked;render();};
    return group;
  }
  function line(group,name,points,color){
    const mesh=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(V)),
      new THREE.LineBasicMaterial({color,transparent:true,opacity:.95}));
    mesh.name=name;group.add(mesh);return mesh;
  }
  function region(group,name,low,high,color,opacity=.045){
    if(low.some((value,i)=>value>=high[i]))return;
    const size=low.map((value,i)=>high[i]-value),center=low.map((value,i)=>(value+high[i])/2);
    const mesh=box(size,center,color,opacity);mesh.name=name;group.add(mesh);
    const outline=edges(size,center,color);outline.material.opacity=.35;group.add(outline);
  }
  const teal=0x49e5c1,blue=0x72b5ff,yellow=0xf1d16a,pink=0xff80bc,red=0xff8577;
  const returned=zone('returned','1 · Returned toward the shooter',teal,
    `Floor line: x > ${fmt(returnedX)} m. Also needs ball velocity vx > ${fmt(D.clear_speed_m_s)} m/s. Teal floor shading points toward +X; this rule has no Y/Z limit.`,true);
  region(returned,'returned-floor-footprint',[returnedX,-half,.004],[half,half,.006],teal,.10);
  line(returned,'returned-floor-line',[[returnedX,-half,.012],[returnedX,half,.012]],teal);
  returned.add(new THREE.ArrowHelper(V([1,0,0]),V([returnedX+.06,0,.03]),.45,teal,.09,.06));
  returned.add(label(`Return x > ${returnedX.toFixed(3)} m`,[returnedX,.6,.17],1.65));

  const wide=zone('wide','2 · Safely wide behind the goal',blue,
    `x < ${fmt(goalX)} m AND (y < ${fmt(yMin)} or y > ${fmt(yMax)} or z < ${fmt(zMin)} or z > ${fmt(zMax)}) m. Blue volumes show the outside of the goal rectangle, behind its ball-centre crossing plane.`,true);
  region(wide,'wide-left',[-half,-half,0],[goalX,Math.min(yMin,half),height],blue);
  region(wide,'wide-right',[-half,Math.max(yMax,-half),0],[goalX,half,height],blue);
  region(wide,'wide-above',[-half,Math.max(yMin,-half),Math.max(zMax,0)],[goalX,Math.min(yMax,half),height],blue);
  region(wide,'wide-below',[-half,Math.max(yMin,-half),0],[goalX,Math.min(yMax,half),Math.min(zMin,height)],blue);
  line(wide,'wide-opening',[[goalX,yMin,Math.max(0,zMin)],[goalX,yMin,zMax],[goalX,yMax,zMax],[goalX,yMax,Math.max(0,zMin)]],blue);
  wide.add(label('Wide clearance · behind goal',[goalX,yMin-.45,1.4],1.9));

  const floor=zone('floor','3 · Floor landing in front of goal',yellow,
    `After a racket hit: floor contact with x > ${fmt(goalX)} m can also clear the ball. The floor penalty still applies. The fallback floor check uses ball-centre z ≤ ${fmt(D.ball_radius_m+D.outcome_rules.floor_slack_m)} m.`,false);
  region(floor,'floor-save-footprint',[goalX,-half,.007],[half,half,.009],yellow,.12);
  line(floor,'floor-save-boundary',[[goalX,-half,.015],[goalX,half,.015]],yellow);
  floor.add(label('Floor save · after a racket hit',[.9,-half+.65,.17],1.9));

  const miss=zone('miss','Miss / escape limits',pink,
    `Miss if NOT already cleared and (|x| > ${fmt(escape.x)} m OR |y| > ${fmt(escape.y)} m OR z > ${fmt(escape.z)} m). Pink shading is OUTSIDE the limits. There is no lower-Z escape boundary; the floor is separate.`,false);
  region(miss,'miss-negative-x',[-half,-half,0],[-escape.x,half,height],pink,.075);
  region(miss,'miss-positive-x',[escape.x,-half,0],[half,half,height],pink,.075);
  region(miss,'miss-negative-y',[-escape.x,-half,0],[escape.x,-escape.y,height],pink,.075);
  region(miss,'miss-positive-y',[-escape.x,escape.y,0],[escape.x,half,height],pink,.075);
  region(miss,'miss-above',[-escape.x,-escape.y,escape.z],[escape.x,escape.y,height],pink,.075);
  miss.add(label('Miss outside ±X / ±Y limits',[escape.x,0,.18],1.9));
  miss.add(label(`Miss above z = ${fmt(escape.z)} m`,[0,0,escape.z],1.9));

  const goalGate=zone('goal','Goal crossing / failure boundary',red,
    `A ball moving toward −X crosses x = ${fmt(goalX)} m inside y [${fmt(yMin)}, ${fmt(yMax)}] and z [${fmt(zMin)}, ${fmt(zMax)}] m: goal. Bounds include the ball radius. A real goal contact also counts.`,false);
  region(goalGate,'goal-crossing-plane',[goalX-.0005,yMin,zMin],[goalX+.0005,yMax,zMax],red,.13);
  goalGate.add(label(`Goal x = ${goalX.toFixed(3)} m`,[goalX,yMin+.55,Math.max(.3,zMin+.2)],1.6));

  $('outcome-note').textContent=`All clearance routes require a prior racket hit and no goal, arm/base collision, robot crash, tilt or area failure. Recovery required: ${D.require_recovery_for_save?'yes':'no'}. These are ball-centre guides, clipped to the displayed floor/height; they do not score the unblocked ball animation. The orange base-position box is a separate toggle under Scene guides. Other non-spatial checks: ${D.robot_tilt_threshold_degs}° tilt and ${D.shot_timeout_s} s shot deadline, plus physical contacts. No environment settings are changed.`;
  $('zones-fit').onclick=()=>{view('3d');target.set(0,0,height/2);distance=1.9*Math.max(floorSide,height);render();};
  return {snapshot:()=>({returned_x_m:returnedX,goal_front_x_m:goalX,goal_y_m:[yMin,yMax],goal_z_m:[zMin,zMax],
    escape_bounds_m:{...escape},floor_center_height_m:D.ball_radius_m+D.outcome_rules.floor_slack_m,
    display_bounds:{x:[-half,half],y:[-half,half],z:[0,height]},
    zones:Object.fromEntries(Object.entries(zones).map(([key,group])=>[key,{visible:group.visible,
      boxes:group.children.filter(child=>child.isMesh).map(mesh=>{const bounds=new THREE.Box3().setFromObject(mesh);return {name:mesh.name,min:bounds.min.toArray(),max:bounds.max.toArray()};}),
      lines:group.children.filter(child=>child.isLine&&child.name).map(mesh=>({name:mesh.name,vertices:Array.from(mesh.geometry.attributes.position.array)}))}]))})};
}
