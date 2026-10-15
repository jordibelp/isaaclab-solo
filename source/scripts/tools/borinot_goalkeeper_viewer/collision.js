// Convex overlap preview. Cylinders are 64-sided; PhysX cooking/margins are not modeled.
function convexOverlap(a, b) {
  const support = d => {
    let p=a[0], q=b[0], hi=-Infinity, lo=Infinity;
    for(const v of a){const x=v.dot(d);if(x>hi){hi=x;p=v;}}
    for(const v of b){const x=v.dot(d);if(x<lo){lo=x;q=v;}}
    return p.clone().sub(q);
  };
  // Project the origin onto every face of the (at most tetrahedral) simplex.
  // Keeping the closest face avoids fragile line/triangle special cases.
  function closest(points) {
    let best=null;
    for(let mask=1;mask<(1<<points.length);mask++){
      const ids=points.filter((_,i)=>mask&(1<<i)), k=ids.length;
      if(k>4)continue;
      const edges=ids.slice(1).map(v=>v.clone().sub(ids[0]));
      const rows=edges.map(u=>[...edges.map(v=>u.dot(v)),-u.dot(ids[0])]);
      let singular=false;
      for(let c=0;c<k-1;c++){
        let pivot=c;for(let r=c+1;r<k-1;r++)if(Math.abs(rows[r][c])>Math.abs(rows[pivot][c]))pivot=r;
        [rows[c],rows[pivot]]=[rows[pivot],rows[c]];
        const scale=rows[c][c];if(Math.abs(scale)<1e-18){singular=true;break;}
        for(let j=c;j<k;j++)rows[c][j]/=scale;
        for(let r=0;r<k-1;r++)if(r!==c){const f=rows[r][c];for(let j=c;j<k;j++)rows[r][j]-=f*rows[c][j];}
      }
      if(singular)continue;
      const weights=rows.map(row=>row[k-1]);weights.unshift(1-weights.reduce((x,y)=>x+y,0));
      if(weights.some(w=>w < -1e-9))continue;
      const p=new THREE.Vector3();ids.forEach((v,i)=>p.addScaledVector(v,weights[i]));
      if(best===null||p.lengthSq()<best.p.lengthSq())best={p,points:ids};
    }
    return best;
  }
  const ba=new THREE.Box3().setFromPoints(a), bb=new THREE.Box3().setFromPoints(b);
  if(!ba.intersectsBox(bb))return false;
  let simplex=[support(new THREE.Vector3(1,0,0))], current=closest(simplex);
  for(let iteration=0;iteration<80;iteration++){
    if(current.p.lengthSq()<1e-16)return true;
    const direction=current.p.clone().negate(), next=support(direction);
    if(next.dot(direction)<-1e-12)return false;
    if(simplex.some(p=>p.distanceToSquared(next)<1e-22))return false;
    simplex=[...current.points,next];current=closest(simplex);
  }
  return null; // Indeterminate is displayed, never called collision-free.
}

function createCollisionInspector() {
  const shapes=[], material=new THREE.LineBasicMaterial({color:0x65ed9d,transparent:true,opacity:.8});
  for(const item of MODEL.collision_shapes){
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(item.vertices.flat(),3));
    geometry.setIndex(item.faces.flat());
    const mesh=new THREE.LineSegments(new THREE.EdgesGeometry(geometry,1),material.clone());mesh.matrix.fromArray(item.matrix);mesh.matrixAutoUpdate=false;
    bodies[item.body].add(mesh);shapes.push({...item,mesh});
  }
  for(const [name,part] of [['paddle_head',paddleHead],['paddle_handle',paddleHandle]]){
    const positions=part.geometry.attributes.position,vertices=new Map();
    for(let i=0;i<positions.count;i++){const v=new THREE.Vector3().fromBufferAttribute(positions,i).toArray();vertices.set(v.join(','),v);}
    const mesh=new THREE.LineSegments(new THREE.EdgesGeometry(part.geometry),material.clone());
    part.add(mesh);shapes.push({name,body:'paddle',mesh,vertices:[...vertices.values()]});
  }
  const adjacent=new Set(MODEL.joints.map(j=>[j.parent,j.child].sort().join('|')));
  adjacent.add(['flying_arm_2__link2','paddle'].sort().join('|'));
  let result=[];
  function update(){
    scene.updateMatrixWorld(true);
    for(const s of shapes){s.mesh.visible=$('colliders').checked;s.mesh.material.color.setHex(0x65ed9d);
      s.world=s.vertices.map(v=>new THREE.Vector3(...v).applyMatrix4(s.mesh.matrixWorld));}
    result=[];
    for(let i=0;i<shapes.length;i++)for(let j=i+1;j<shapes.length;j++){
      const a=shapes[i],b=shapes[j];if(a.body===b.body)continue;
      const overlap=convexOverlap(a.world,b.world);
      if(overlap===false)continue;
      const parentChild=adjacent.has([a.body,b.body].sort().join('|'));
      const enabled=$('collision-mode').value==='on'&&!parentChild;
      result.push({a:a.name,b:b.name,body_a:a.body,body_b:b.body,overlap,parent_child:parentChild,enabled});
      for(const s of [a,b])s.mesh.material.color.setHex(enabled?0xff6767:0xffc16e);
    }
    const active=result.filter(r=>r.enabled), ignored=result.filter(r=>!r.enabled);
    $('collision-status').textContent=(result.length?result.map(r=>`${r.overlap===null?'Uncertain: ':''}${r.a} ↔ ${r.b}: ${r.enabled?'collision eligible':'IGNORED — '+(r.parent_child?'parent–child':'self-collisions OFF')}`).join('\n'):'No geometric overlap detected at this pose.')+
      '\nPreview only: excludes PhysX contact margins and hull cooking. Move slowly in the IsaacLab bench to test actual motion.';
    $('collision-summary').textContent=`${active.length} eligible / ${ignored.length} ignored overlaps`;
  }
  $('collision-mode').value=D.enabled_self_collisions?'on':'off';
  $('collision-mode').onchange=()=>{update();render();};
  return {update,snapshot:()=>result,shapes};
}
