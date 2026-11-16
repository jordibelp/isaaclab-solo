/* Run with NODE_PATH pointing to an existing playwright-core installation. */
const {chromium}=require('playwright-core');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const [input,out]=process.argv.slice(2);
if(!input||!out)throw Error('Usage: node browser.cjs <offline HTML or URL> <evidence directory>');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try{
 const page=await browser.newPage({viewport:{width:1600,height:1100},acceptDownloads:true});
 const errors=[],requests=[];let checks=0;
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto(/^https?:/.test(input)?input:'file://'+path.resolve(input));
 await page.waitForFunction(()=>window.jointViewer?.getState().joint_names.length===12);
 const state=()=>page.evaluate(()=>jointViewer.getState());
 const before=await state(),names=before.joint_names,cases=[],envelopes=[];
 async function record(label){cases.push({label,state:await state(),bodies:await page.evaluate(()=>jointViewer.getBodyPositions()),bounds:await page.evaluate(()=>names.map(n=>({hard:jointViewer.getBounds(n,'physical'),soft:jointViewer.getBounds(n,'soft')})))});}
 async function checkBounds(name,kind,expected){const actual=await page.evaluate(([n,k])=>jointViewer.getBounds(n,k),[name,kind]);expected.forEach((v,i)=>assert.ok(Math.abs(actual[i]-v)<1e-10));checks++;}
 assert.equal(await page.locator('#joint-controls input[type=range]').count(),12);checks++;
 assert.equal(before.limits.use_asymmetric_hip_limits,true);checks++;
 for(let i=0;i<12;i++){
  const type=i%3,left=i%6<3,hard=type?[-3.5*180/Math.PI,3.5*180/Math.PI]:left?[-1.22*180/Math.PI,2.35*180/Math.PI]:[-2.35*180/Math.PI,1.22*180/Math.PI];
  await checkBounds(names[i],'physical',hard);await checkBounds(names[i],'soft',[hard[0]+[5,10,5][type],hard[1]-[5,10,5][type]]);
 }
 await page.screenshot({path:path.join(out,'desktop.png')});await record('default safe');
 for(const name of names){
  await page.selectOption('#focus-joint',name);
  for(const endpoint of [0,1]){await page.click('#safe');await page.getByRole('button',{name:name.replace('_joint','').replace('_',' ')+(endpoint?' maximum':' minimum'),exact:true}).click();await record(name+' hard '+endpoint);checks++;}
  await page.click('#safe');
  envelopes.push(await page.evaluate(()=>({state:jointViewer.getState(),selected,soft:bounds(selected,'soft'),positions:Array.from(pathGeometry.attributes.position.array),colors:Array.from(pathGeometry.attributes.color.array),ghosts:ghosts.map(g=>({feet:legs.map(leg=>g.getObjectByName(leg+'_foot').getWorldPosition(new THREE.Vector3()).toArray()),angles:names.map(n=>{const q= g.getObjectByName(joints[n].group.name).quaternion;return [q.x,q.y,q.z,q.w];})}))})));
 }
 await page.click('#earlier-soft');
 for(const [name,soft] of [['FL_hip_joint',[-45,45]],['FR_thigh_joint',[-125,35]],['RR_thigh_joint',[-35,125]],['RL_calf_joint',[-155,155]]])await checkBounds(name,'soft',soft);
 assert.deepEqual((await state()).limits.joint_physical_limit_left_hip,before.limits.joint_physical_limit_left_hip);checks++;
 await page.selectOption('#focus-joint','FL_thigh_joint');
 await page.click('#soft-min');assert.equal((await state()).joint_positions_deg[1],-125);checks++;
 await page.click('#soft-max');assert.equal((await state()).joint_positions_deg[1],35);checks++;
 await page.click('#sweep');await page.waitForFunction(()=>angles[1]<34);const start=await state();await page.waitForTimeout(150);assert.notEqual((await state()).joint_positions_deg[1],start.joint_positions_deg[1]);checks++;
 await page.click('#stop-sweep');const stopped=await state();await page.waitForTimeout(100);assert.deepEqual(await state(),stopped);checks++;
 await page.click('#safe');await page.click('#fit');await page.screenshot({path:path.join(out,'candidate-thigh.png')});await record('earlier soft trial');
 // Edit an exact side pair with the real form; do not replace hard limits.
 await page.fill('#soft-front_thigh-0','-118.75');await page.fill('#soft-front_thigh-1','31.25');await page.click('#apply-soft');await checkBounds('FR_thigh_joint','soft',[-118.75,31.25]);
 const valid=await state();await page.fill('#soft-front_thigh-1','');await page.click('#apply-soft');assert.match(await page.locator('#message').innerText(),/finite/);assert.deepEqual(await state(),valid);checks++;
 await page.fill('#soft-front_thigh-1','300');await page.click('#apply-soft');assert.match(await page.locator('#message').innerText(),/inside/);assert.deepEqual(await state(),valid);checks++;
 await page.evaluate(()=>jointViewer.applyLimits(DATA.task_limits,'task'));
 // Precedence is independent of hard flags. Unused deltas are ignored by production.
 await page.evaluate(()=>jointViewer.importOverrides('env.joint_soft_limit_hip=[-40,40] env.joint_soft_limit_left_hip=[-30,60] env.joint_soft_limit_thigh=[-150,150] env.joint_soft_limit_front_thigh=[-120,30] env.joint_soft_limit_calf=[-140,140] env.joint_soft_limit_hip_delta=-5'));
 await checkBounds('FL_hip_joint','soft',[-30,60]);await checkBounds('FR_hip_joint','soft',[-40,40]);await checkBounds('FL_thigh_joint','soft',[-120,30]);await checkBounds('RR_thigh_joint','soft',[-150,150]);await record('absolute precedence');
 const stable=await state();
 for(const text of ['env.joint_soft_limit_calf=[0,0]','env.joint_soft_limit_left_hip=[-500,0]','env.joint_soft_limit_thigh=[-90,]','env.joint_soft_limit_calf_delta=NaN env.joint_soft_limit_calf=null','env.use_asymmetric_hip_limits=yes','env.joint_physical_limit_left_hip=[2,1]','env.joint_physical_limit_hips=[-70,70]','env.kp=9']){
  assert.equal(await page.evaluate(text=>{try{jointViewer.importOverrides(text);return false;}catch{return true;}},text),true,text);assert.deepEqual(await state(),stable);checks++;
 }
 // Null clears old pairs, last duplicate wins; exported soft-only command round-trips.
 const command=await page.locator('#soft-overrides').inputValue();
 await page.evaluate(()=>jointViewer.applyLimits(DATA.task_limits,'task'));await page.evaluate(text=>jointViewer.importOverrides(text),command);assert.deepEqual((await state()).limits,stable.limits);checks++;
 await page.evaluate(()=>jointViewer.importOverrides('env.joint_soft_limit_hip_delta=5 env.joint_soft_limit_left_hip=None env.joint_soft_limit_hip=null env.joint_soft_limit_left_hip=[-35,50]'));
 await checkBounds('RL_hip_joint','soft',[-35,50]);
 await page.click('#freeze-soft');const frozen=await state();for(const group of ['left_hip','right_hip','front_thigh','rear_thigh','front_calf','rear_calf'])assert.ok(Array.isArray(frozen.limits['joint_soft_limit_'+group]));checks++;
 await page.evaluate(()=>jointViewer.applyLimits(DATA.task_limits,'task'));await page.click('#clear-soft');assert.deepEqual((await state()).limits,before.limits);checks++;
 // Legacy preset and saved poses remain usable.
 await page.evaluate(()=>jointViewer.applyLimits(DATA.command_limits,'command'));await checkBounds('FL_hip_joint','soft',[-50,50]);await record('legacy command');
 const old=await state();delete old.limits.use_asymmetric_hip_limits;for(const k of Object.keys(old.limits))if(k.startsWith('joint_soft_limit_')&&!k.endsWith('_delta'))delete old.limits[k];
 await page.evaluate(p=>jointViewer.loadPose(p),old);await checkBounds('RR_hip_joint','physical',[-70,70]);
 const saved=await state();await page.evaluate(()=>jointViewer.setAngles(Array(12).fill(0)));await page.evaluate(p=>jointViewer.loadPose(p),saved);assert.deepEqual(await state(),saved);checks++;
 await page.locator('summary').filter({hasText:'Save / load'}).click();const downloadPromise=page.waitForEvent('download');await page.click('#save-pose');const download=await downloadPromise;await download.saveAs(path.join(out,'pose.json'));assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out,'pose.json'),'utf8')),saved);checks++;
 // Asymmetric calves and hard editor controls.
 await page.locator('summary').filter({hasText:'Edit limits'}).click();await page.selectOption('#preset','task');
 await page.uncheck('#asymmetric-hip');await page.check('#asymmetric-thigh');await page.fill('#limit-front_thigh-0','-145');await page.fill('#limit-front_thigh-1','55');await page.click('#apply-limits');await checkBounds('FL_thigh_joint','physical',[-145,55]);await checkBounds('FL_hip_joint','physical',before.limits.joint_physical_limit_hip);await record('hard form');
 await page.evaluate(()=>jointViewer.applyLimits(DATA.task_limits,'task'));await page.evaluate(()=>jointViewer.importOverrides('env.use_asymmetric_calf_limits=True env.joint_physical_limit_front_calf=[-170,20] env.joint_physical_limit_rear_calf=[-20,170]'));
 await checkBounds('FR_calf_joint','soft',[-165,15]);await checkBounds('RR_calf_joint','soft',[-15,165]);await record('asymmetric calf');
 // Mixed pose and body rotations, plus soft-mode clamping.
 await page.evaluate(()=>jointViewer.applyLimits(DATA.task_limits,'task'));
 await page.locator('summary').filter({hasText:'Body position'}).click();
 for(let k=0;k<6;k++){
  await page.fill('#base-pitch',String(k*25-55));await page.fill('#base-yaw',String(k*35-70));
  await page.evaluate(k=>jointViewer.setAngles(names.map((n,i)=>{const [lo,hi]=bounds(n,'physical');return lo+(hi-lo)*(((k+1)*(i+3)*.173)%1);})),k);await record('mixed tilted '+k);
 }
 await page.selectOption('#range-mode','soft');await record('soft clamp');await page.selectOption('#range-mode','physical');
 await page.click('#reset-body');await page.click('#safe');await page.click('#earlier-soft');
 for(const id of ['front','side','top','orbit','fit']){await page.click('#'+id);checks++;}
 const canvas=page.locator('#stage canvas').first();const box=await canvas.boundingBox();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+50,box.y+box.height/2+30);await page.mouse.up();await page.mouse.wheel(0,-50);checks++;
 await page.uncheck('#show-envelope');assert.equal(await page.evaluate(()=>pathLine.visible),false);await page.check('#show-envelope');checks++;
 await page.setViewportSize({width:390,height:850});await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);checks++;
 assert.deepEqual(errors,[]);if(!/^https?:/.test(input))assert.equal(requests.filter(x=>/^https?:/.test(x)).length,0);checks++;
 fs.writeFileSync(path.join(out,'parity-cases.json'),JSON.stringify(cases));fs.writeFileSync(path.join(out,'envelopes.json'),JSON.stringify(envelopes));
 fs.writeFileSync(path.join(out,'browser-result.json'),JSON.stringify({checks,poses:cases.length,envelopes:envelopes.length,errors,httpRequests:requests.filter(x=>/^https?:/.test(x))},null,2));
 console.log(`PASS: ${checks} browser checks; ${cases.length} FK/config cases; ${envelopes.length} endpoint/path cases; no JS errors.`);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
