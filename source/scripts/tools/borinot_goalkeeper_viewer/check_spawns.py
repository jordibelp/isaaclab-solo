#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check spawn markers, stage activation and exact/noisy robot reset previews."""
import argparse
import itertools
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("page",help="Standalone HTML path or live viewer URL")
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
checks,errors,requests=[],[],[]


def check(name,condition):
    assert condition,name
    checks.append(name)


with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':1450,'height':1000})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:requests.append(r.url))
    offline=not args.page.startswith('http')
    page.goto(Path(args.page).resolve().as_uri() if offline else args.page)
    page.wait_for_function('window.viewer')
    snapshot=lambda:page.evaluate('viewer.snapshot()')
    initial=snapshot();cfg=initial['config']
    for level in range(len(cfg['shot_half_extents_yz'])):
        page.locator('#level').select_option(str(level))
        active=cfg['reset_grid_enabled'] and level>=cfg['reset_grid_start_level']
        count=len(cfg['reset_grid_y_offsets_m'])*len(cfg['reset_grid_z_offsets_m']) if active else 1
        snap=snapshot()
        check(f'phase {level} activation',snap['spawn']['active']==active)
        check(f'phase {level} marker and menu counts',len(snap['spawn']['locations'])==count and len(snap['spawn']['markers'])==count and page.locator('#spawn-location option').count()==count)
    page.locator('#level').select_option(str(cfg['reset_grid_start_level']))
    expected=[[cfg['robot_home'][0],cfg['robot_home'][1]+y,cfg['robot_home'][2]+z]
              for z,y in itertools.product(cfg['reset_grid_z_offsets_m'],cfg['reset_grid_y_offsets_m'])]
    positions=[]
    for i,position in enumerate(expected):
        page.locator('#yaw-num').fill('12')
        page.locator('#joint0-num').fill('25')
        page.locator('#spawn-location').select_option(str(i))
        snap=snapshot();state=snap['state']
        actual=[state[k] for k in ('x','y','z')]
        check(f'location {i} exact base centre',max(abs(a-b) for a,b in zip(actual,position))<1e-10)
        check(f'location {i} restores config yaw and joints',abs(state['yaw']+90)<1e-6 and state['joint0']==0 and state['joint1']==0)
        marker=snap['spawn']['markers'][i]['vertices']
        centroid=[sum(marker[k::3])/4 for k in range(3)]
        check(f'location {i} actual marker geometry',max(abs(a-b) for a,b in zip(centroid,position))<1e-6)
        positions.append({'position':position,'paddle':snap['paddle']})
        # Force a known uniform cell and known continuous XYZ noise, independently.
        page.evaluate('(v)=>{window.originalRandom=Math.random;let i=0;Math.random=()=>v[i++];}',[(i+.5)/len(expected),.25,.8,.9])
        page.locator('#spawn-sample').click()
        page.evaluate('Math.random=window.originalRandom')
        state=snapshot()['state']
        noisy=[position[0]-.5*cfg['reset_xy_noise_m'],position[1]+.6*cfg['reset_xy_noise_m'],position[2]+.8*cfg['reset_z_noise_m']]
        check(f'location {i} exact sampling with continuous noise',max(abs(state[k]-v) for k,v in zip(('x','y','z'),noisy))<1e-10)
    # Every link/tool translates by the same grid displacement, independently of FK.
    for i,point in enumerate(positions):
        diff=[point['paddle'][k]-positions[0]['paddle'][k] for k in range(3)]
        expected_diff=[point['position'][k]-positions[0]['position'][k] for k in range(3)]
        check(f'location {i} full robot tool translation',max(abs(a-b) for a,b in zip(diff,expected_diff))<1e-7)
    page.locator('#spawn-markers').uncheck()
    check('marker toggle',not snapshot()['spawn']['visible'])
    page.locator('#spawn-markers').check()
    page.locator('#level').select_option('0')
    page.locator('#spawn-sample').click()
    state=snapshot()['state']
    check('early reset stays near home',[abs(state[k]-cfg['robot_home'][i])<=(cfg['reset_z_noise_m'] if i==2 else cfg['reset_xy_noise_m']) for i,k in enumerate(('x','y','z'))]==[True]*3)
    check('source config never mutated',snapshot()['config']==cfg)
    page.locator('#home').click()
    page.locator('#level').select_option(str(cfg['reset_grid_start_level']))
    page.locator('[data-view="front"]').click()
    page.locator('#area').uncheck()
    page.locator('#trajectory').uncheck()
    page.locator('aside').evaluate('e=>e.scrollTop=document.getElementById("spawn-editor").offsetTop-90')
    page.screenshot(path=str(args.output.with_suffix('.png')))
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(200)
    page.locator('#spawn-location').select_option(str(len(expected)-1))
    check('mobile dropdown moves robot',abs(snapshot()['state']['z']-expected[-1][2])<1e-10)
    check('mobile no horizontal overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
    check('no JavaScript errors',not errors)
    check('no external resources',not any(r.startswith('http') for r in requests) if offline else all(r.startswith(args.page.rstrip('/')) for r in requests))
    browser.close()
args.output.write_text(json.dumps({'passed':True,'checks':checks,'errors':errors,'requests':requests},indent=2)+'\n')
print('Spawn browser assertions passed:',len(checks))
