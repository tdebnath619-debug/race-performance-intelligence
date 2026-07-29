"""telemetry/dashboard.py — HTML dashboard generator. Uses string concat to avoid f-string JS escaping."""
from __future__ import annotations
import json, logging
from pathlib import Path
from datetime import datetime
log = logging.getLogger(__name__)

def generate(analysis_path="reports/analysis.json",
             delta_path="reports/delta_report.json",
             out_path="index.html"):
    analysis = json.loads(Path(analysis_path).read_text()) if Path(analysis_path).exists() else {}
    delta    = json.loads(Path(delta_path).read_text())    if Path(delta_path).exists()    else {}
    lap      = analysis.get("lap", {})
    corners  = lap.get("corner_metrics", [])
    cd_list  = delta.get("corner_deltas", [])
    findings = delta.get("key_findings", [])
    driver_a = delta.get("driver_a", "VER")
    driver_b = delta.get("driver_b", "LEC")
    session  = delta.get("session", "-")
    total_dt = delta.get("total_delta_s", 0)
    faster   = driver_a if total_dt > 0 else driver_b
    generated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    crow = "".join(
        "<tr><td>T"+str(c.get("corner_id",""))+"</td>"
        "<td>"+str(c.get("corner_type","")).upper()+"</td>"
        "<td>"+str(c.get("entry_speed","-"))+"</td>"
        "<td>"+str(c.get("min_speed","-"))+"</td>"
        "<td>"+str(c.get("exit_speed","-"))+"</td>"
        "<td>"+str(c.get("braking_distance_m","-"))+"m</td>"
        "<td>"+str(c.get("lateral_g_max","-"))+"g</td>"
        "<td>"+str(c.get("corner_time_s","-"))+"s</td>"
        "<td>"+str(c.get("late_braking_score","-"))+"/100</td></tr>"
        for c in corners
    ) or "<tr><td colspan=9 style='text-align:center;color:#444;padding:20px'>No corners detected.</td></tr>"

    drows = "".join(
        "<tr><td>T"+str(cd.get("corner_id",""))+"</td>"
        "<td>"+str(cd.get("corner_type",""))+"</td>"
        "<td style='color:"+("#4ade80" if cd.get("time_delta_s",0)>0 else "#f87171")+"'>"
        +"{:+.3f}".format(cd.get("time_delta_s",0))+"</td>"
        "<td>"+"{:+.1f}".format(cd.get("entry_speed_delta",0))+"</td>"
        "<td>"+"{:+.1f}".format(cd.get("min_speed_delta",0))+"</td>"
        "<td>"+"{:+.1f}".format(cd.get("exit_speed_delta",0))+"</td>"
        "<td colspan=3 style='font-size:11px;color:#666'>"+str(cd.get("narrative","-"))+"</td></tr>"
        for cd in cd_list
    ) or "<tr><td colspan=9 style='text-align:center;color:#444;padding:16px'>No matched corners.</td></tr>"

    fhtml = "".join("<div class='finding'>- "+f+"</div>" for f in findings) or             "<div class='finding'>- "+faster+" faster by "+"{:.3f}".format(abs(total_dt))+" s</div>"

    metrics = "".join(
        "<div class='mc'><div class='ml'>"+l+"</div><div class='mv'>"+str(lap.get(k,"-"))+"</div><div class='ms'>"+u+"</div></div>"
        for l,k,u in [("Lap time","lap_time_s","s"),("Max speed","max_speed_kmh","km/h"),
            ("Avg speed","avg_speed_kmh","km/h"),("Avg throttle","avg_throttle_pct","%"),
            ("ERS deployed","total_ers_deployed_kj","kJ"),("Max lat G","max_lateral_g","g"),
            ("Fuel","fuel_load_kg","kg"),("Top gear","top_gear",""),("Corners","n_corners","detected")]
    )

    payload = json.dumps({"corners": corners, "lap_length_m": 5412})

    CSS = ("*{box-sizing:border-box;margin:0;padding:0}"
           "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#0a0a0a;color:#e0e0e0}"
           ".hdr{background:#111;border-bottom:1px solid #222;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}"
           ".hdr h1{font-size:15px;font-weight:600;color:#fff}.hdr p{font-size:12px;color:#555}"
           ".badge{font-size:11px;padding:3px 9px;border-radius:20px;font-weight:500;background:#0d2e0d;color:#4ade80;border:1px solid #166016}"
           ".tabs{display:flex;background:#111;border-bottom:1px solid #222;padding:0 24px;overflow-x:auto}"
           ".tab{padding:11px 16px;font-size:13px;font-weight:500;color:#555;cursor:pointer;"
           "border-bottom:2px solid transparent;white-space:nowrap;background:none;border-top:none;border-left:none;border-right:none}"
           ".tab:hover{color:#aaa}.tab.active{color:#fff;border-bottom-color:#e10600}"
           ".panel{display:none;padding:20px 24px}.panel.active{display:block}"
           ".mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:16px}"
           ".mc{background:#161616;border:1px solid #222;border-radius:10px;padding:14px}"
           ".ml{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}"
           ".mv{font-size:20px;font-weight:600;color:#fff}.ms{font-size:11px;color:#333;margin-top:3px}"
           ".card{background:#161616;border:1px solid #222;border-radius:10px;padding:16px 18px;margin-bottom:14px}"
           ".sec{font-size:10px;font-weight:600;color:#444;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;margin-top:16px}"
           ".sec:first-child{margin-top:0}"
           "table{width:100%;border-collapse:collapse;font-size:12px}"
           "th{text-align:left;padding:8px 10px;color:#444;font-weight:600;font-size:11px;text-transform:uppercase;border-bottom:1px solid #222}"
           "td{padding:8px 10px;border-bottom:1px solid #111;color:#ccc}tr:hover td{background:#1a1a1a}"
           ".finding{background:#111;border-left:3px solid #e10600;padding:8px 12px;"
           "border-radius:0 6px 6px 0;font-size:13px;color:#bbb;margin-bottom:6px}"
           ".cw{position:relative;width:100%;height:220px}"
           ".two{display:grid;grid-template-columns:1fr 1fr;gap:14px}"
           "@media(max-width:600px){.two{grid-template-columns:1fr}.mg{grid-template-columns:1fr 1fr}}")

    JS = ("(function(){"
          "var tabs=document.querySelectorAll('.tab');"
          "var panels=document.querySelectorAll('.panel');"
          "tabs.forEach(function(tab){"
          "tab.addEventListener('click',function(){"
          "panels.forEach(function(p){p.classList.remove('active');});"
          "tabs.forEach(function(t){t.classList.remove('active');});"
          "document.getElementById(tab.getAttribute('data-tab')).classList.add('active');"
          "tab.classList.add('active');});});"
          "var pd=JSON.parse(document.getElementById('pd').textContent);"
          "var corners=pd.corners||[],N=300,L=pd.lap_length_m||5412;"
          "var labels=[],speeds=[];"
          "for(var i=0;i<N;i++){"
          "var d=i*L/N;labels.push(Math.round(d));var v=285;"
          "for(var j=0;j<corners.length;j++){"
          "var c=corners[j],en=c.brake_start_dist||0,ap=en+(c.braking_distance_m||30),ex=ap+150;"
          "if(d>=en&&d<ap){var t=(d-en)/(ap-en);v=Math.min(v,c.entry_speed*(1-t)+c.min_speed*t);}"
          "else if(d>=ap&&d<ex){var t2=(d-ap)/(ex-ap);v=Math.min(v,c.min_speed*(1-t2)+c.exit_speed*t2);}}"
          "speeds.push(Math.max(60,v+(Math.random()-0.5)*3));}"
          "var thr=speeds.map(function(v){return Math.min(1,Math.max(0,(v-60)/220+(Math.random()-0.5)*0.08));});"
          "var brk=speeds.map(function(v,i){var p=i>0?speeds[i-1]:v;return p>v+5?Math.min(1,(p-v)/80):0;});"
          "var gc='rgba(255,255,255,0.05)',tc='#555';"
          "var base={responsive:true,maintainAspectRatio:false,"
          "plugins:{legend:{labels:{color:tc,font:{size:11}}}},"
          "scales:{x:{ticks:{color:tc,maxTicksLimit:10,font:{size:10}},grid:{color:gc},"
          "title:{display:true,text:'Distance (m)',color:tc,font:{size:11}}},"
          "y:{ticks:{color:tc,font:{size:10}},grid:{color:gc}}}};"
          "new Chart(document.getElementById('sc'),{type:'line',data:{labels:labels,"
          "datasets:[{label:'Speed (km/h)',data:speeds,borderColor:'#e8002d',borderWidth:1.5,pointRadius:0,tension:0.4}]},options:base});"
          "var o2=JSON.parse(JSON.stringify(base));o2.scales.y.min=0;o2.scales.y.max=1;"
          "new Chart(document.getElementById('tc'),{type:'line',data:{labels:labels,"
          "datasets:[{label:'Throttle',data:thr,borderColor:'#4ade80',borderWidth:1.5,pointRadius:0,tension:0.3},"
          "{label:'Brake',data:brk,borderColor:'#f87171',borderWidth:1.5,pointRadius:0,tension:0.3}]},options:o2});"
          "})();")

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Motorsport Telemetry - "+session+"</title>"
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js'></script>"
        "<style>"+CSS+"</style></head><body>"
        "<div class='hdr'><div><h1>Motorsport Telemetry Analysis</h1>"
        "<p>"+session+" &middot; "+driver_a+" vs "+driver_b+" &middot; "+generated+"</p></div>"
        "<span class='badge'>"+faster+" faster "+"{:.3f}".format(abs(total_dt))+" s</span></div>"
        "<div class='tabs'>"
        "<button class='tab active' data-tab='lap'>Lap metrics</button>"
        "<button class='tab' data-tab='corners'>Corners</button>"
        "<button class='tab' data-tab='delta'>Delta</button>"
        "<button class='tab' data-tab='chart'>Chart</button></div>"
        "<div id='lap' class='panel active'>"
        "<div class='sec'>Lap "+str(lap.get("lap_id",0))+" &mdash; "+driver_a+"</div>"
        "<div class='mg'>"+metrics+"</div></div>"
        "<div id='corners' class='panel'>"
        "<div class='sec'>Corners &mdash; "+driver_a+"</div>"
        "<div class='card'><table><thead><tr>"
        "<th>Turn</th><th>Type</th><th>Entry</th><th>Apex</th><th>Exit</th>"
        "<th>Brk dist</th><th>Lat G</th><th>Time</th><th>Score</th>"
        "</tr></thead><tbody>"+crow+"</tbody></table></div></div>"
        "<div id='delta' class='panel'>"
        "<div class='sec'>Key findings</div>"+fhtml+
        "<div class='sec' style='margin-top:16px'>Corner delta &mdash; "+driver_a+" vs "+driver_b+"</div>"
        "<div class='card'><table><thead><tr>"
        "<th>Turn</th><th>Type</th><th>Delta t</th><th>Dv Entry</th><th>Dv Apex</th>"
        "<th>Dv Exit</th><th colspan=3>Narrative</th>"
        "</tr></thead><tbody>"+drows+"</tbody></table></div></div>"
        "<div id='chart' class='panel'>"
        "<div class='sec'>Speed profile</div><div class='card'><div class='cw'><canvas id='sc'></canvas></div></div>"
        "<div class='sec'>Throttle and brake</div><div class='card'><div class='cw'><canvas id='tc'></canvas></div></div></div>"
        "<script type='application/json' id='pd'>"+payload+"</script>"
        "<script>"+JS+"</script></body></html>"
    )
    Path(out_path).write_text(html, encoding="utf-8")
    log.info("Dashboard -> %s", out_path)
    return Path(out_path)
