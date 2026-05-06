from flask import Flask, render_template_string, request, jsonify
import pickle, json
import pandas as pd, numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    md = pickle.load(f)
    model, encoders, feature_names = md['model'], md['encoders'], md['feature_names']
    feat_imp = md.get('feature_importances', {})

THRESH = {
    'CGPA':{'v':7,'tip':'Focus on academics and aim for CGPA above 7.5'},
    'Coding_Skills':{'v':5,'tip':'Try to improve your coding skills. Practice DSA on LeetCode.'},
    'Communication_Skills':{'v':6,'tip':'Keep improving your communication and soft skills.'},
    'Aptitude_Test_Score':{'v':70,'tip':'Practice aptitude questions daily on IndiaBix.'},
    'Projects':{'v':3,'tip':'Build more projects to strengthen your portfolio.'},
    'Internships':{'v':1,'tip':'Apply for internships to gain real-world experience.'},
    'Certifications':{'v':2,'tip':'Get certified on Coursera/Udemy in your domain.'},
    'Backlogs':{'v':0,'tip':'Clear all backlogs ASAP — companies filter on this.'},
    'Soft_Skills_Rating':{'v':6,'tip':'Work on teamwork and leadership abilities.'},
}

RESOURCE_LINKS = {
    'CGPA':{'msg':'Focus on academic subjects and clear difficult concepts.','link':'https://www.coursera.org/courses?query=study%20skills'},
    'Coding_Skills':{'msg':'Practice coding on LeetCode, CodeChef, or HackerRank.','link':'https://leetcode.com/'},
    'Communication_Skills':{'msg':'Engage in group discussions and watch public speaking tutorials.','link':'https://www.ted.com/talks'},
    'Aptitude_Test_Score':{'msg':'Solve quantitative and logical reasoning puzzles daily.','link':'https://www.indiabix.com/'},
    'Projects':{'msg':'Build real-world applications and showcase on GitHub.','link':'https://github.com/explore'},
    'Internships':{'msg':'Apply for internships on Internshala/LinkedIn.','link':'https://www.internshala.com/'},
    'Certifications':{'msg':'Take online courses to validate your knowledge.','link':'https://www.udemy.com/'},
    'Backlogs':{'msg':'Focus on clearing pending backlogs ASAP.','link':'https://www.geeksforgeeks.org/'},
    'Soft_Skills_Rating':{'msg':'Develop interpersonal and teamwork abilities.','link':'https://www.coursera.org/courses?query=soft%20skills'},
}

PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Placement Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif;}
:root{--bg:#020617;--card:#0F172A;--card2:#131C31;--border:#1E293B;--primary:#6366F1;--cyan:#22D3EE;--pink:#D946EF;--green:#22C55E;--red:#EF4444;--text:#F8FAFC;--muted:#94A3B8;--dim:#64748B;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}

/* Thin transparent scrollbar */
::-webkit-scrollbar{width:6px;background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.15);}
*{scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.08) transparent;}

/* NAV */
.nav{background:var(--card);border-bottom:1px solid var(--border);padding:0.8rem 2rem;display:flex;align-items:center;gap:2rem;position:sticky;top:0;z-index:100;}
.nav-brand{font-size:1.1rem;font-weight:700;display:flex;align-items:center;gap:0.5rem;}
.nav-links{display:flex;gap:1.5rem;}
.nav-links a{text-decoration:none;font-size:0.9rem;font-weight:500;color:var(--dim);display:flex;align-items:center;gap:0.4rem;padding:0.4rem 0;border-bottom:2px solid transparent;transition:0.3s;}
.nav-links a.active,.nav-links a:hover{color:var(--primary);border-bottom-color:var(--primary);}

/* FULL PAGE SINGLE SCROLL */
.page-wrap{max-width:1300px;margin:0 auto;padding:2rem 2rem 4rem;}

/* TWO COL — form fills page initially, results appear beside */
.layout{display:flex;gap:0;transition:all 0.6s cubic-bezier(0.4,0,0.2,1);}
.left-col{flex:1;min-width:0;padding:0 1rem 0 0;transition:all 0.6s cubic-bezier(0.4,0,0.2,1);}
.right-col{flex:0;min-width:0;overflow:hidden;opacity:0;transform:translateX(40px);transition:all 0.6s cubic-bezier(0.4,0,0.2,1);}
.layout.expanded .left-col{flex:0 0 48%;}
.layout.expanded .right-col{flex:0 0 52%;opacity:1;transform:translateX(0);padding:0 0 0 1rem;}
@media(max-width:900px){.layout,.layout.expanded{flex-direction:column;}.left-col,.layout.expanded .left-col,.layout.expanded .right-col{flex:none;width:100%;padding:0;}.right-col{margin-top:1.5rem;}}

/* LEFT CARD */
.form-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:2rem;}
.form-title{font-size:1.5rem;font-weight:700;color:var(--cyan);margin-bottom:0.3rem;}
.form-sub{color:var(--dim);font-size:0.85rem;margin-bottom:1.5rem;}
.section-head{font-size:0.95rem;font-weight:600;margin-bottom:1.2rem;display:flex;align-items:center;gap:0.5rem;color:var(--text);}
.form-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;}
@media(max-width:700px){.form-grid{grid-template-columns:1fr 1fr;}}
.fg{display:flex;flex-direction:column;gap:0.3rem;}
.fg label{font-size:0.78rem;font-weight:500;color:var(--dim);}
.fg .iw{display:flex;align-items:center;background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:0 0.7rem;transition:0.3s;}
.fg .iw:focus-within{border-color:var(--primary);box-shadow:0 0 0 3px rgba(99,102,241,0.12);}
.fg .iw .ico{font-size:0.9rem;margin-right:0.5rem;color:var(--dim);}
.fg .iw input,.fg .iw select{background:transparent;border:none;color:var(--text);font-size:0.9rem;padding:0.65rem 0;width:100%;outline:none;}
.fg .iw select option{background:var(--card);color:var(--text);}
.predict-btn{margin-top:1.5rem;width:100%;padding:0.9rem;border:none;border-radius:12px;font-size:1rem;font-weight:600;color:white;background:linear-gradient(135deg,var(--primary),var(--pink));cursor:pointer;transition:0.3s;display:flex;align-items:center;justify-content:center;gap:0.5rem;}
.predict-btn:hover{filter:brightness(1.15);transform:translateY(-1px);}
.predict-btn:disabled{opacity:0.6;cursor:wait;}
.predict-btn .sp{display:none;width:18px;height:18px;border:2px solid rgba(255,255,255,0.3);border-top-color:white;border-radius:50%;animation:spin 0.7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.about-box{margin-top:1.2rem;background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;}
.about-box .ab-title{font-size:0.85rem;font-weight:600;display:flex;align-items:center;gap:0.4rem;margin-bottom:0.4rem;}
.about-box p{font-size:0.78rem;color:var(--dim);line-height:1.5;}

/* RIGHT — RESULT CARDS */
.result-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:1.2rem;animation:cardPop 0.5s ease both;}
@keyframes cardPop{from{opacity:0;transform:scale(0.95) translateY(10px);}to{opacity:1;transform:scale(1) translateY(0);}}
.result-card:nth-child(2){animation-delay:0.1s;}
.result-card:nth-child(3){animation-delay:0.2s;}
.result-card:nth-child(4){animation-delay:0.3s;}
.rc-head{font-size:0.85rem;font-weight:600;color:var(--muted);display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;}

/* STATUS + CIRCLE */
.status-wrap{display:flex;align-items:center;gap:1.5rem;}
.status-info{flex:1;}
.status-row{display:flex;align-items:center;gap:0.8rem;margin-bottom:0.3rem;}
.status-emoji{font-size:2.2rem;}
.status-text{font-size:2rem;font-weight:700;}
.status-msg{color:var(--dim);font-size:0.85rem;}
.s-green{color:var(--green);}.s-red{color:var(--red);}

/* RADIAL CIRCLE */
.radial{position:relative;width:120px;height:120px;flex-shrink:0;}
.radial svg{width:120px;height:120px;transform:rotate(-90deg);}
.radial .track{fill:none;stroke:var(--border);stroke-width:8;}
.radial .arc{fill:none;stroke-width:8;stroke-linecap:round;transition:stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1);}
.radial .val{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:1.3rem;font-weight:700;}

/* PROB BAR */
.prob-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;}
.prob-label{font-weight:600;font-size:0.95rem;}
.prob-pct{font-weight:700;font-size:1.3rem;}
.prob-track{height:10px;background:var(--card2);border-radius:5px;overflow:hidden;margin-bottom:0.4rem;}
.prob-fill{height:100%;border-radius:5px;transition:width 1.2s cubic-bezier(0.4,0,0.2,1);}
.prob-note{font-size:0.78rem;color:var(--dim);}

/* FI */
.fi-item{display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;}
.fi-name{width:130px;text-align:right;font-size:0.78rem;color:var(--muted);flex-shrink:0;}
.fi-track{flex:1;height:16px;background:var(--card2);border-radius:4px;overflow:hidden;}
.fi-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--primary),var(--cyan));transition:width 1.2s cubic-bezier(0.4,0,0.2,1);}

/* IMPROVE */
.imp-item{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.6rem;display:flex;align-items:flex-start;gap:0.6rem;font-size:0.85rem;color:var(--muted);}
.imp-arrow{color:var(--pink);font-weight:700;flex-shrink:0;}
.roadmap-btn{display:block;text-align:center;padding:0.8rem;border:1px solid var(--border);border-radius:12px;color:var(--text);text-decoration:none;font-weight:600;font-size:0.9rem;margin-top:1rem;transition:0.3s;}
.roadmap-btn:hover{background:var(--card2);border-color:var(--primary);}
.back-link{display:block;text-align:center;margin-top:0.8rem;color:var(--dim);text-decoration:none;font-size:0.85rem;}
.back-link:hover{color:var(--text);}
</style></head><body>
<nav class="nav"><div class="nav-brand">🎯 Placement Predictor</div>
<div class="nav-links"><a href="/" class="active">🏠 Predict</a><a href="/resources">📖 Roadmap</a></div></nav>

<div class="page-wrap">
<div class="layout" id="layout">
<div class="left-col">
<div class="form-card">
<div class="form-title">Student Placement Predictor</div>
<div class="form-sub">Enter your details to predict your placement chances</div>
<div class="section-head">🎓 Academic & Profile Details</div>
<form id="pf">
<div class="form-grid">
<div class="fg"><label>Branch</label><div class="iw"><span class="ico">🏛</span><select name="Branch"><option>CSE</option><option>IT</option><option>ECE</option><option>ME</option><option>Civil</option></select></div></div>
<div class="fg"><label>CGPA (0 - 10)</label><div class="iw"><span class="ico">🎓</span><input type="number" step="0.01" name="CGPA" placeholder="e.g. 7.0" min="0" max="10" required></div></div>
<div class="fg"><label>Internships</label><div class="iw"><span class="ico">🏢</span><input type="number" name="Internships" placeholder="e.g. 1" min="0" max="10" required></div></div>
<div class="fg"><label>Projects</label><div class="iw"><span class="ico">🔧</span><input type="number" name="Projects" placeholder="e.g. 2" min="0" max="10" required></div></div>
<div class="fg"><label>Coding Skills (1 - 10)</label><div class="iw"><span class="ico">&lt;/&gt;</span><input type="number" name="Coding_Skills" placeholder="e.g. 5" min="1" max="10" required></div></div>
<div class="fg"><label>Communication Skills (1 - 10)</label><div class="iw"><span class="ico">💬</span><input type="number" name="Communication_Skills" placeholder="e.g. 6" min="1" max="10" required></div></div>
<div class="fg"><label>Aptitude Score (1 - 100)</label><div class="iw"><span class="ico">🧠</span><input type="number" name="Aptitude_Test_Score" placeholder="e.g. 65" min="0" max="100" required></div></div>
<div class="fg"><label>Soft Skills Rating (1 - 10)</label><div class="iw"><span class="ico">⭐</span><input type="number" name="Soft_Skills_Rating" placeholder="e.g. 6" min="1" max="10" required></div></div>
<div class="fg"><label>Certifications</label><div class="iw"><span class="ico">📜</span><input type="number" name="Certifications" placeholder="e.g. 1" min="0" max="10" required></div></div>
<div class="fg"><label>Backlogs</label><div class="iw"><span class="ico">🔄</span><input type="number" name="Backlogs" placeholder="e.g. 0" min="0" max="10" required></div></div>
</div>
<input type="hidden" name="Age" value="22"><input type="hidden" name="Gender" value="Male"><input type="hidden" name="Degree" value="B.Tech">
<button type="submit" class="predict-btn" id="pbtn"><span id="btxt">🚀 Predict Status</span><div class="sp" id="sp"></div></button>
</form>
<div class="about-box"><div class="ab-title">ℹ️ About this model</div><p>This model is trained on student placement data and uses machine learning to predict placement chances based on your profile.</p></div>
</div>
</div>

<div class="right-col" id="rightCol">
<div class="result-card"><div class="rc-head">🎯 Prediction Result</div>
<div class="status-wrap">
<div class="status-info"><div class="status-row"><span class="status-emoji" id="sEmoji"></span><span class="status-text" id="sText"></span></div><div class="status-msg" id="sMsg"></div></div>
<div class="radial"><svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="52"/><circle class="arc" id="arc" cx="60" cy="60" r="52" stroke-dasharray="326.7" stroke-dashoffset="326.7"/></svg><div class="val" id="arcVal">0%</div></div>
</div></div>

<div class="result-card"><div class="prob-row"><span class="prob-label">Placement Probability</span><span class="prob-pct" id="pPct"></span></div>
<div class="prob-track"><div class="prob-fill" id="pFill" style="width:0%"></div></div>
<div class="prob-note" id="pNote"></div></div>

<div class="result-card"><div class="rc-head">💡 Key Factors (Feature Importance)</div><div id="fiBars"></div></div>

<div class="result-card" id="impCard" style="display:none"><div class="rc-head">💡 Areas for Improvement</div><div id="impList"></div>
<a href="#" class="roadmap-btn" id="rmLink">View Roadmap to Success →</a>
<a href="/" class="back-link">← Back to Predictor</a></div>
</div>
</div>
</div>

<script>
const CIRC=2*Math.PI*52;
document.getElementById('pf').addEventListener('submit',async e=>{
e.preventDefault();const btn=document.getElementById('pbtn'),bt=document.getElementById('btxt'),sp=document.getElementById('sp'),ly=document.getElementById('layout');
bt.style.display='none';sp.style.display='block';btn.disabled=true;
const fd=new FormData(e.target),data=Object.fromEntries(fd.entries());
try{const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const r=await res.json();if(r.error){alert(r.error);return;}
const placed=r.status==='Placed',pct=r.probability,clr=placed?'var(--green)':'var(--red)';

// Expand layout
ly.classList.add('expanded');

setTimeout(()=>{
// Status
document.getElementById('sEmoji').textContent=placed?'🎉':'😔';
const st=document.getElementById('sText');st.textContent=placed?'Placed':'Not Placed';st.className='status-text '+(placed?'s-green':'s-red');
document.getElementById('sMsg').textContent=r.message;

// Radial circle
const arc=document.getElementById('arc'),arcV=document.getElementById('arcVal');
arc.style.stroke=placed?'#22C55E':'#EF4444';
arc.setAttribute('stroke-dashoffset',CIRC);
arcV.textContent='0%';arcV.style.color=placed?'#22C55E':'#EF4444';
setTimeout(()=>{arc.setAttribute('stroke-dashoffset',CIRC-(pct/100)*CIRC);
// Animate number
let cur=0;const step=pct/40;const iv=setInterval(()=>{cur+=step;if(cur>=pct){cur=pct;clearInterval(iv);}arcV.textContent=cur.toFixed(1)+'%';},30);
},100);

// Prob bar
const pp=document.getElementById('pPct');pp.textContent=pct+'%';pp.style.color=placed?'var(--green)':'var(--red)';
const pf=document.getElementById('pFill');pf.style.width='0%';pf.style.background=placed?'var(--green)':'var(--red)';
setTimeout(()=>pf.style.width=pct+'%',50);
document.getElementById('pNote').textContent='You have '+pct+'% chance of getting placed';

// FI bars — all same gradient color
const fi=r.feature_importances,maxV=Math.max(...Object.values(fi));
const sorted=Object.entries(fi).sort((a,b)=>b[1]-a[1]);
let fh='';sorted.forEach(([k,v])=>{
const w=((v/maxV)*100).toFixed(1);
fh+=`<div class="fi-item"><div class="fi-name">${k.replace(/_/g,' ')}</div><div class="fi-track"><div class="fi-fill" style="width:0%" data-w="${w}%"></div></div></div>`;
});
document.getElementById('fiBars').innerHTML=fh;
setTimeout(()=>document.querySelectorAll('.fi-fill').forEach(b=>b.style.width=b.dataset.w),200);

// Improvements
const ic=document.getElementById('impCard'),il=document.getElementById('impList');
if(r.improvements&&r.improvements.length>0){
il.innerHTML=r.improvements.map(i=>`<div class="imp-item"><span class="imp-arrow">→</span>${i.tip}</div>`).join('');
document.getElementById('rmLink').href='/resources?points='+encodeURIComponent(JSON.stringify(r.weak_keys));
ic.style.display='block';
}else{ic.style.display='none';}
},300);
}catch(err){alert('Failed');}
finally{bt.style.display='inline';sp.style.display='none';btn.disabled=false;}
});
</script></body></html>"""

RESOURCES_TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Roadmap to Success</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif;}
:root{--bg:#020617;--card:#0F172A;--border:#1E293B;--primary:#6366F1;--text:#F8FAFC;--muted:#94A3B8;}
body{background:var(--bg);color:var(--text);min-height:100vh;}
::-webkit-scrollbar{width:0;background:transparent;}*{scrollbar-width:none;}
.nav{background:var(--card);border-bottom:1px solid var(--border);padding:0.8rem 2rem;display:flex;align-items:center;gap:2rem;position:sticky;top:0;z-index:100;}
.nav-brand{font-size:1.1rem;font-weight:700;}
.nav-links{display:flex;gap:1.5rem;}
.nav-links a{text-decoration:none;font-size:0.9rem;font-weight:500;color:var(--muted);display:flex;align-items:center;gap:0.4rem;padding:0.4rem 0;border-bottom:2px solid transparent;transition:0.3s;}
.nav-links a.active,.nav-links a:hover{color:var(--primary);border-bottom-color:var(--primary);}
.page{max-width:800px;margin:0 auto;padding:2rem 1.5rem 4rem;}
.page h1{font-size:2rem;font-weight:700;background:linear-gradient(135deg,#fb7185,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:0.5rem;}
.page .sub{text-align:center;color:var(--muted);margin-bottom:2rem;}
.rcard{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:1.2rem;transition:0.3s;}
.rcard:hover{border-color:var(--primary);transform:translateY(-3px);}
.rcard h3{font-size:1.1rem;color:#818cf8;margin-bottom:0.6rem;}
.rcard p{color:var(--muted);line-height:1.6;margin-bottom:1rem;}
.rcard a{display:inline-block;background:var(--primary);color:white;text-decoration:none;padding:0.5rem 1rem;border-radius:8px;font-size:0.85rem;font-weight:600;transition:0.3s;}
.rcard a:hover{filter:brightness(1.2);}
.back{display:block;text-align:center;margin-top:2rem;color:var(--muted);text-decoration:none;}.back:hover{color:var(--text);}
</style></head><body>
<nav class="nav"><div class="nav-brand">🎯 Placement Predictor</div>
<div class="nav-links"><a href="/">🏠 Predict</a><a href="/resources" class="active">📖 Roadmap</a></div></nav>
<div class="page"><h1>Roadmap to Success</h1><p class="sub">Tailored resources to help you bridge the gaps</p>
{% for point in resources %}
<div class="rcard"><h3>🚀 {{ point.title.replace('_', ' ') }}</h3><p>{{ point.msg }}</p>
<a href="{{ point.link }}" target="_blank">Explore Resources →</a></div>
{% endfor %}
<a href="/" class="back">← Back to Predictor</a></div></body></html>"""

@app.route('/')
def home():
    return render_template_string(PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        for col, le in encoders.items():
            if col in input_df.columns:
                try: input_df[col] = le.transform(input_df[col])
                except ValueError: input_df[col] = 0
        input_df = input_df[feature_names]
        for col in input_df.columns: input_df[col] = pd.to_numeric(input_df[col])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        placed_idx = list(encoders['Placement_Status'].classes_).index('Placed')
        probability = round(proba[placed_idx] * 100, 1)
        status = encoders['Placement_Status'].inverse_transform([prediction])[0]
        improvements, weak_keys = [], []
        for key, info in THRESH.items():
            val = float(data[key])
            is_weak = val > info['v'] if key == 'Backlogs' else val < info['v']
            if is_weak:
                improvements.append({'tip': info['tip']})
                weak_keys.append(key)
        message = "Congratulations! You have high chances of getting placed." if status == 'Placed' else "You have areas to improve — check suggestions below."
        return jsonify({'status':status,'message':message,'probability':probability,'feature_importances':feat_imp,'improvements':improvements,'weak_keys':weak_keys})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/resources')
def resources():
    points = json.loads(request.args.get('points', '[]'))
    selected = [{'title':p,'msg':RESOURCE_LINKS[p]['msg'],'link':RESOURCE_LINKS[p]['link']} for p in points if p in RESOURCE_LINKS]
    if not selected:
        selected = [{'title':p,'msg':v['msg'],'link':v['link']} for p,v in RESOURCE_LINKS.items()]
    return render_template_string(RESOURCES_TEMPLATE, resources=selected)

if __name__ == '__main__':
    app.run(debug=True)
