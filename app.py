from flask import Flask, render_template_string, request, jsonify
import pickle, json
import pandas as pd, numpy as np
import pdfplumber, re, os, tempfile, io

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

def _clamp_number(value, low, high, decimals=False):
    try:
        value = re.sub(r'\s+', '', str(value))
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num < low or num > high:
        return None
    num = max(low, min(high, num))
    return f"{num:.2f}".rstrip('0').rstrip('.') if decimals else str(int(round(num)))

def _score_from_patterns(text, patterns, low, high, decimals=False):
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            score = _clamp_number(match.group(1), low, high, decimals)
            if score is not None:
                return score
    return None

def _count_from_patterns(text, patterns, max_value=10):
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            count = _clamp_number(match.group(1), 0, max_value)
            if count is not None:
                return count
    return None

def _section_lines(lines, headings):
    stop_headings = {
        'education', 'experience', 'work experience', 'internship', 'internships',
        'projects', 'project experience', 'certifications', 'certification',
        'certificates', 'skills', 'technical skills', 'achievements', 'awards',
        'summary', 'profile', 'objective', 'contact', 'languages', 'interests',
        'extra curricular', 'extracurricular', 'positions of responsibility'
    }
    selected = []
    in_section = False
    for raw_line in lines:
        line = raw_line.strip()
        clean = re.sub(r'[^a-z0-9 &/+-]', '', line.lower()).strip()
        if not clean:
            continue
        is_heading = clean in stop_headings or (len(clean) <= 35 and clean.isalpha())
        if any(h == clean or clean.startswith(h + ' ') for h in headings):
            in_section = True
            continue
        if in_section and is_heading:
            break
        if in_section:
            selected.append(line)
    return selected

def _count_section_items(lines, headings, max_value=10):
    count = 0
    bullet_re = re.compile(r'^(\s*[-*•]|\s*\d+[.)])\s+')
    item_lines = []
    for line in _section_lines(lines, headings):
        compact = line.strip()
        item_text = bullet_re.sub('', compact).strip()
        if re.match(r'^(built|created|developed|implemented|used|using|worked|designed|integrated|managed|optimized|deployed)\b', item_text, re.I):
            continue
        if bullet_re.match(compact):
            item_lines.append(compact)
            continue
        if len(compact) <= 90:
            item_lines.append(compact)
    count = len(item_lines)
    return str(min(count, max_value)) if count > 0 else None

def _extract_cgpa(text):
    normalized = re.sub(r'\s+', ' ', text)
    label_matches = list(re.finditer(r'\b(?:cgpa|gpa|cpi|grade point average)\b', normalized, re.I))
    if not label_matches:
        return None

    number_matches = []
    for match in re.finditer(r'(?<!\d)([0-9]+(?:\s*\.\s*[0-9]+)?)(?!\d)', normalized):
        score = _clamp_number(match.group(1), 0, 10, decimals=True)
        if not score:
            continue
        value = float(score)
        if value < 4 and '.' not in score:
            continue
        number_matches.append((match.start(), score))

    best = None
    for label in label_matches:
        for pos, score in number_matches:
            distance = abs(pos - label.end())
            if distance > 80:
                continue
            is_decimal = '.' in score
            rank = (0 if is_decimal else 1, distance)
            if best is None or rank < best[0]:
                best = (rank, score)
    return best[1] if best else None
    return None

def _count_certifications(lines):
    section_count = _count_section_items(lines, ['certification', 'certifications', 'certificate', 'certificates'])
    if section_count:
        return section_count

    count = 0
    for line in lines:
        clean = re.sub(r'[^a-z]', '', line.lower())
        if clean in {'certification', 'certifications', 'certificate', 'certificates'}:
            continue
        if re.search(r'\b(certification|certificate)\b', line, re.I):
            count += 1
    return str(min(count, 10)) if count else None

def _coding_rating_from_questions(text):
    matches = re.findall(
        r'(\d{2,4})\s*\+?\s*(?:coding\s*)?(?:questions|problems|dsa\s*problems|leetcode|codechef|hackerrank)',
        text,
        re.I
    )
    if not matches:
        return None
    questions = max(int(m) for m in matches)
    if questions >= 400:
        return '9'
    if questions >= 300:
        return '8.5'
    if questions >= 200:
        return '7.5'
    if questions >= 150:
        return '7'
    if questions >= 100:
        return '6.5'
    if questions >= 70:
        return '6'
    if questions >= 50:
        return '4'
    return '3'

def _detect_branch(text):
    branch_patterns = [
        ('AIML', r'\b(aiml|ai\s*ml|ai[-\s]*ml|artificial intelligence and machine learning|artificial intelligence\s*&\s*machine learning)\b'),
        ('CSE', r'\b(cse|computer science|computer engineering|cs\b|b\.?tech\s+cs)\b'),
        ('IT', r'\b(information technology|b\.?tech\s+it|it\b)\b'),
        ('ECE', r'\b(ece|electronics and communication|electronics & communication)\b'),
        ('ME', r'\b(mechanical engineering|mechanical|me\b)\b'),
        ('Civil', r'\b(civil engineering|civil)\b'),
    ]
    for branch, pattern in branch_patterns:
        if re.search(pattern, text, re.I):
            return branch
    return None

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
.up-box{margin:1.2rem 0;padding:1rem;background:var(--card2);border:1px dashed var(--border);border-radius:12px;text-align:center;transition:0.3s;}.up-box:hover{border-color:var(--primary);}.up-box label{display:block;font-size:0.8rem;color:var(--dim);margin-bottom:0.5rem;cursor:pointer;}.up-box input{display:none;}.up-box .up-btn{font-size:0.85rem;font-weight:600;color:var(--primary);display:flex;align-items:center;justify-content:center;gap:0.4rem;cursor:pointer;}.up-status{font-size:0.75rem;margin-top:0.4rem;color:var(--cyan);display:none;}
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
<div class="fg"><label>Branch</label><div class="iw"><span class="ico">🏛</span><select name="Branch"><option>CSE</option><option>AIML</option><option>IT</option><option>ECE</option><option>ME</option><option>Civil</option></select></div></div>
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
<div class="up-box">
<label>📄 Upload Resume (optional)</label>
<div class="up-btn" onclick="document.getElementById('res').click()"><span>📂 Select PDF</span></div>
<input type="file" id="res" accept=".pdf" onchange="uploadResume(this)">
<div id="sk-tags" style="margin-top:0.5rem;display:flex;flex-wrap:wrap;gap:0.4rem;justify-content:center;"></div>
<div class="up-status" id="ustat"></div>
</div>
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
async function uploadResume(input){
const f=input.files[0];if(!f)return;
const st=document.getElementById('ustat'), sk=document.getElementById('sk-tags');
st.textContent='⏳ Analyzing...';st.style.display='block';
const fd=new FormData();fd.append('file',f);
try{
const res=await fetch('/upload/',{method:'POST',body:fd});
const d=await res.json();if(d.error){st.textContent='❌ '+d.error;return;}st.textContent='✅ '+d.message;
if(d.skills) sk.innerHTML=d.skills.map(s=>`<span style="font-size:0.7rem;background:var(--primary);padding:2px 6px;border-radius:4px;">${s}</span>`).join('');
if(d.extracted_data){for(const [k,v] of Object.entries(d.extracted_data)){const el=document.querySelector(`[name="${k}"]`);if(el)el.value=v;}}
}catch(e){st.textContent='❌ Upload failed';}
}
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

@app.route('/upload/', methods=['POST'])
def upload_resume():
    if 'file' not in request.files: return jsonify({'error':'No file'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.pdf'): return jsonify({'error':'Only PDF files accepted'}), 400
    try:
        # Read directly from memory to avoid Windows file-locking issues
        resume_bytes = io.BytesIO(f.read())
        text = ''
        with pdfplumber.open(resume_bytes) as pdf:
            for page in pdf.pages:
                pg = page.extract_text()
                if pg: text += pg + '\n'
        text_lower = text.lower()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # --- Extract skills ---
        SKILL_LIST = ['python','java','c++','c','sql','javascript','html','css','react','node.js',
                      'django','flask','machine learning','deep learning','data science','tensorflow',
                      'pytorch','pandas','numpy','git','docker','kubernetes','aws','azure','gcp',
                      'mongodb','mysql','postgresql','linux','r','matlab','excel','power bi','tableau',
                      'natural language processing','nlp','computer vision','opencv','scikit-learn',
                      'spring boot','angular','vue','typescript','php','ruby','go','rust','swift','kotlin']
        found_skills = []
        for skill in SKILL_LIST:
            pattern = r'(?<![a-z0-9+#.])' + re.escape(skill) + r'(?![a-z0-9+#.])'
            if re.search(pattern, text_lower):
                found_skills.append(skill.title())
        # --- Extract form fields ---
        extracted = {}
        branch = _detect_branch(text_lower)
        if branch:
            extracted['Branch'] = branch

        cgpa = _extract_cgpa(text)
        if cgpa:
            extracted['CGPA'] = cgpa

        internships = _count_from_patterns(text_lower, [
            r'(\d+)\s*(?:\+?\s*)?(?:internships?|internship experience)',
            r'(?:internships?|internship experience)\s*(?:completed|done|:|-)?\s*(\d+)',
        ])
        if not internships:
            internships = _count_section_items(lines, ['internship', 'internships'])
        if not internships:
            intern_count = len(re.findall(r'\bintern(?:ship)?\b', text_lower))
            if intern_count > 0: internships = str(min(intern_count, 10))
        if internships:
            extracted['Internships'] = internships

        projects = _count_section_items(lines, ['project', 'projects'], max_value=4)
        if projects:
            extracted['Projects'] = projects

        certifications = _count_certifications(lines)
        if certifications:
            extracted['Certifications'] = certifications

        backlogs = _score_from_patterns(text_lower, [
            r'(?:backlogs?|arrears?)\s*(?:standing|active|current|:|-)?\s*(\d+)',
            r'(\d+)\s*(?:active\s*)?(?:backlogs?|arrears?)',
        ], 0, 10)
        if backlogs:
            extracted['Backlogs'] = backlogs
        elif re.search(r'\b(no|zero|nil)\s+(?:active\s*)?(?:backlogs?|arrears?)\b', text_lower):
            extracted['Backlogs'] = '0'

        aptitude = _score_from_patterns(text_lower, [
            r'(?:aptitude|aptitude test|quantitative aptitude)\s*(?:score|rating|:|-)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:/|out of)\s*100\s*(?:aptitude|aptitude test)',
        ], 0, 100)
        if aptitude:
            extracted['Aptitude_Test_Score'] = aptitude

        communication = _score_from_patterns(text_lower, [
            r'(?:communication skills?|communication)\s*(?:score|rating|:|-)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:/|out of)\s*10\s*(?:communication skills?|communication)',
        ], 1, 10)
        if communication:
            extracted['Communication_Skills'] = communication

        soft_skills = _score_from_patterns(text_lower, [
            r'(?:soft skills?|soft skill rating)\s*(?:score|rating|:|-)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:/|out of)\s*10\s*(?:soft skills?)',
        ], 1, 10)
        if soft_skills:
            extracted['Soft_Skills_Rating'] = soft_skills

        coding_skill = _coding_rating_from_questions(text_lower)
        if not coding_skill:
            coding_skill = _score_from_patterns(text_lower, [
                r'(?:coding skills?|programming skills?|technical skills?)\s*(?:score|rating|:|-)?\s*([0-9]+(?:\.[0-9]+)?)',
                r'([0-9]+(?:\.[0-9]+)?)\s*(?:/|out of)\s*10\s*(?:coding skills?|programming skills?)',
            ], 1, 10, decimals=True)
        if not coding_skill and found_skills:
            coding_skill = str(min(10, max(1, len(found_skills))))
        if coding_skill:
            extracted['Coding_Skills'] = coding_skill
        return jsonify({
            "skills": found_skills if found_skills else ["No specific skills detected"],
            "message": "Resume analyzed successfully",
            "extracted_data": extracted
        })
    except Exception as e:
        return jsonify({'error': f'Failed to parse resume: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
