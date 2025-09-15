#!/usr/bin/env python
"""Minimal FDV Report v2 runner (restored).
Rollback build: numeric RBER limit only; no variability/dispositions; skips MONITOR/SHMOO FDV OUTPUT lines.
"""
from __future__ import annotations

import json, os, tempfile, threading, uuid, re, datetime, time
from pathlib import Path
from typing import Dict, List
from flask import Flask, Response, flash, redirect, render_template, request, url_for
from flask import Flask, Response, flash, redirect, render_template, request, url_for, jsonify
from pathlib import Path as _Path
import threading as _threading
from fdv_report2_webapp import (
	stats_by_fdv_with_splits, stats_by_testname_selected, _parse_fdv_selector,
	_get_split_tuple, _extract_wl_or_page, _get_rber, derive_testname,
)
try:  # optional accelerated parser
	from process_fdv.core import process_file, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT  # type: ignore
except Exception:  # pragma: no cover
	process_file = None  # type: ignore
	PREFIX_DEFAULT = "FDV OUTPUT"  # type: ignore
	IGNORE_VALUE_DEFAULT = None  # type: ignore

app = Flask(__name__)
app.secret_key = os.environ.get('FDV_REPORT2_SECRET', 'dev-secret')
CACHE: Dict[str, Dict] = {}
# Minimal jobs registry for listing similar to full app
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()

_PERSIST_BASE = Path(os.environ.get('FDV_PERSIST_BASE', str(Path.home() / '.fdv_persist')))

def _persist_dir(token: str) -> Path:
	p = _PERSIST_BASE / token
	p.mkdir(parents=True, exist_ok=True)
	return p

def _load_json_helper(p: Path) -> Dict[str,str]:
	try:
		with open(p,'r',encoding='utf-8') as f:
			return json.load(f) or {}
	except Exception:
		return {}

def _save_json_helper(p: Path, data: Dict[str,str]):
	try:
		tmp = p.with_suffix('.tmp')
		with open(tmp,'w',encoding='utf-8') as f:
			json.dump(data,f,indent=2,sort_keys=True)
		os.replace(tmp,p)
	except Exception:
		pass

def _dispositions_path(token: str) -> Path:
	return _persist_dir(token) / 'dispositions.json'

def _comments_path(token: str) -> Path:
	return _persist_dir(token) / 'comments.json'

def _load_dispositions(token: str) -> Dict[str,str]:
	return _load_json_helper(_dispositions_path(token))

def _save_dispositions(token: str, data: Dict[str,str]):
	_save_json_helper(_dispositions_path(token), data)

def _load_comments(token: str) -> Dict[str,str]:
	return _load_json_helper(_comments_path(token))

def _save_comments(token: str, data: Dict[str,str]):
	_save_json_helper(_comments_path(token), data)

def _resolve_job_token(job_id: str) -> str | None:
	try:
		with JOBS_LOCK:
			rec = JOBS.get(job_id)
			if rec:
				return rec.get('token')  # type: ignore[return-value]
	except Exception:
		pass
	return None

def _list_files(root: Path) -> List[Path]:
	if root.is_file():
		return [root]
	out: List[Path] = []
	for r, _d, files in os.walk(root):
		for n in sorted(files):
			fp = Path(r) / n
			if fp.is_file():
				out.append(fp)
	return out

def _start_job(token: str, files: List[Path], used_dir: str, *, passfail_mode: bool, limit: float | None, job_id: str | None = None) -> None:
	def job():
		progress = {"files_total": len(files), "files_done": 0, "current_file": "", "lines": 0, "lines_total": 0, "percent": 0.0}
		CACHE[token].update(status='running', progress=progress, rows=[], dir=used_dir)
		if job_id:
			with JOBS_LOCK:
				rec = JOBS.get(job_id)
				if rec:
					rec['status'] = 'running'
		# Pre-count total lines
		total_lines = 0
		line_counts: Dict[str,int] = {}
		for fp in files:
			cnt = 0
			try:
				with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
					for _ in f:
						cnt += 1
			except Exception:
				cnt = 0
			line_counts[str(fp)] = cnt
			total_lines += cnt
		progress['lines_total'] = total_lines
		processed_lines = 0
		rows: List[Dict[str,str]] = []
		for idx, fp in enumerate(files, start=1):
			progress['current_file'] = str(fp)
			try:
				if process_file is not None:
					recs, _k, _m = process_file(fp, PREFIX_DEFAULT, IGNORE_VALUE_DEFAULT)  # type: ignore
					processed_lines += line_counts.get(str(fp),0)
				else:
					recs = []
					start_line = ''
					end_line = ''
					first_fdv_output = ''
					last_fdv_output = ''
					with open(fp, 'r', encoding='utf-8', errors='replace') as f:
						for i, line in enumerate(f, start=1):
							processed_lines += 1
							ls = line.lstrip()
							# Capture explicit Test Start/End Date lines even if no FDV OUTPUT yet
							ul = ls.upper()
							if ul.startswith('TEST START DATE') and not start_line:
								start_line = line.strip()
							if ul.startswith('TEST END DATE'):
								end_line = line.strip()
							if ls.startswith('FDV OUTPUT'):
								up = ls.upper()
								if 'MONITOR' in up or 'SHMOO' in up:
									continue
								entry = {'raw_line': line.rstrip('\n'), 'line_number': str(i), 'fdv_file': str(fp)}
								m_fid = re.search(r'(K\d{1,6}_[0-9]+_[0-9]+_[0-9\-]+)', line, re.I)
								if m_fid:
									entry['fuseid'] = m_fid.group(1)
								m_dut = re.search(r'\bDUT\s*[:=]?\s*(\d+)\b', line, re.I)
								if m_dut:
									entry['dut_id'] = m_dut.group(1)
								m_pr = re.search(r'\bPR\s*[:=]?\s*(\d+)\b', line, re.I)
								if m_pr:
									entry['pr'] = m_pr.group(1)
								m_vcc = re.search(r'\bVCC\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)', line, re.I)
								if m_vcc:
									entry['vcc'] = m_vcc.group(1)
								m_tm = re.search(r'\bTM\s*[:=]?\s*([0-9]+)\b', line, re.I)
								if m_tm:
									entry['tm'] = m_tm.group(1)
								m_temp = re.search(r'\bT(?:EMP)?\s*[:=]?\s*([-]?[0-9]+)\b', line, re.I)
								if m_temp:
									entry['temp'] = m_temp.group(1)
								m_rber = re.search(r'RBER\s*[:=]\s*([0-9.eE\-+]+)', line, re.I)
								if m_rber:
									entry['rber'] = m_rber.group(1)
								recs.append(entry)
								if not first_fdv_output:
									first_fdv_output = line.strip()
								last_fdv_output = line.strip()
							upline = line.upper()
							if not start_line and ('TEST START' in upline or 'BEGIN TEST' in upline):
								start_line = line.strip()
							if ('TEST END' in upline or 'END TEST' in upline or 'TEST COMPLETE' in upline):
								end_line = line.strip()
					if recs:
						if not start_line and first_fdv_output:
							start_line = first_fdv_output
						if not end_line and last_fdv_output:
							end_line = last_fdv_output
						# Attach start/end to first record and propagate to all for downstream grouping
						if start_line:
							recs[0].setdefault('test_start', start_line)
						if end_line:
							recs[0].setdefault('test_end', end_line)
						# Propagate to every record for consistency (some aggregators look at arbitrary row)
						if ('test_start' in recs[0]) or ('test_end' in recs[0]):
							_ts_val = recs[0].get('test_start','')
							_te_val = recs[0].get('test_end','')
							for _rr in recs:
								if _ts_val and 'test_start' not in _rr:
									_rr['test_start'] = _ts_val
								if _te_val and 'test_end' not in _rr:
									_rr['test_end'] = _te_val
					if recs and ('test_start' not in recs[0] or 'test_end' not in recs[0]):
						try:
							stat = fp.stat()
							ft = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(sep=' ')
							recs[0].setdefault('test_start', ft)
							recs[0].setdefault('test_end', ft)
						except Exception:
							pass
				for r in recs:
					r.setdefault('fdv_file', str(fp))
			except Exception:
				recs = []
			rows.extend(recs)
			progress['files_done'] = idx
			progress['lines'] = processed_lines
			try:
				if total_lines > 0:
					progress['percent'] = round((processed_lines / total_lines) * 100.0, 2)
			except Exception:
				pass
			# Incremental stats update so partial report can render progressively.
			try:
				if rows:
					# Compute quick stats on current accumulated rows (cheap subset aggregation)
					current_stats = stats_by_fdv_with_splits(rows, limit=(limit if not passfail_mode else 1e9), passfail_mode=passfail_mode)
					seen_i, order_i = set(), []
					for _s in current_stats:
						_f = _s.get('fdv_file','')
						if _f and _f not in seen_i:
							seen_i.add(_f); order_i.append(_f)
					CACHE[token].update(rows=rows[:], stats=current_stats, fdv_order=order_i)
			except Exception:
				pass
		for i, r in enumerate(rows):
			r.setdefault('_idx', i)
		stats = stats_by_fdv_with_splits(rows, limit=(limit if not passfail_mode else 1e9), passfail_mode=passfail_mode)
		seen, order = set(), []
		for s in stats:
			f = s.get('fdv_file', '')
			if f and f not in seen:
				seen.add(f); order.append(f)
		CACHE[token].update(status='done', rows=rows, stats=stats, fdv_order=order)
		progress['percent'] = 100.0
		if job_id:
			with JOBS_LOCK:
				rec = JOBS.get(job_id)
				if rec:
					rec['status'] = 'done'
					# Record end time for UI (jobs page needs ended_hms)
					rec['ended_at'] = time.time()
	threading.Thread(target=job, daemon=True).start()

@app.route('/', methods=['GET','POST'])
def home():
	if request.method == 'POST':
		dirpath = (request.form.get('dirpath') or '').strip()
		user_jobname = (request.form.get('jobname') or '').strip()
		limit_raw = (request.form.get('limit') or '').strip()
		limit_lower = limit_raw.lower()
		passfail_mode = (limit_lower == 'none')
		try:
			print(f"[home POST] limit_raw='{limit_raw}' passfail_mode={passfail_mode}")
		except Exception:
			pass
		if passfail_mode:
			limit_val: float | None = None
		else:
			try:
				limit_val = float(limit_raw)
			except Exception:
				limit_val = None
		try:
			if dirpath:
				root = Path(dirpath)
				if not root.exists() or not root.is_dir():
					flash('Directory not found.')
					return redirect(url_for('home'))
				files = _list_files(root)
				used_dir = str(root)
			else:
				uploads = request.files.getlist('dirfiles') or request.files.getlist('files')
				if not uploads:
					flash('Select files or specify directory.')
					return redirect(url_for('home'))
				tmp = Path(tempfile.mkdtemp(prefix='fdv_up_'))
				files = []
				for i, f in enumerate(uploads):
					name = f.filename or f'fdv_{i}.txt'
					dst = tmp / f'{i:05d}_{Path(name).name}'
					dst.write_bytes(f.read())
					files.append(dst)
				used_dir = str(tmp)
			if not files:
				flash('No files found.')
				return redirect(url_for('home'))
			token = uuid.uuid4().hex
			CACHE[token] = {'status':'queued','progress':{'files_total':len(files),'files_done':0},'dir':used_dir, 'limit': limit_val, 'passfail_mode': passfail_mode}
			# Create a pseudo job_id for UI parity with full app
			job_id = uuid.uuid4().hex[:10]
			# Derive simple job name from user override else folder/files
			if user_jobname:
				base_name = user_jobname.strip()
			else:
				if dirpath:
					base_name = Path(dirpath).name
				else:
					base_name = ','.join([p.name for p in files[:3]]) if files else 'job'
			if len(base_name) > 120:
				base_name = base_name[:117] + '...'
			with JOBS_LOCK:
				JOBS[job_id] = {'token': token, 'created_at': time.time(), 'status': 'queued', 'name': base_name}
			# Name derivation for upload path (used_dir variable) ensure consistent naming
			# Second pass naming (keep user provided if any)
			try:
				if not user_jobname:
					base_name2 = Path(used_dir).name if used_dir else ''
					if not base_name2:
						base_name2 = ','.join([p.name for p in files[:3]]) if files else 'job'
					if len(base_name2) > 120:
						base_name2 = base_name2[:117] + '...'
				else:
					base_name2 = base_name
			except Exception:
				base_name2 = base_name
			with JOBS_LOCK:
				JOBS[job_id]['name'] = base_name2
			_start_job(token, files, used_dir, passfail_mode=passfail_mode, limit=limit_val, job_id=job_id)
			return render_template('fdv2_progress.html', token=token, job_id=job_id)
		except Exception as e:
			flash(f'Failed: {e}')
			return redirect(url_for('home'))
	tok = (request.args.get('token') or '').strip()
	limit_param_present = 'limit' in request.args  # distinguish missing vs blank
	limit_q = (request.args.get('limit') or '').strip()
	limit_q_lower = limit_q.lower()
	passfail_mode = (limit_q_lower == 'none')
	try:
		print(f"[home GET] token={tok} limit_param_present={limit_param_present} limit_q='{limit_q}' passfail_mode={passfail_mode}")
	except Exception:
		pass
	if passfail_mode:
		limit_val = None
	else:
		try:
			limit_val = float(limit_q) if limit_q else None
		except Exception:
			limit_val = None

	if tok and tok in CACHE:
		d = CACHE[tok]
		# attempt to find job id for this token to show job links like full app
		job_id_for_token = None
		try:
			with JOBS_LOCK:
				for _jid, _rec in JOBS.items():
					if _rec.get('token') == tok:
						job_id_for_token = _jid
						break
		except Exception:
			pass
		# If no limit param supplied in query, keep stored values (user didn't request change)
		if not limit_param_present:
			stored_passfail = d.get('passfail_mode')
			stored_limit = d.get('limit')
			rows = d.get('rows', [])
			stats = d.get('stats', [])
			# Load dispositions/comments on demand
			disp = d.get('dispositions'); comm = d.get('comments')
			if disp is None: disp = _load_dispositions(tok); d['dispositions'] = disp
			if comm is None: comm = _load_comments(tok); d['comments'] = comm
			return render_template('fdv2_report.html', token=tok, job_id=job_id_for_token, stats=stats, used_dir=d.get('dir'), fdv_order=d.get('fdv_order', []), limit=(None if stored_passfail else stored_limit), dispositions=disp, comments=comm)
		# limit param present: consider recompute
		rows = d.get('rows', [])
		stats = d.get('stats', [])
		if rows and ((d.get('passfail_mode') != passfail_mode) or (d.get('limit') != limit_val)):
			stats = stats_by_fdv_with_splits(rows, limit=(limit_val if not passfail_mode else 1e9), passfail_mode=passfail_mode)
			d.update(stats=stats, passfail_mode=passfail_mode, limit=limit_val)
		# Load dispositions/comments for updated view
		disp = d.get('dispositions'); comm = d.get('comments')
		if disp is None: disp = _load_dispositions(tok); d['dispositions'] = disp
		if comm is None: comm = _load_comments(tok); d['comments'] = comm
		return render_template('fdv2_report.html', token=tok, job_id=job_id_for_token, stats=stats, used_dir=d.get('dir'), fdv_order=d.get('fdv_order', []), limit=(None if passfail_mode else limit_val), dispositions=disp, comments=comm)
	# No token provided or token not found: render empty report shell
	return render_template('fdv2_report.html', token='', job_id=None, stats=[], used_dir=None, fdv_order=[], limit=(None if passfail_mode else limit_val), dispositions={}, comments={})

@app.route('/api/dispositions/<token>', methods=['POST'])
def api_dispositions_update(token: str):
    d = CACHE.get(token)
    if d is None:
        # create minimal container so UI can still work after restart
        d = CACHE.setdefault(token, {'status':'dispositions-only'})
    disp = d.setdefault('dispositions', _load_dispositions(token))
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}
    key = (payload.get('key') or '').strip(); val = (payload.get('value') or '').strip()
    if not key:
        return Response(json.dumps({'error':'missing key'}), mimetype='application/json', status=400)
    if val: disp[key] = val
    else: disp.pop(key, None)
    try: _save_dispositions(token, disp)
    except Exception: pass
    return Response(json.dumps({'ok':True,'key':key,'value':disp.get(key,'')}), mimetype='application/json')

@app.route('/api/dispositions/<token>', methods=['GET'])
def api_dispositions_get(token: str):
    d = CACHE.get(token)
    disp = (d.get('dispositions') if d else None) or _load_dispositions(token)
    if d is not None:
        d['dispositions'] = disp
    return Response(json.dumps({'token':token,'dispositions':disp}), mimetype='application/json')

@app.route('/job/<job_id>/status')
def job_status(job_id: str):
	token = _resolve_job_token(job_id)
	if not token:
		return Response(json.dumps({'status':'missing'}), mimetype='application/json', status=404)
	d = CACHE.get(token) or {}
	out = {
		'status': d.get('status','unknown'),
		'progress': d.get('progress', {}),
		'error': d.get('error')
	}
	return Response(json.dumps(out), mimetype='application/json')

@app.route('/job/<job_id>/progress')
def job_progress(job_id: str):
	token = _resolve_job_token(job_id)
	if not token:
		return redirect(url_for('home'))
	# Reuse progress template similar to full app
	return render_template('fdv2_progress.html', token=token, job_id=job_id, limit_raw='')

@app.route('/api/partial/<token>')
def api_partial(token: str):
	d = CACHE.get(token)
	if not d:
		return Response(json.dumps({'error':'unknown token'}), mimetype='application/json', status=404)
	stats = d.get('stats', []) or []
	dispositions = d.get('dispositions', {}) or {}
	comments = d.get('comments', {}) or {}
	progress = d.get('progress', {})
	return Response(json.dumps({'token':token,'stats':stats,'dispositions':dispositions,'comments':comments,'progress':progress,'status':d.get('status')}), mimetype='application/json')

@app.route('/job/<job_id>/report')
def job_report(job_id: str):
	token = _resolve_job_token(job_id)
	if not token:
		return redirect(url_for('home'))
	lr = request.args.get('limit')
	if lr:
		return redirect(url_for('home', token=token, limit=lr))
	return redirect(url_for('home', token=token))

@app.route('/status/<token>')
def status(token: str):
	d = CACHE.get(token)
	if not d:
		return Response(json.dumps({'status':'missing'}), mimetype='application/json')
	return Response(json.dumps({'status': d.get('status'), 'progress': d.get('progress', {}), 'error': d.get('error')}), mimetype='application/json')

@app.route('/stream/job/<job_id>')
def stream_job(job_id: str):
	# Resolve job id to token
	with JOBS_LOCK:
		rec = JOBS.get(job_id)
	if not rec:
		return Response('not found', status=404)
	token = rec.get('token')
	if not token:
		return Response('not found', status=404)
	def gen():
		last_sent = 0.0
		while True:
			data = CACHE.get(token) or {}
			prog = data.get('progress', {}) or {}
			st = data.get('status','unknown')
			now = time.time()
			if now - last_sent >= 1.0:
				last_sent = now
				payload = {
					'status': st,
					'progress': {
						'percent': prog.get('percent'),
						'lines': prog.get('lines'),
						'lines_total': prog.get('lines_total'),
						'files_done': prog.get('files_done'),
						'files_total': prog.get('files_total'),
						'current_file': prog.get('current_file'),
					}
				}
				yield f"data: {json.dumps(payload)}\n\n"
			if st in ('done','error'):
				break
			time.sleep(0.5)
	return Response(gen(), mimetype='text/event-stream')

@app.route('/fdv/<token>/tests', methods=['GET','POST'])
def tests(token: str):
	d = CACHE.get(token)
	if not d or 'rows' not in d:
		flash('Session expired.')
		return redirect(url_for('home'))
	if request.method == 'POST':
		fdvs = [s.strip() for s in request.form.getlist('fdv') if s.strip()]
	else:
		fdvs = [s.strip() for s in request.args.getlist('fdv') if s.strip()]
	if not fdvs:
		one = (request.values.get('fdv') or '').strip()
		if one:
			fdvs = [one]
	if not fdvs:
		flash('Missing fdv selection.')
		return redirect(url_for('home', token=token))
	limit_raw = (request.values.get('limit') or '').strip()
	passfail_mode = (limit_raw.lower() == 'none')
	if passfail_mode:
		limit = 1e9  # large placeholder; tokens decide PASS/FAIL
	else:
		try:
			limit = float(limit_raw)
		except Exception:
			limit = 0.0  # fallback minimal
	rows = d.get('rows', [])
	stats = stats_by_testname_selected(rows, fdvs, limit=limit, passfail_mode=passfail_mode)
	sel_rows = []
	for s in fdvs:
		fdv, pr, vcc, tm, temp = _parse_fdv_selector(s)
		sel_rows.append({'fdv': fdv, 'pr': pr, 'vcc': vcc, 'tm': tm, 'temp': temp})
	return render_template('fdv2_report_tests.html', token=token, fdvs=fdvs, stats=stats, sel_rows=sel_rows, limit=limit)

@app.route('/fdv/<token>/tests/sample')
def tests_sample(token: str):
	d = CACHE.get(token)
	if not d or 'rows' not in d:
		return Response('session expired', status=400)
	rows: List[Dict[str,str]] = d.get('rows', [])
	sels = [s.strip() for s in request.args.getlist('fdv') if s.strip()]
	if not sels:
		one = (request.args.get('fdv') or '').strip()
		if one:
			sels = [one]
	if not sels:
		return Response('missing fdv', status=400)
	parsed = [_parse_fdv_selector(x) for x in sels]
	recs: List[Dict] = []
	for idx, r in enumerate(rows):
		rk = _get_split_tuple(r)
		if not any((sf and sf == rk[0] or not sf) and (spr and spr == rk[1] or not spr) and (svcc and svcc == rk[2] or not svcc) and (stm and stm == rk[3] or not stm) and (stemp and stemp == rk[4] or not stemp) for (sf,spr,svcc,stm,stemp) in parsed):
			continue
		if (r.get('tname','') or '').strip().upper() == 'PR':
			continue
		tn = (r.get('testname','') or '').strip() or derive_testname((r.get('tname','') or '').strip())
		wl = _extract_wl_or_page(r)
		rv = _get_rber(r)
		if rv is None:
			continue
		if rv <= 0:
			rv = 1e-12
		recs.append({'testname': tn, 'DUT': f"DUT{(r.get('dut_id','') or '').strip() or '?'}", 'plane': r.get('plane',''), 'plane_addr': r.get('plane_addr',''), 'blk': r.get('blk'), 'WL': wl, 'RBER': rv, 'pagetype': r.get('pagetype',''), 'line_number': r.get('line_number',''), '_idx': r.get('_idx', idx)})
	try:
		lim = int((request.args.get('limit') or '10').strip())
	except Exception:
		lim = 10
	head = recs[:max(0, lim)]
	def fmt(r: Dict) -> str:
		wl_txt = '-' if r.get('WL') is None or float(r.get('WL') or -1.0) < 0 else str(int(float(r.get('WL'))))
		blk_txt = '' if r.get('blk') is None else str(r.get('blk'))
		return f"<tr><td>{r.get('testname','')}</td><td>{r.get('DUT','')}</td><td>{r.get('plane','')}</td><td>{r.get('plane_addr','')}</td><td>{blk_txt}</td><td>{wl_txt}</td><td>{float(r.get('RBER',0.0)):.3e}</td><td>{r.get('pagetype','')}</td><td>{r.get('line_number','')}</td></tr>"
	rows_html = ''.join(fmt(r) for r in head)
	html = ("<!doctype html><html><head><meta charset='utf-8'><title>FDV sample</title>" "<style>body{font-family:Arial;margin:16px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:4px 6px;font-size:13px;}th{background:#f7f7f7;}</style></head><body>" + "<table><thead><tr><th>testname</th><th>DUT</th><th>plane</th><th>plane_addr</th><th>blk</th><th>WL</th><th>RBER</th><th>pagetype</th><th>line #</th></tr></thead><tbody>" + rows_html + "</tbody></table></body></html>")
	return Response(html, mimetype='text/html')

@app.route('/fdv/<token>/rawline')
def rawline(token: str):
	d = CACHE.get(token)
	if not d or 'rows' not in d:
		return Response('session expired', status=400)
	rows: List[Dict[str,str]] = d.get('rows', [])
	try:
		idx = int(request.args.get('idx','-1'))
	except Exception:
		idx = -1
	if idx < 0 or idx >= len(rows):
		return Response('bad idx', status=400)
	return Response('<pre>' + (rows[idx].get('raw_line','') or '') + '</pre>', mimetype='text/html')

@app.route('/fdv/<token>/fails')
def fails(token: str):
	d = CACHE.get(token)
	if not d or 'rows' not in d:
		return Response('session expired', status=400)
	rows: List[Dict[str,str]] = d.get('rows', [])
	fdvsel = (request.args.get('fdv') or '').strip()
	if not fdvsel:
		return Response('missing fdv selection', status=400)
	limit = request.args.get('limit')
	passfail_mode = (limit in (None, ''))
	try:
		limit_f = float(limit) if not passfail_mode else None
	except Exception:
		limit_f = None
	fdv, pr, vcc, tm, temp = _parse_fdv_selector(fdvsel)
	failing = []
	for r in rows:
		key = _get_split_tuple(r)
		if (fdv and key[0] != fdv) or (pr and key[1] != pr) or (vcc and key[2] != vcc) or (tm and key[3] != tm) or (temp and key[4] != temp):
			continue
		if (r.get('tname','') or '').strip().upper() == 'PR':
			continue
		rv = _get_rber(r)
		if rv is None:
			continue
		if (limit_f is not None and rv >= limit_f) or (passfail_mode and 'FAIL' in (r.get('raw_line','').upper())):
			failing.append(r)
	rows_html = []
	def esc(s: str) -> str:
		return (s or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
	for r in failing[:5000]:
		idx = r.get('_idx') or ''
		rows_html.append('<tr>'
			f"<td>{esc(r.get('testname') or derive_testname((r.get('tname','') or '').strip()) or '')}</td>"
			f"<td>{esc(r.get('dut_id','') or '')}</td>"
			f"<td>{esc(r.get('fuseid','') or '')}</td>"
			f"<td>{_get_rber(r) if _get_rber(r) is not None else ''}</td>"
			f"<td>{esc(r.get('line_number',''))}</td>"
			f"<td><a href='/fdv/{token}/rawline?idx={idx}' target='_blank'>raw</a></td>" '</tr>')
	html = (
		"<!doctype html><html><head><meta charset='utf-8'><title>Fail Rows</title>"
		"<style>body{font-family:Arial;margin:16px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:4px 6px;font-size:13px;}th{background:#f7f7f7;} .num{font-family:Consolas,monospace;}</style></head><body>"
		f"<h3>Fail rows for {esc(fdvsel)}</h3><div>Total failing rows: {len(failing)}</div>"
		+ ("<div>Mode: PASS/FAIL from logs.</div>" if passfail_mode else (f"<div>Threshold (limit) = {limit_f:.6g}</div>" if limit_f is not None else ""))
		+ "<table><thead><tr><th>testname</th><th>DUT</th><th>FUSEID</th><th>RBER</th><th>line #</th><th>raw</th></tr></thead><tbody>"
		+ ''.join(rows_html) + "</tbody></table></body></html>"
	)
	return Response(html, mimetype='text/html')

@app.route('/api/jobs')
def api_jobs():
	jobs_out = []
	with JOBS_LOCK:
		for jid, rec in JOBS.items():
			tok = rec.get('token')
			data = CACHE.get(tok) or {}
			prog = data.get('progress', {}) or {}
			created_at = rec.get('created_at') if isinstance(rec.get('created_at'), (int, float)) else None
			ended_at = rec.get('ended_at') if isinstance(rec.get('ended_at'), (int, float)) else None
			def _fmt(ts):
				try:
					return time.strftime('%H:%M:%S', time.localtime(ts)) if ts else None
				except Exception:
					return None
			status = data.get('status', 'unknown')
			jobs_out.append({
				'job_id': jid,
				'token': tok,
				'status': status,
				'percent': prog.get('percent'),
				'files_done': prog.get('files_done'),
				'files_total': prog.get('files_total'),
				'lines': prog.get('lines'),
				'created_at': created_at,
				'created_hms': _fmt(created_at) if created_at else None,
				'ended_at': ended_at,
				'ended_hms': _fmt(ended_at) if ended_at else None,
				'duration_secs': (round((ended_at - created_at), 2) if (created_at and ended_at) else None),
				'report_url': f"/job/{jid}/report",
				'progress_url': f"/job/{jid}/progress",
				'status_url': f"/job/{jid}/status",
				'report_ready': status == 'done' and bool(data.get('rows')),
				'name': rec.get('name', '')
			})
	return jsonify({'jobs': jobs_out, 'count': len(jobs_out)})

@app.route('/api/job/<job_id>/name', methods=['POST'])
def api_job_set_name(job_id: str):
	try:
		payload = request.get_json(force=True, silent=True) or {}
	except Exception:
		payload = {}
	name = (payload.get('name') or '').strip()
	if len(name) > 120:
		return jsonify({'ok': False, 'error': 'name too long'}), 400
	with JOBS_LOCK:
		if job_id not in JOBS:
			return jsonify({'ok': False, 'error': 'unknown job id'}), 404
		JOBS[job_id]['name'] = name
	return jsonify({'ok': True, 'name': name})

@app.route('/jobs')
def jobs_page():
	with JOBS_LOCK:
		ids = list(JOBS.keys())
	return render_template('fdv2_jobs.html', job_ids=ids)

# Compatibility endpoints for legacy template names
app.add_url_rule('/', 'report_home', home, methods=['GET','POST'])
app.add_url_rule('/status/<token>', 'report_status', status, methods=['GET'])
app.add_url_rule('/fdv/<token>/tests', 'report_tests', tests, methods=['GET','POST'])
app.add_url_rule('/fdv/<token>/tests/sample', 'report_tests_sample', tests_sample, methods=['GET'])
app.add_url_rule('/fdv/<token>/rawline', 'report_rawline', rawline, methods=['GET'])

@app.route('/status/<token>/fdvtable')
def report_status_fdvtable(token: str):
	d = CACHE.get(token)
	if not d:
		return '<div class="small">No session.</div>'
	# Support dynamic limit override via query (?limit=... or limit=none)
	limit_q = (request.args.get('limit') or '').strip()
	limit_lower = limit_q.lower()
	passfail_mode_req = (limit_lower in ('', 'none', 'default'))
	rows = d.get('rows', []) or []
	# Determine if we must recompute stats for this view
	stats_cached = d.get('stats') or []
	need_recompute = False
	if passfail_mode_req:
		# Want PASS/FAIL from logs; recompute only if cached not already passfail_mode
		if not d.get('passfail_mode'):
			need_recompute = True
	else:
		# Threshold mode requested. Parse limit and compare with cached.
		try:
			limit_val = float(limit_q)
		except Exception:
			limit_val = d.get('limit')  # fallback
		if d.get('passfail_mode') or (limit_val is not None and limit_val != d.get('limit')):
			need_recompute = True
	if need_recompute and rows:
		try:
			if passfail_mode_req:
				stats_cached = stats_by_fdv_with_splits(rows, limit=1e9, passfail_mode=True)
				# Do not overwrite stored non-passfail stats permanently unless job originally passfail
			else:
				stats_cached = stats_by_fdv_with_splits(rows, limit=limit_val if limit_q else d.get('limit'), passfail_mode=False)
			# Update cache so subsequent calls faster
			if passfail_mode_req:
				# Preserve original limit; just mark transient view
				pass
			else:
				d.update(limit=limit_val, passfail_mode=False, stats=stats_cached)
		except Exception:
			pass
	stats = stats_cached
	if not stats:
		return '<div class="small">Collecting…</div>'
	# Build full column set mirroring final report (condensed styling)
	headers = [
		'FDV Test','PR','VCC','TM','Temp','Pagemap','Unit Count','Count','PASS','FAIL','% FAIL','Vector','Min','Max','Mean','Stdev','Median','Comment','testtime','Start','End','Unit Info'
	]
	rows_fragments = []
	for r in stats[:400]:  # cap interim rows for performance
		count_raw = r.get('count')
		try:
			count = int(count_raw) if count_raw not in (None, '') else 0
		except Exception:
			count = 0
		fail_field = r.get('fail_n', r.get('fail',''))
		try:
			failn = int(fail_field) if fail_field not in (None,'') else 0
		except Exception:
			failn = 0
		pct = (100.0 * failn / count) if count > 0 else 0.0
		key = f"{r.get('fdv_file','')}|{r.get('pr','')}|{r.get('vcc','')}|{r.get('tm','')}|{r.get('temp','')}"
		comment_val = (d.get('dispositions', {}) or {}).get(key,'')
		cls = ''
		if count>0:
			if pct==0: cls='pct-ok'
			elif pct<=25: cls='pct-warn'
			else: cls='pct-bad'
		rows_fragments.append(
			'<tr>'
			f"<td>{r.get('fdv_file','')}</td>"
			f"<td>{r.get('pr','')}</td>"
			f"<td>{r.get('vcc','')}</td>"
			f"<td>{r.get('tm','')}</td>"
			f"<td>{r.get('temp','')}</td>"
			f"<td>{r.get('pagemap', r.get('product_type',''))}</td>"
			f"<td>{r.get('valid_fuseid_count','')}</td>"
			f"<td>{count}</td>"
			f"<td>{r.get('pass_n', r.get('pass',''))}</td>"
			f"<td>{failn}</td>"
			f"<td class='{cls}'>{pct:.1f}%</td>"
			f"<td>{r.get('vector','')}</td>"
			f"<td>{r.get('min','')}</td>"
			f"<td>{r.get('max','')}</td>"
			f"<td>{r.get('mean','')}</td>"
			f"<td>{r.get('stdev','')}</td>"
			f"<td>{r.get('median','')}</td>"
			f"<td style='max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{comment_val}</td>"
			f"<td>{r.get('testtime_label','')}</td>"
			f"<td>{r.get('test_start','')}</td>"
			f"<td>{r.get('test_end','')}</td>"
			f"<td style='max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{r.get('unit_info','')}</td>"
			'</tr>'
		)
	# Inline minimal styles for % fail classes
	style = "<style>.pct-ok{background:#e8f9e8;color:#167a16;font-weight:600}.pct-warn{background:#ffe6e6;color:#a11}.pct-bad{background:#ff4d4d;color:#fff;font-weight:700} table.partial{border-collapse:collapse;font-size:11px} table.partial th,table.partial td{border:1px solid #ccc;padding:2px 4px;}</style>"
	head_html = '<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>'
	return style + '<table class="partial">' + '<thead>' + head_html + '</thead><tbody>' + ''.join(rows_fragments) + '</tbody></table>'

if __name__ == '__main__':
	debug = os.environ.get('FDV_REPORT2_DEBUG','1').lower() not in {'0','false','no','off'}
	host = os.environ.get('FDV_REPORT2_HOST','0.0.0.0').strip() or '0.0.0.0'
	try:
		port = int(os.environ.get('FDV_REPORT2_PORT','5057'))
	except Exception:
		port = 5057
	app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
