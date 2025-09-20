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

# ---------------- Job Persistence (registry + snapshots) ----------------
_REGISTRY_FILE = _PERSIST_BASE / 'jobs_registry.json'
_JOB_REGISTRY: Dict[str, Dict] = {}

def _load_job_registry() -> Dict[str, Dict]:
	try:
		if _REGISTRY_FILE.exists():
			with open(_REGISTRY_FILE, 'r', encoding='utf-8') as f:
				data = json.load(f) or {}
			if isinstance(data, dict):
				return data.get('jobs', data) if isinstance(data.get('jobs'), dict) else data
	except Exception:
		pass
	return {}

def _save_job_registry():
	try:
		_PERSIST_BASE.mkdir(parents=True, exist_ok=True)
		tmp = _REGISTRY_FILE.with_suffix('.tmp')
		with open(tmp, 'w', encoding='utf-8') as f:
			json.dump({'jobs': _JOB_REGISTRY}, f, indent=2, sort_keys=True)
		os.replace(tmp, _REGISTRY_FILE)
	except Exception:
		pass

def _snapshot_path(token: str) -> Path:
	return _persist_dir(token) / 'snapshot.json'

def _save_snapshot(token: str, cache_entry: Dict):
	try:
		snap = {
			'status': cache_entry.get('status'),
			'rows': cache_entry.get('rows') or [],
			'stats': cache_entry.get('stats') or [],
			'fdv_order': cache_entry.get('fdv_order') or [],
			'dir': cache_entry.get('dir'),
			'limit': cache_entry.get('limit'),
			'passfail_mode': cache_entry.get('passfail_mode'),
			'dispositions': cache_entry.get('dispositions') or {},
			'comments': cache_entry.get('comments') or {},
		}
		p = _snapshot_path(token)
		tmp = p.with_suffix('.tmp')
		with open(tmp, 'w', encoding='utf-8') as f:
			json.dump(snap, f)
		os.replace(tmp, p)
	except Exception:
		pass

def _load_snapshot(token: str) -> Dict:
	try:
		p = _snapshot_path(token)
		if p.exists():
			with open(p, 'r', encoding='utf-8') as f:
				return json.load(f) or {}
	except Exception:
		pass
	return {}

def _rehydrate_jobs():
	global _JOB_REGISTRY
	try:
		_JOB_REGISTRY = _load_job_registry()
	except Exception:
		_JOB_REGISTRY = {}
	for jid, meta in list(_JOB_REGISTRY.items()):
		status = meta.get('status')
		if status in {'deleted'}:
			continue
		token = meta.get('token')
		if not token:
			continue
		snap = _load_snapshot(token) if status == 'done' else {}
		cache_entry = {
			'status': snap.get('status', status or 'done'),
			'rows': snap.get('rows', []),
			'stats': snap.get('stats', []),
			'fdv_order': snap.get('fdv_order', []),
			'dir': snap.get('dir'),
			'limit': snap.get('limit'),
			'passfail_mode': snap.get('passfail_mode'),
			'dispositions': snap.get('dispositions', {}),
			'comments': snap.get('comments', {}),
			'progress': {'percent': 100.0} if status == 'done' else {'percent': 0.0},
		}
		CACHE.setdefault(token, cache_entry)
		with JOBS_LOCK:
			JOBS.setdefault(jid, {
				'token': token,
				'created_at': meta.get('created_at'),
				'ended_at': meta.get('ended_at'),
				'status': cache_entry.get('status','done'),
				'name': meta.get('name',''),
			})

try:
	_rehydrate_jobs()
except Exception:
	pass

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

def _interpret_limit_mode(raw: str | None) -> tuple[bool, float | None]:
	"""Return (passfail_mode, numeric_limit).
	passfail_mode True means use PASS/FAIL tokens from log, ignoring numeric threshold.
	Accept synonyms: '', 'none', 'default', 'passfail'.
	If numeric parse fails, fallback to passfail mode.
	"""
	if raw is None:
		return True, None
	s = raw.strip().lower()
	if s in ('', 'none', 'default', 'passfail', 'pf'):
		return True, None
	try:
		return False, float(raw)  # numeric threshold mode
	except Exception:
		return True, None

def _start_job(token: str, files: List[Path], used_dir: str, *, passfail_mode: bool, limit: float | None, job_id: str | None = None, prodmode: bool = False, ledger_map: Dict[str, Path] | None = None) -> None:
	def job():
		progress = {"files_total": len(files), "files_done": 0, "current_file": "", "lines": 0, "lines_total": 0, "percent": 0.0}
		CACHE[token].update(status='running', progress=progress, rows=[], dir=used_dir)
		if job_id:
			with JOBS_LOCK:
				rec = JOBS.get(job_id)
				if rec:
					rec['status'] = 'running'
		# If prodmode, keep only files that have a corresponding ledger .ready entry
		if prodmode and ledger_map:
			try:
				files = [fp for fp in files if str(fp) in {str(k) for k in ledger_map.keys()}]
			except Exception:
				pass
			try:
				progress['files_total'] = len(files)
				CACHE[token]['progress'] = progress
			except Exception:
				pass
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
		# For overall ETA smoothing (store recent value)
		eta_last = None  # type: ignore
		job_start_time = time.time()
		# Files list is final at this point; iterate and process
		for idx, fp in enumerate(files, start=1):
			# Support pause: spin while _pause_flag set (allow cooperative delete)
			while CACHE.get(token, {}).get('_pause_flag'):
				CACHE[token]['status'] = 'paused'
				if CACHE.get(token, {}).get('_stop_flag'):
					CACHE[token]['status'] = 'deleted'
					return
				time.sleep(0.25)
			# Ensure status returns to running after pause released (if not terminal)
			if CACHE.get(token, {}).get('status') == 'paused' and not CACHE.get(token, {}).get('_pause_flag'):
				CACHE[token]['status'] = 'running'
			if CACHE.get(token, {}).get('_stop_flag'):
				CACHE[token]['status'] = 'deleted'
				return
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
			# In prodmode, rename corresponding ledger .ready to .done after processing this file
			try:
				if prodmode and ledger_map:
					key = str(fp)
					ready_path = ledger_map.get(key)
					if ready_path and ready_path.exists():
						done_path = ready_path.with_suffix('.done')
						try:
							os.replace(str(ready_path), str(done_path))
						except Exception:
							try:
								ready_path.rename(done_path)
							except Exception:
								pass
			except Exception:
				pass
			# Update overall ETA each iteration (based on lines processed vs total)
			try:
				if total_lines > 0 and processed_lines > 0 and processed_lines <= total_lines:
					elapsed = time.time() - job_start_time
					frac = processed_lines / float(total_lines)
					if 0 < frac < 1.0 and elapsed > 0:
						rem = elapsed * (1.0 - frac) / frac
						if eta_last is not None:
							rem = 0.5 * rem + 0.5 * eta_last
						eta_last = rem
						# Store both top-level and inside progress for client consumption
						eta_int = int(rem)
						if eta_int == 0 and rem > 0:
							eta_int = 1  # avoid misleading 0s when fractional second remains
						CACHE[token]['eta_overall_secs'] = eta_int
						try:
							CACHE[token]['progress']['eta_overall_secs'] = eta_int
						except Exception:
							pass
			except Exception:
				pass
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
					# Persist registry + snapshot
					meta = _JOB_REGISTRY.get(job_id)
					if meta is not None:
						meta['status'] = 'done'
						meta['ended_at'] = rec.get('ended_at')
						_save_job_registry()
					try:
						_save_snapshot(token, CACHE.get(token, {}))
					except Exception:
						pass
	threading.Thread(target=job, daemon=True).start()

@app.route('/', methods=['GET','POST'])
def home():
	if request.method == 'POST':
		dirpath = (request.form.get('dirpath') or '').strip()
		prod_raw = (request.form.get('prodmode') or '').strip().lower()
		prodmode = prod_raw in ('1','true','on','yes')
		user_jobname = (request.form.get('jobname') or '').strip()
		limit_raw = (request.form.get('limit') or '').strip()
		passfail_mode, limit_val = _interpret_limit_mode(limit_raw)
		try:
			print(f"[home POST] limit_raw='{limit_raw}' passfail_mode={passfail_mode} limit_val={limit_val}")
		except Exception:
			pass
		# limit_val already determined by helper
		try:
			if dirpath:
				root = Path(dirpath)
				if not root.exists() or not root.is_dir():
					flash('Directory not found.')
					return redirect(url_for('home'))
				ledger_map: Dict[str, Path] = {}
				if prodmode:
					# Require ledger directory with .ready files; convert lingering .done -> .ready first
					led_dir = root / 'ledger'
					if led_dir.is_dir():
						for p in led_dir.iterdir():
							try:
								if p.is_file() and p.suffix.lower() == '.done':
									tgt = p.with_suffix('.ready')
									if not tgt.exists():
										try:
											os.replace(str(p), str(tgt))
										except Exception:
											try:
												p.rename(tgt)
											except Exception:
												pass
							except Exception:
								continue
					# Gather .ready list and map by stem to .txt under root (excluding ledger)
					ready_files = [p for p in led_dir.iterdir() if p.is_file() and p.suffix.lower() == '.ready'] if led_dir.is_dir() else []
					all_files = _list_files(root)
					stem_index: Dict[str, List[Path]] = {}
					for f in all_files:
						try:
							if 'ledger' in f.parts and (root / 'ledger') in f.parents:
								continue
						except Exception:
							pass
						stem_index.setdefault(f.stem, []).append(f)
					selected: List[Path] = []
					for rf in ready_files:
						base = rf.stem
						candidates = [c for c in stem_index.get(base, []) if c.suffix.lower() == '.txt']
						chosen = candidates[0] if candidates else None
						if chosen is not None and chosen.is_file():
							selected.append(chosen)
							ledger_map[str(chosen)] = rf
					if not selected:
						flash('No matching ledger .ready files found under directory/ledger.')
						return redirect(url_for('home'))
					files = selected
				else:
					files = _list_files(root)
				used_dir = str(root)
			else:
				if prodmode:
					flash('Production mode requires a directory with a ledger folder (uploads are not allowed).')
					return redirect(url_for('home'))
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
				# Persist registry entry
				_JOB_REGISTRY[job_id] = {
					'job_id': job_id,
					'token': token,
					'name': base_name2,
					'status': 'queued',
					'created_at': JOBS[job_id]['created_at'],
					'ended_at': None,
					'limit': limit_val,
					'passfail_mode': passfail_mode,
				}
				_save_job_registry()
			_start_job(token, files, used_dir, passfail_mode=passfail_mode, limit=limit_val, job_id=job_id, prodmode=prodmode, ledger_map=ledger_map if dirpath and prodmode else None)
			return render_template('fdv2_progress.html', token=token, job_id=job_id)
		except Exception as e:
			flash(f'Failed: {e}')
			return redirect(url_for('home'))
	tok = (request.args.get('token') or '').strip()
	limit_param_present = 'limit' in request.args  # distinguish missing vs blank
	limit_q = (request.args.get('limit') or '').strip()
	passfail_mode, limit_val = _interpret_limit_mode(limit_q)
	try:
		print(f"[home GET] token={tok} limit_param_present={limit_param_present} limit_q='{limit_q}' passfail_mode={passfail_mode} limit_val={limit_val}")
	except Exception:
		pass

	if tok and tok in CACHE:
		d = CACHE[tok]
		# attempt to find job id for this token to show job links like full app
		job_id_for_token = None
		with JOBS_LOCK:
			for _jid, _rec in JOBS.items():
				if _rec.get('token') == tok:
					data = CACHE.get(tok) or {}
					if data.get('status') == 'deleted':
						continue
					job_id_for_token = _jid
					break
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
			if passfail_mode:
				# In pass/fail mode we store limit=None to avoid future confusion
				d.update(stats=stats, passfail_mode=True, limit=None)
			else:
				d.update(stats=stats, passfail_mode=False, limit=limit_val)
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
	prog = d.get('progress', {}) or {}
	out = {
		'status': d.get('status','unknown'),
		'progress': prog,
		'eta_overall_secs': (prog.get('eta_overall_secs') if isinstance(prog.get('eta_overall_secs'), (int,float)) else d.get('eta_overall_secs')),
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

@app.route('/api/stats/<token>')
def api_stats(token: str):
	"""Return compact stats focused on vector/count/pass/fail for verification.
	Includes fdv split fields plus pass/fail mode & limit stored.
	"""
	d = CACHE.get(token)
	if not d:
		return Response(json.dumps({'error': 'unknown token'}), mimetype='application/json', status=404)
	rows = d.get('rows', []) or []
	stats_cached = d.get('stats', []) or []
	passfail_mode = d.get('passfail_mode')
	limit = d.get('limit')
	# Build minimal projection
	out_list = []
	for s in stats_cached:
		out_list.append({
			'fdv_file': s.get('fdv_file',''),
			'pr': s.get('pr',''),
			'vcc': s.get('vcc',''),
			'tm': s.get('tm',''),
			'temp': s.get('temp',''),
			'vector': s.get('vector',''),
			'count': s.get('count'),
			'pass': s.get('pass_n', s.get('pass')),
			'fail': s.get('fail_n', s.get('fail')),
			'valid_fuseid_count': s.get('valid_fuseid_count'),
		})
	payload = {
		'token': token,
		'status': d.get('status'),
		'passfail_mode': passfail_mode,
		'limit': limit,
		'vector_stats': out_list,
		'row_count': len(rows)
	}
	return Response(json.dumps(payload), mimetype='application/json')

@app.route('/job/<job_id>/report')
def job_report(job_id: str):
	# Render single job report (reuse main report template if available)
	token = _resolve_job_token(job_id)
	if not token:
		return Response('unknown job', status=404)
	return redirect(url_for('home', token=token))

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
	html = ("<!doctype html><html><head><meta charset='utf-8'><title>FDV sample</title>"
	"<style>body{font-family:Arial;margin:16px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:4px 6px;font-size:13px;}th{background:#f7f7f7;}</style></head><body>"
	+ "<table><thead><tr><th>testname</th><th>DUT</th><th>plane</th><th>plane_addr</th><th>blk</th><th>WL</th><th>RBER</th><th>pagetype</th><th>line #</th></tr></thead><tbody>"
	+ rows_html + "</tbody></table></body></html>")
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
	# Determine mode: blank or 'default' => pass/fail-from-logs
	raw_limit = request.args.get('limit')
	raw_limit = None if raw_limit is None else raw_limit.strip()
	passfail_mode = (raw_limit in (None, '', 'default', 'none'))
	if passfail_mode:
		limit_f = None
	else:
		try:
			limit_f = float(raw_limit)
		except Exception:
			limit_f = None
	fdv, pr, vcc, tm, temp = _parse_fdv_selector(fdvsel)
	# Build grouped failing rows using same criteria as stats fail_n.
	# We'll rely on raw line PASS/FAIL tokens (passfail_mode) OR threshold comparison.
	failing: List[Dict[str,str]] = []
	for r in rows:
		key = _get_split_tuple(r)
		if (fdv and key[0] != fdv) or (pr and key[1] != pr) or (vcc and key[2] != vcc) or (tm and key[3] != tm) or (temp and key[4] != temp):
			continue
		if (r.get('tname','') or '').strip().upper() == 'PR':
			continue
		rv = _get_rber(r)
		if rv is None:
			continue
		raw_u = (r.get('raw_line','') or '').upper()
		is_fail = False
		if passfail_mode:
			# Require explicit FAIL token (bounded) in raw line.
			if 'FAIL' in raw_u:
				# Avoid counting PASS lines containing substrings; refine with simple regex boundary when cheap.
				try:
					import re as _re
					if _re.search(r'(?<![A-Z0-9_])FAIL(?![A-Z0-9_])', raw_u):
						is_fail = True
					else:
						is_fail = False
				except Exception:
					is_fail = True
		else:
			if limit_f is not None and rv >= limit_f:
				is_fail = True
		if is_fail:
			failing.append(r)
	# Build HTML with raw line highlight
	def esc(s: str) -> str:
		return (s or '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
	# Precompute shading parameters for threshold mode
	max_fail_rber = None
	if not passfail_mode and limit_f is not None and failing:
		try:
			vals = [(_get_rber(r) or 0.0) for r in failing if _get_rber(r) is not None]
			if vals:
				max_fail_rber = max(vals)
		except Exception:
			max_fail_rber = None
	def _shade_style(rv: float | None) -> str:
		if passfail_mode or limit_f is None or rv is None:
			return ''
		try:
			if max_fail_rber is None or max_fail_rber <= limit_f:
				frac = 1.0
			else:
				frac = (rv - limit_f) / (max_fail_rber - limit_f) if rv >= limit_f else 0.0
			if frac < 0: frac = 0.0
			if frac > 1: frac = 1.0
			# Interpolate from light yellow (#fff9cc) to strong red (#ff2a00)
			# Simple linear interpolation in RGB space
			import math as _m
			c1 = (255, 249, 204)
			c2 = (255, 42, 0)
			r = int(c1[0] + (c2[0]-c1[0])*frac)
			g = int(c1[1] + (c2[1]-c1[1])*frac)
			b = int(c1[2] + (c2[2]-c1[2])*frac)
			# Adjust text color for contrast if very red
			fg = '#000'
			if frac > 0.55:
				fg = '#fff'
			return f"style=\"background:rgb({r},{g},{b});color:{fg};\""
		except Exception:
			return ''
	rows_html: List[str] = []
	def _choose_raw_file(rec: Dict[str, str]) -> str:
		# Only accept original .txt source files. Do NOT use fdv_file here.
		cands = [rec.get('source_file'), rec.get('file'), rec.get('filepath'), rec.get('filename')]
		for c in cands:
			p = str(c or '').strip()
			if p and p.lower().endswith('.txt'):
				return p
		# If no .txt known, leave empty rather than showing a derived fdv_file.
		return ''
	for r in failing[:10000]:
		idx = r.get('_idx') or ''  # retained in case future linking needed
		raw_line = r.get('raw_line','') or ''
		raw_disp = esc(raw_line)
		# Highlight FAIL token or numeric threshold exceedance
		if passfail_mode:
			try:
				import re as _re2
				raw_disp = _re2.sub(r'(?i)(?<![A-Z0-9_])(FAIL)(?![A-Z0-9_])', r'<span class="failtok">\1</span>', raw_disp)
			except Exception:
				pass
		else:
			# Highlight RBER value if present and exceeding limit
			if limit_f is not None:
				try:
					import re as _re3
					pat = r'(RBER\s*[:=]\s*)([0-9.eE\-+]+)'
					def _hl(m):
						try:
							val = float(m.group(2))
							if val >= limit_f: return m.group(1) + '<span class="failtok">' + m.group(2) + '</span>'
						except Exception: pass
						return m.group(0)
					raw_disp = _re3.sub(_hl, raw_disp)
				except Exception:
					pass
		rv_cell = _get_rber(r)
		shade_attr = _shade_style(rv_cell)
		# Raw file column: prefer .txt source; show basename; keep full path in title for hover
		_full = _choose_raw_file(r)
		_base = _full.replace('\\','/').split('/')[-1] if _full else ''
		rows_html.append(
			'<tr>'
			f"<td>{esc(r.get('line_number',''))}</td>"
			f"<td>{esc(r.get('testname') or derive_testname((r.get('tname','') or '').strip()) or '')}</td>"
			f"<td>{esc(r.get('dut_id','') or '')}</td>"
			f"<td>{esc(r.get('fuseid','') or '')}</td>"
			f"<td {shade_attr}>{(rv_cell if rv_cell is not None else '')}</td>"
			f"<td class='path' title='{esc(_full)}'>{esc(_base)}</td>"
			f"<td class='rl'>{raw_disp}</td>"
			'</tr>'
		)
	consistency_note = ''
	# Attempt to fetch corresponding stats fail_n to compare
	try:
		stats_for_group = [s for s in (d.get('stats') or []) if s.get('fdv_file','') == fdv and s.get('pr','') == pr and s.get('vcc','') == vcc and s.get('tm','') == tm and s.get('temp','') == temp]
		if stats_for_group:
			fail_n_val = None
			try:
				fail_n_val = int(stats_for_group[0].get('fail_n', stats_for_group[0].get('fail','0')))
			except Exception: fail_n_val = None
			if fail_n_val is not None and fail_n_val != len(failing):
				consistency_note = f"<div style='color:#b00;font-weight:600'>Warning: displayed failing rows ({len(failing)}) != fail count in stats ({fail_n_val}).</div>"
		elif len(failing) == 0:
			consistency_note = '<div style="color:#555">No failing rows matched this selection.</div>'
	except Exception:
		pass
	legend_html = ''
	if not passfail_mode and limit_f is not None and max_fail_rber is not None:
		legend_html = (
			"<div style='margin:6px 0 10px 0;font-size:11px;'>"
			f"<span style='display:inline-block;width:14px;height:14px;vertical-align:middle;background:#fff9cc;border:1px solid #ccc;margin-right:4px;'></span> at limit {limit_f:.3g} "
			"&rarr; "
			f"<span style='display:inline-block;width:14px;height:14px;vertical-align:middle;background:#ff2a00;border:1px solid #ccc;margin-right:4px;'></span> max {max_fail_rber:.3g}"
			" (RBER scale)"
			"</div>"
		)
	html = (
	"<!doctype html><html><head><meta charset='utf-8'><title>Fail Rows</title>"
		"<style>body{font-family:Arial;margin:16px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ccc;padding:4px 6px;font-size:12px;vertical-align:top;}th{background:#f7f7f7;cursor:pointer;} .failtok{background:#ff4444;color:#fff;padding:0 2px;border-radius:2px;} td.rl{white-space:pre-wrap;word-break:break-word;font-family:Consolas,monospace;font-size:11px;} .mono{font-family:Consolas,monospace;} td.path{font-family:Consolas,monospace;font-size:11px;} thead input{width:98%;box-sizing:border-box;font-size:11px;padding:2px;margin:2px 0}</style>"
		"<script>(function(){function toNum(v){var n=parseFloat(v);return isNaN(n)?null:n}function cmp(a,b){if(a===b) return 0;return a<b?-1:1}function sortTable(tbl,colIndex,asNumber,desc){var tbody=tbl.tBodies[0];var rows=[].slice.call(tbody.rows);rows.sort(function(r1,r2){var t1=(r1.cells[colIndex]||{}).innerText||'';var t2=(r2.cells[colIndex]||{}).innerText||'';if(asNumber){var n1=toNum(t1);var n2=toNum(t2);if(n1===null&&n2!==null) return 1;if(n1!==null&&n2===null) return -1;if(n1===null&&n2===null) return 0;return desc?(n2-n1):(n1-n2)} else {t1=t1.toLowerCase();t2=t2.toLowerCase();return desc?(t2>t1?1:(t2<t1?-1:0)):(t1>t2?1:(t1<t2?-1:0))}});rows.forEach(function(r){tbody.appendChild(r)})}function addFilters(tbl){var thead=tbl.tHead; if(!thead) return; var hdr=thead.rows[0]; var filterRow=document.createElement('tr'); for(var i=0;i<hdr.cells.length;i++){var th=document.createElement('th'); var inp=document.createElement('input'); inp.placeholder='filter'; (function(idx){inp.addEventListener('input',function(){var val=this.value.toLowerCase(); var rows=[].slice.call(tbl.tBodies[0].rows); rows.forEach(function(r){var cell=(r.cells[idx]||{}); var txt=(cell.innerText||'').toLowerCase(); r.style.display = (val=='' || txt.indexOf(val)>=0)?'':'none';});});})(i); th.appendChild(inp); filterRow.appendChild(th);} thead.appendChild(filterRow);}function init(){var tbl=document.getElementById('failTable'); if(!tbl) return; addFilters(tbl); var hdr=tbl.tHead.rows[0]; for(let i=0;i<hdr.cells.length;i++){let th=hdr.cells[i]; let numeric = /^(rber|line)/i.test(th.innerText.trim()); let desc=false; th.addEventListener('click',function(){desc=!desc; sortTable(tbl,i,numeric,desc);});}} if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init)} else {init()}})();</script>"
		"</head><body>"
	f"<h3>Fail rows for {esc(fdvsel)}</h3>"
	f"<div>Total failing rows displayed: {len(failing)}</div>"
	+ ("<div>Mode: PASS/FAIL from logs.</div>" if passfail_mode else (f"<div>Threshold (limit) = {limit_f:.6g}</div>" if limit_f is not None else ""))
	+ legend_html
		+ consistency_note
			+ "<table id='failTable'><thead><tr><th>line #</th><th>testname</th><th>DUT</th><th>FUSEID</th><th>RBER</th><th>raw file</th><th>raw line</th></tr></thead><tbody>"
	+ ''.join(rows_html) + "</tbody></table>"
	+ "</body></html>"
	)
	return Response(html, mimetype='text/html')

@app.route('/api/jobs')
def api_jobs():
	jobs_out = []
	def _fmt(ts):
		try:
			return time.strftime('%H:%M:%S', time.localtime(ts)) if ts else None
		except Exception:
			return None
	with JOBS_LOCK:
		items = list(JOBS.items())
	for jid, rec in items:
		tok = rec.get('token')
		data = CACHE.get(tok) or {}
		if data.get('status') == 'deleted':
			continue
		prog = data.get('progress', {}) or {}
		created_at = rec.get('created_at') if isinstance(rec.get('created_at'), (int, float)) else None
		ended_at = rec.get('ended_at') if isinstance(rec.get('ended_at'), (int, float)) else None
		status = data.get('status', 'unknown')
		if data.get('_pause_flag') and status not in {'done','error','deleted'}:
			status = 'paused'
		now = time.time()
		updated_secs = int(now - created_at) if (prog.get('percent') is not None and created_at) else None
		# Server-side ETA (seconds) based on line progress for stability
		eta_secs = None
		if status == 'running':
			try:
				lines_total = prog.get('lines_total') or 0
				lines_done = prog.get('lines') or 0
				if created_at and lines_total and lines_done > 0 and lines_done <= lines_total:
					frac = lines_done / float(lines_total)
					if 0 < frac < 1.0:
						elapsed = now - created_at
						if elapsed > 0:
							remaining = elapsed * (1.0 - frac) / frac
							eta_secs = int(remaining)
			except Exception:
				eta_secs = None
		eta_overall = data.get('eta_overall_secs') if isinstance(data.get('eta_overall_secs'), (int,float)) else None
		jobs_out.append({
			'job_id': jid,
			'token': tok,
			'status': status,
			'percent': prog.get('percent'),
			'files_done': prog.get('files_done'),
			'files_total': prog.get('files_total'),
			'lines': prog.get('lines'),
			'lines_total': prog.get('lines_total'),
			'updated_secs_ago': updated_secs,
			'eta_secs': eta_secs,
			'eta_overall_secs': int(eta_overall) if eta_overall is not None else None,
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

@app.route('/status/<token>', endpoint='report_status')
def status(token: str):
	d = CACHE.get(token)
	if not d:
		return Response(json.dumps({'error':'unknown token'}), mimetype='application/json', status=404)
	return Response(json.dumps({'token': token, 'status': d.get('status'), 'progress': d.get('progress', {})}), mimetype='application/json')

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
		meta = _JOB_REGISTRY.get(job_id)
		if meta is not None:
			meta['name'] = name
			_save_job_registry()
	return jsonify({'ok': True, 'name': name})

def _resolve_cache_by_job(job_id: str):
	with JOBS_LOCK:
		rec = JOBS.get(job_id)
	if not rec:
		return None, None
	token = rec.get('token')
	if not token:
		return None, None
	return token, CACHE.get(token)

@app.route('/api/job/<job_id>/pause', methods=['POST'])
def api_job_pause(job_id: str):
	token, entry = _resolve_cache_by_job(job_id)
	if not token:
		return jsonify({'ok': False, 'error': 'unknown job'}), 404
	if not entry or entry.get('status') not in {'running'}:
		return jsonify({'ok': False, 'error': 'not running'}), 400
	entry['_pause_flag'] = True
	# Immediately reflect paused state for UI
	entry['status'] = 'paused'
	return jsonify({'ok': True, 'status': 'paused'})

@app.route('/api/job/<job_id>/continue', methods=['POST'])
def api_job_continue(job_id: str):
	token, entry = _resolve_cache_by_job(job_id)
	if not token:
		return jsonify({'ok': False, 'error': 'unknown job'}), 404
	if not entry or entry.get('status') not in {'paused'}:
		# Allow continue if still flagged but loop not yet updated
		if not (entry and entry.get('_pause_flag')):
			return jsonify({'ok': False, 'error': 'not paused'}), 400
	if entry:
		entry['_pause_flag'] = False
		entry['status'] = 'running'
	return jsonify({'ok': True, 'status': entry.get('status') if entry else 'unknown'})

@app.route('/api/job/<job_id>/delete', methods=['POST'])
def api_job_delete(job_id: str):
	token, entry = _resolve_cache_by_job(job_id)
	if not token:
		return jsonify({'ok': False, 'error': 'unknown job'}), 404
	if entry:
		entry['_stop_flag'] = True
	with JOBS_LOCK:
		if job_id in JOBS:
			del JOBS[job_id]
		meta = _JOB_REGISTRY.get(job_id)
		if meta is not None:
			meta['status'] = 'deleted'
			if not meta.get('ended_at'):
				meta['ended_at'] = time.time()
			_save_job_registry()
	return jsonify({'ok': True, 'status': 'deleted'})

@app.route('/jobs')
def jobs_page():
	with JOBS_LOCK:
		ids = list(JOBS.keys())
	return render_template('fdv2_jobs.html', job_ids=ids)

# Compatibility endpoints for legacy template names
app.add_url_rule('/', 'report_home', home, methods=['GET','POST'])
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
	passfail_mode_req, numeric_limit = _interpret_limit_mode(limit_q)
	rows = d.get('rows', []) or []
	# Determine if we must recompute stats for this view
	stats_cached = d.get('stats') or []
	need_recompute = False
	if passfail_mode_req:
		# Want PASS/FAIL from logs; recompute only if cached not already passfail_mode
		if not d.get('passfail_mode'):
			need_recompute = True
	else:
		limit_val = numeric_limit
		if d.get('passfail_mode') or (limit_val is not None and limit_val != d.get('limit')):
			need_recompute = True
	if need_recompute and rows:
		try:
			if passfail_mode_req:
				stats_cached = stats_by_fdv_with_splits(rows, limit=1e9, passfail_mode=True)
				# Update cache to reflect mode switch cleanly
				d.update(passfail_mode=True, limit=None, stats=stats_cached)
			else:
				stats_cached = stats_by_fdv_with_splits(rows, limit=limit_val if limit_q else d.get('limit'), passfail_mode=False)
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
		# Build FAIL cell content safely (avoid nested f-strings quoting issues)
		try:
			limit_q = "" if (d.get('passfail_mode') or d.get('limit') is None) else f"&limit={d.get('limit')}"
			href = (
				f"/fdv/{token}/fails?fdv="
				f"{r.get('fdv_file','')}|{r.get('pr','')}|{r.get('vcc','')}|{r.get('tm','')}|{r.get('temp','')}"
				f"{limit_q}"
			)
			fail_cell = str(failn)
			if (failn > 0) and token:
				fail_cell = f'<a href="{href}" target="_blank" rel="noopener">{failn}</a>'
		except Exception:
			fail_cell = str(failn)
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
			f"<td>{fail_cell}</td>"
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
