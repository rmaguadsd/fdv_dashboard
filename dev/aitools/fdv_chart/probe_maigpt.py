"""Probe ConnectMaiGPT: test different modelname values and plain-data awareness."""
import http.client, ssl, json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
host  = 'fmgnpsgautoplt01.elements.local'
TOKEN = 'Token d4f8e2a1-3c9b-4e7a-9b2f-1a2b3c4d5e6f'
USER  = 'russel.maguad@solidigm.com'

def mcp_post(sid, payload, timeout=60):
    body = json.dumps(payload).encode()
    conn = http.client.HTTPSConnection(host, context=ctx, timeout=timeout)
    conn.request('POST', '/mcp', body=body, headers={
        'Content-Type':'application/json','Accept':'application/json, text/event-stream',
        'Authorization':TOKEN,'X-User-ID':USER,'mcp-session-id':sid})
    r = conn.getresponse(); raw = r.read(32768).decode('utf-8','replace'); conn.close()
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith('data:'):
            try: out.append(json.loads(line[5:].strip()))
            except: pass
    return out

def new_session():
    conn = http.client.HTTPSConnection(host, context=ctx, timeout=10)
    conn.request('POST', '/mcp',
        body=json.dumps({'jsonrpc':'2.0','id':1,'method':'initialize',
            'params':{'protocolVersion':'2024-11-05','capabilities':{},
                      'clientInfo':{'name':'fdv','version':'1.0'}}}).encode(),
        headers={'Content-Type':'application/json','Accept':'application/json, text/event-stream',
                 'Authorization':TOKEN,'X-User-ID':USER})
    r = conn.getresponse(); sid = r.getheader('mcp-session-id',''); r.read(); conn.close()
    mcp_post(sid, {'jsonrpc':'2.0','id':2,'method':'tools/call',
        'params':{'name':'connectmaigpt',
                  'arguments':{'userid':USER,'apikey':'d4f8e2a1-3c9b-4e7a-9b2f-1a2b3c4d5e6f'}}})
    return sid

DATA_QUERY = (
    "You are a data analysis assistant. "
    "Here is test measurement data:\n"
    "  DUT1: mean RBER = 4.1e-5\n"
    "  DUT2: mean RBER = 6.8e-4\n"
    "  DUT3: mean RBER = 2.2e-4\n\n"
    "Question: Which DUT has the worst RBER and by how much compared to the best?"
)

for model in ['gpt-4o', 'gpt4o', 'gpt-4', 'gpt4', 'claude-3-5-sonnet', None]:
    sid = new_session()
    args = {
        'query':           DATA_QUERY,
        'username':        USER,
        'chattitle':       'FDV Data Analysis',
        'includeprogress': False
    }
    if model is not None:
        args['modelname'] = model

    res = mcp_post(sid, {'jsonrpc':'2.0','id':3,'method':'tools/call',
        'params':{'name':'askmaigpt','arguments':args}}, timeout=60)

    reply = ''
    agentname = ''
    actual_model = ''
    for obj in res:
        sc = obj.get('result',{}).get('structuredContent',{})
        rb = sc.get('response',{})
        if isinstance(rb, dict):
            reply       = rb.get('response','')
            agentname   = rb.get('agentname','')
            actual_model= rb.get('modelname','')
        if reply:
            break

    print(f"\n--- modelname={model!r} -> agentname={agentname!r} model={actual_model!r} ---")
    print(reply[:300] if reply else '(no reply)')
