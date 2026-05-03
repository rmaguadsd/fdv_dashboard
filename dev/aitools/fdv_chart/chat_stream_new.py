        elif self.path == '/chat_stream':
            # Streaming chat via Server-Sent Events
            # Strategy: Request non-streaming response from Ollama, split into words, stream back to client
            try:
                length  = int(self.headers.get('Content-Length', 0))
                body    = json.loads(self.rfile.read(length).decode('utf-8'))
                csv_id  = body.get('csv_id', 'default') or 'default'
                message = body.get('message', '').strip()
                context = body.get('context', '').strip()
                model   = body.get('model', '').strip() or _LLM_MODEL
                if not message:
                    raise ValueError('Empty message')

                # ── Build / update session history (same logic as /chat) ──
                with _chat_sessions_lock:
                    if csv_id not in _chat_sessions:
                        sess = [{'role': 'system', 'content': _LLM_SYSTEM_PROMPT}]
                        if context:
                            sess.append({'role': 'system',
                                         'content': 'Current chart context:\n' + context})
                        _chat_sessions[csv_id] = sess
                    elif context:
                        sess = _chat_sessions[csv_id]
                        replaced = False
                        for i, m in enumerate(sess):
                            if m['role'] == 'system' and (
                                    m['content'].startswith('Current chart context:') or
                                    m['content'].startswith('Updated chart context:')):
                                sess[i] = {'role': 'system',
                                           'content': 'Updated chart context:\n' + context}
                                replaced = True
                                break
                        if not replaced:
                            sess.append({'role': 'system',
                                         'content': 'Updated chart context:\n' + context})

                    _chat_sessions[csv_id].append({'role': 'user', 'content': message})

                    # Trim history
                    sess        = _chat_sessions[csv_id]
                    system_msgs = [m for m in sess if m['role'] == 'system']
                    conv_msgs   = [m for m in sess if m['role'] != 'system']
                    max_conv    = _CHAT_MAX_TURNS * 2
                    if len(conv_msgs) > max_conv:
                        conv_msgs = conv_msgs[-max_conv:]
                    _chat_sessions[csv_id] = system_msgs + conv_msgs
                    messages_snapshot = list(_chat_sessions[csv_id])

                # ── Send SSE headers ──
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()

                print(f'[CHAT_STREAM] Headers sent, model={model}', flush=True)

                # ── Get response from Ollama (non-streaming) ──
                # This avoids all the HTTP streaming issues
                payload = json.dumps({
                    'model':       model,
                    'messages':    messages_snapshot,
                    'stream':      False,  # KEY: Don't stream from Ollama
                    'temperature': 0.3
                }).encode('utf-8')
                
                print(f'[CHAT_STREAM] Requesting from Ollama (non-streaming)...', flush=True)
                req = urllib.request.Request(
                    'http://localhost:11434/api/chat',
                    data=payload,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                
                try:
                    print(f'[CHAT_STREAM] Opening connection...', flush=True)
                    with urllib.request.urlopen(req, timeout=300) as response:
                        print(f'[CHAT_STREAM] Got response, reading...', flush=True)
                        resp_data = response.read().decode('utf-8')
                        print(f'[CHAT_STREAM] Response size: {len(resp_data)} bytes', flush=True)
                        
                        msg_obj = json.loads(resp_data)
                        full_text = msg_obj.get('message', {}).get('content', '')
                        print(f'[CHAT_STREAM] Response text length: {len(full_text)}', flush=True)
                        
                        # Stream the response back to client word-by-word
                        token_count = 0
                        words = full_text.split()
                        for word in words:
                            if word:
                                token_count += 1
                                # Send word + space as a token
                                sse_line = 'data: ' + json.dumps({'token': word + ' '}) + '\n\n'
                                self.wfile.write(sse_line.encode('utf-8'))
                                self.wfile.flush()
                                if token_count % 10 == 0:
                                    print(f'[CHAT_STREAM] Sent {token_count} tokens', flush=True)
                        
                        print(f'[CHAT_STREAM] Finished sending {token_count} tokens', flush=True)
                        
                        # Save to history
                        with _chat_sessions_lock:
                            _chat_sessions[csv_id].append({'role': 'assistant', 'content': full_text})
                        
                        # Send done marker
                        self.wfile.write(b'data: {"done":true}\n\n')
                        self.wfile.flush()
                        print(f'[CHAT_STREAM] Complete!', flush=True)
                        
                except urllib.error.URLError as ue:
                    print(f'[CHAT_STREAM] URLError: {ue}', flush=True)
                    err_msg = f'Connection error to Ollama: {str(ue)[:100]}'
                    self.wfile.write(('data: ' + json.dumps({'error': err_msg}) + '\n\n').encode('utf-8'))
                    self.wfile.flush()
                except Exception as inner_err:
                    print(f'[CHAT_STREAM] Error: {type(inner_err).__name__}: {inner_err}', flush=True)
                    err_msg = f'Stream error: {str(inner_err)[:100]}'
                    self.wfile.write(('data: ' + json.dumps({'error': err_msg}) + '\n\n').encode('utf-8'))
                    self.wfile.flush()
                    
            except Exception as e:
                print(f'[CHAT_STREAM] Outer exception: {type(e).__name__}: {e}', flush=True)
                try:
                    err_data = json.dumps({'error': str(e)})
                    self.wfile.write(('data: ' + err_data + '\n\n').encode('utf-8'))
                    self.wfile.flush()
                except Exception:
                    pass
