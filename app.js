// app.js (module)
// Entangle Chat-2 — deterministic twin + Firebase signaling
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js';
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import {
  getDatabase,
  ref as dbRef,
  set as dbSet,
  push as dbPush,
  onChildAdded,
  onValue,
  remove as dbRemove
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-database.js";

/* ---------- Firebase config (you provided) ---------- */
const firebaseConfig = {
  apiKey: "AIzaSyDyOa1l-Xrw0rarEk2IRg3p0JoT40XHJLQ",
  authDomain: "entangle-chat-2090f.firebaseapp.com",
  projectId: "entangle-chat-2090f",
  storageBucket: "entangle-chat-2090f.firebasestorage.app",
  messagingSenderId: "448597724751",
  appId: "1:448597724751:web:2f74b317f8cd761094643a",
  measurementId: "G-8LQX0PMXGP"
};

/* ---------- Utilities ---------- */
function djb2Hash(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h) + str.charCodeAt(i);
    h = h >>> 0;
  }
  return h;
}
function seededRandomFactory(seed) {
  let s = seed >>> 0;
  return function() {
    s ^= s << 13; s = s >>> 0;
    s ^= s >>> 17; s = s >>> 0;
    s ^= s << 5; s = s >>> 0;
    return (s >>> 0) / 0xFFFFFFFF;
  };
}
function bytesFromString(str) {
  return new TextEncoder().encode(str);
}
function stringFromBytes(bytes) {
  return new TextDecoder().decode(new Uint8Array(bytes));
}
function base64FromBytes(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes)));
}
function bytesFromBase64(b64) {
  return Uint8Array.from(atob(b64), c => c.charCodeAt(0));
}
function xorBuffers(buf, otp) {
  const out = new Uint8Array(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = buf[i] ^ otp[i % otp.length];
  return out;
}
async function sha256Bytes(...parts) {
  // parts: ArrayBuffer / TypedArray / Uint8Array
  const concatenated = parts.reduce((acc, p) => {
    const u = (p instanceof ArrayBuffer) ? new Uint8Array(p) : (p instanceof Uint8Array ? p : new Uint8Array(p.buffer || p));
    const tmp = new Uint8Array(acc.length + u.length);
    tmp.set(acc, 0);
    tmp.set(u, acc.length);
    return tmp;
  }, new Uint8Array(0));
  const hash = await crypto.subtle.digest('SHA-256', concatenated);
  return new Uint8Array(hash);
}

/* ---------- Deterministic twin model & OTP ---------- */
let model = null;
async function buildDeterministicTwin(seedPhrase) {
  await tf.ready();
  if (model) return model;

  model = tf.sequential();
  model.add(tf.layers.dense({units:64, activation:'relu', inputShape:[33]}));
  model.add(tf.layers.dense({units:32, activation:'tanh'}));

  const seed = djb2Hash(seedPhrase);
  const rnd = seededRandomFactory(seed);

  const k1 = [33,64], b1 = [64], k2 = [64,32], b2 = [32];
  const tot1 = k1[0]*k1[1], tot2 = k2[0]*k2[1];

  const makeArr = (n) => {
    const a = new Float32Array(n);
    for (let i=0;i<n;i++) a[i] = (rnd() * 2 - 1) * 0.5;
    return a;
  };
  const w = [
    tf.tensor2d(makeArr(tot1), k1),
    tf.tensor1d(makeArr(b1[0])),
    tf.tensor2d(makeArr(tot2), k2),
    tf.tensor1d(makeArr(b2[0]))
  ];
  model.setWeights(w);
  w.forEach(t => t.dispose && t.dispose());
  return model;
}

function inferVector(seedBytes) {
  const input = tf.tensor2d([seedBytes.map(v => v/255)], [1,33]);
  const out = model.predict(input);
  const arr = out.dataSync();
  out.dispose(); input.dispose();
  return arr;
}

async function deriveOTP(seedPhrase, vec) {
  // Use WebCrypto SHA-256 over (seedPhrase bytes || vector bytes) => 32 bytes OTP
  const phraseBytes = bytesFromString(seedPhrase);
  // convert vec (Float32Array or regular array) to bytes
  const f32 = new Float32Array(vec);
  const vecBytes = new Uint8Array(f32.buffer);
  const hash = await sha256Bytes(phraseBytes, vecBytes);
  // hash is 32 bytes; use as OTP
  return new Uint8Array(hash);
}

/* ---------- UI: QR generation & scanning ---------- */
const qrcodeEl = document.getElementById('qrcode');
const qrCanvas = document.getElementById('qrcvs');
const cam = document.getElementById('cam');

document.getElementById('make-qr').addEventListener('click', () => {
  const phrase = document.getElementById('seed-input').value || ('entangle:' + Math.random().toString(36).slice(2,12));
  qrcodeEl.innerHTML = '';
  new QRCode(qrcodeEl, { text: phrase, width: 192, height: 192 });
  window.currentSeedPhrase = phrase;
  statusSet('seed generated');
  prepareTwinFromPhrase(phrase);
});

document.getElementById('scan-btn').addEventListener('click', async () => {
  statusSet('starting camera for scan');
  try {
    cam.hidden = false;
    const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
    cam.srcObject = stream;
    await cam.play();

    const ctx = qrCanvas.getContext('2d');
    qrCanvas.width = 320; qrCanvas.height = 240;
    const scanLoop = () => {
      if (cam.paused || cam.ended) return;
      ctx.drawImage(cam,0,0,qrCanvas.width,qrCanvas.height);
      const img = ctx.getImageData(0,0,qrCanvas.width,qrCanvas.height);
      const code = jsQR(img.data, img.width, img.height);
      if (code) {
        window.currentSeedPhrase = code.data;
        statusSet('scanned seed');
        cam.srcObject.getTracks().forEach(t => t.stop());
        cam.hidden = true;
        prepareTwinFromPhrase(code.data);
      } else {
        requestAnimationFrame(scanLoop);
      }
    };
    scanLoop();
  } catch (e) {
    console.error(e);
    statusSet('camera error: ' + e.message);
  }
});

/* ---------- Prepare twin & OTP preview ---------- */
async function prepareTwinFromPhrase(phrase) {
  statusSet('preparing twin model');
  await buildDeterministicTwin(phrase);
  const bytes = bytesFromString(phrase);
  const seedBytes = new Array(33).fill(0).map((_,i) => bytes[i % bytes.length] || 0);
  const vec = inferVector(seedBytes);
  const otp = await deriveOTP(phrase, vec);
  window._currentOTP = otp;
  document.getElementById('twin-hash').textContent = (djb2Hash(phrase) >>> 0).toString(16).slice(0,8);
  document.getElementById('otp-preview').textContent = Array.from(otp.slice(0,32)).map(b=>b.toString(16).padStart(2,'0')).join(' ');
  statusSet('twin ready');
}

/* ---------- Text encrypt/decrypt helpers ---------- */
function encryptText(plain) {
  if (!window._currentOTP) throw new Error('OTP not ready');
  const bytes = bytesFromString(plain);
  const enc = xorBuffers(bytes, window._currentOTP);
  return base64FromBytes(enc);
}
function decryptText(b64) {
  if (!window._currentOTP) throw new Error('OTP not ready');
  const enc = bytesFromBase64(b64);
  const dec = xorBuffers(enc, window._currentOTP);
  return stringFromBytes(dec);
}

/* ---------- UI helpers ---------- */
const messagesEl = document.getElementById('messages');
function appendMessage(txt, cls='system') {
  const div = document.createElement('div');
  div.className = cls;
  div.innerText = txt;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------- Send / receive UI ---------- */
document.getElementById('send-txt').addEventListener('click', async () => {
  const txt = document.getElementById('msg-input').value;
  if (!txt) return;
  try {
    const cipher = encryptText(txt);
    appendMessage('SENT (cipher): ' + cipher, 'sent');
    if (window._dc && window._dc.readyState === 'open') {
      window._dc.send(JSON.stringify({type:'text', payload:cipher}));
    } else {
      navigator.clipboard && navigator.clipboard.writeText(cipher).catch(()=>{});
      appendMessage('No P2P channel. Cipher copied to clipboard (or copy manually).', 'system');
    }
  } catch (e) {
    appendMessage('Encrypt error: ' + e.message, 'system');
  }
});

document.getElementById('decrypt-last').addEventListener('click', () => {
  const nodes = messagesEl.querySelectorAll('.received');
  if (!nodes.length) { appendMessage('No received messages found', 'system'); return; }
  const last = nodes[nodes.length-1].innerText;
  const m = last.match(/RECV.*:\s*(.+)$/);
  if (!m) { appendMessage('No cipher found to decrypt', 'system'); return; }
  try {
    const plain = decryptText(m[1].trim());
    appendMessage('DECRYPTED: ' + plain, 'system');
  } catch (e) {
    appendMessage('Decrypt failed: ' + e.message, 'system');
  }
});

/* ---------- WebRTC + Firebase signaling ---------- */
let pc = null;
let dataChannel = null;
let localStream = null;
let firebaseApp = null;
let database = null;
let currentRoomId = null; // used when using Firebase flow

function initFirebase() {
  if (firebaseApp) return;
  firebaseApp = initializeApp(firebaseConfig);
  database = getDatabase(firebaseApp);
}

function roomPath(roomId, node='') {
  return `/rooms/${roomId}${node ? '/' + node : ''}`;
}
async function pushCandidate(roomId, side, candidate) {
  try {
    await dbPush(dbRef(database, roomPath(roomId, `candidates/${side}`)), candidate);
  } catch (e) { console.warn('pushCandidate failed', e); }
}
async function writeAnswer(roomId, pcRef) {
  if (!pcRef.localDescription) throw new Error('Answer not created yet');
  await dbSet(dbRef(database, roomPath(roomId, 'answer')), JSON.parse(JSON.stringify(pcRef.localDescription)));
}
async function cleanupRoom(roomId) {
  if (!roomId) return;
  try { await dbRemove(dbRef(database, roomPath(roomId))); } catch(e){/*ignore*/ }
}

async function createRoomFirebase(pcRef) {
  initFirebase();
  const roomId = Math.random().toString(36).slice(2,10);
  // write offer
  await dbSet(dbRef(database, roomPath(roomId, 'offer')), JSON.parse(JSON.stringify(pcRef.localDescription)));
  // listen for answer
  onValue(dbRef(database, roomPath(roomId, 'answer')), async snap => {
    const data = snap.val();
    if (!data) return;
    try { await pcRef.setRemoteDescription(data); appendMessage('Remote answer applied (via Firebase)', 'system'); }
    catch(e){console.warn('apply answer failed', e);}
  });
  // listen for remote ICE (answerer)
  onChildAdded(dbRef(database, roomPath(roomId, 'candidates/answer')), async snap => {
    const c = snap.val();
    if (!c) return;
    try { await pcRef.addIceCandidate(c); } catch(e){console.warn('addIceCandidate failed', e);}
  });
  return roomId;
}

async function joinRoomFirebase(pcRef, roomId) {
  initFirebase();
  // read offer
  const snap = await new Promise((res, rej) => {
    onValue(dbRef(database, roomPath(roomId, 'offer')), s => res(s), err => rej(err));
  });
  const offer = snap.val();
  if (!offer) throw new Error('Offer not found');
  await pcRef.setRemoteDescription(offer);

  // listen for remote ICE (offerer)
  onChildAdded(dbRef(database, roomPath(roomId, 'candidates/offer')), async snap => {
    const c = snap.val();
    if (!c) return;
    try { await pcRef.addIceCandidate(c); } catch(e){console.warn('addIceCandidate failed', e);}
  });

  appendMessage(`Joined room ${roomId} (offer loaded)`, 'system');
  return offer;
}

/* RTCPeerConnection lifecycle helpers */
async function ensurePeer() {
  if (pc) return pc;
  pc = new RTCPeerConnection({ iceServers: [{urls:'stun:stun.l.google.com:19302'}] });

  pc.onicecandidate = event => {
    if (!event.candidate) return;
    // determine side
    const side = (pc.localDescription && pc.localDescription.type === 'offer') ? 'offer' : 'answer';
    if (currentRoomId) {
      pushCandidate(currentRoomId, side, event.candidate);
    } else {
      // manual mode: append candidate to signaling textarea so users can copy if needed
      // (not required in many browsers)
      appendMessage('Local ICE candidate gathered (manual mode).', 'system');
    }
  };

  pc.ontrack = e => {
    document.getElementById('remote-audio').srcObject = e.streams[0];
  };

  pc.ondatachannel = evt => {
    setupDataChannel(evt.channel);
  };

  return pc;
}

function setupDataChannel(dc) {
  window._dc = dc;
  dataChannel = dc;
  dc.onopen = () => appendMessage('DataChannel open', 'system');
  dc.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'text') {
        appendMessage('RECV (cipher): ' + msg.payload, 'received');
      } else {
        appendMessage('RECV: ' + e.data,'received');
      }
    } catch (err) {
      appendMessage('RECV RAW: ' + e.data,'received');
    }
  };
}

/* Create offer (manual) */
document.getElementById('create-offer').addEventListener('click', async () => {
  await ensurePeer();
  const dc = pc.createDataChannel('entangle-text');
  setupDataChannel(dc);

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // WAIT briefly for ICE (simple)
  setTimeout(() => {
    document.getElementById('signaling').value = JSON.stringify(pc.localDescription);
    appendMessage('Offer created. Paste this offer to partner (manual).', 'system');
    document.getElementById('start-voice').disabled = false;
  }, 800);
});

/* Create Room (Firebase) */
document.getElementById('create-room-fb').addEventListener('click', async () => {
  await ensurePeer();
  const dc = pc.createDataChannel('entangle-text');
  setupDataChannel(dc);

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // wait briefly for local ICE candidates collection
  setTimeout(async () => {
    try {
      const roomId = await createRoomFirebase(pc);
      currentRoomId = roomId;
      document.getElementById('signaling').value = roomId;
      appendMessage('Room created: ' + roomId + ' (share this ID with partner)', 'system');
      document.getElementById('start-voice').disabled = false;
    } catch (e) {
      appendMessage('Create room failed: ' + e.message, 'system');
    }
  }, 1000);
});

/* Accept offer (manual) */
document.getElementById('accept-offer').addEventListener('click', async () => {
  const text = document.getElementById('signaling').value;
  if (!text) { appendMessage('Paste offer here first', 'system'); return; }
  try {
    const obj = JSON.parse(text);
    await ensurePeer();
    await pc.setRemoteDescription(obj);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    setTimeout(() => {
      document.getElementById('signaling').value = JSON.stringify(pc.localDescription);
      appendMessage('Answer created. Send this back to offerer (manual).', 'system');
      document.getElementById('start-voice').disabled = false;
    }, 500);
  } catch (e) {
    appendMessage('Accept failed: ' + e.message, 'system');
  }
});

/* Join Room (Firebase) */
document.getElementById('join-room-fb').addEventListener('click', async () => {
  const roomId = document.getElementById('signaling').value.trim();
  if (!roomId) { appendMessage('Paste the Room ID to join', 'system'); return; }
  try {
    await ensurePeer();
    // set currentRoomId so ICE candidates will be pushed
    currentRoomId = roomId;
    const offer = await joinRoomFirebase(pc, roomId);
    // create and send answer
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await writeAnswer(roomId, pc);
    appendMessage('Answer written to Firebase. Waiting for connection...', 'system');
    document.getElementById('start-voice').disabled = false;
  } catch (e) {
    appendMessage('Join room failed: ' + e.message, 'system');
  }
});

/* start voice: add local audio tracks and update localDescription */
document.getElementById('start-voice').addEventListener('click', async () => {
  if (!pc) { appendMessage('Create/accept offer first', 'system'); return; }
  try {
    localStream = await navigator.mediaDevices.getUserMedia({audio:true});
    localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
    const desc = await pc.createOffer();
    await pc.setLocalDescription(desc);
    // If using Firebase flow and we have a roomId, update DB with new localDescription
    if (currentRoomId && pc.localDescription && pc.localDescription.type === 'offer') {
      // update the existing offer in DB so remote can setRemoteDescription again (simple approach)
      await dbSet(dbRef(database, roomPath(currentRoomId, 'offer')), JSON.parse(JSON.stringify(pc.localDescription)));
      appendMessage('Local audio added and offer updated in Firebase.', 'system');
    } else {
      appendMessage('Local audio added. In manual flow, share updated SDP if needed.', 'system');
    }
    document.getElementById('hangup').disabled = false;
  } catch (e) {
    appendMessage('Microphone error: ' + e.message, 'system');
  }
});

/* Hangup */
document.getElementById('hangup').addEventListener('click', async () => {
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  if (currentRoomId) {
    // cleanup optional
    try { await cleanupRoom(currentRoomId); } catch(e){}
    currentRoomId = null;
  }
  appendMessage('Call ended', 'system');
  document.getElementById('hangup').disabled = true;
});

/* manual signaling textarea handler - if offerer pastes answer, it will be applied */
document.getElementById('signaling').addEventListener('input', async () => {
  try {
    const s = document.getElementById('signaling').value.trim();
    if (!s) return;
    // if contains JSON offer/answer
    if (s.startsWith('{')) {
      const obj = JSON.parse(s);
      if (!pc || !pc.localDescription) return;
      if (pc.localDescription.type === 'offer' && obj.type === 'answer') {
        await pc.setRemoteDescription(obj);
        appendMessage('Remote answer set (manual)', 'system');
      }
    }
  } catch (e) {
    // ignore parse issues
  }
});

/* DataChannel receive handler - already handled in setupDataChannel */

/* ---------- Expose debug API ---------- */
window.entangle = {
  buildDeterministicTwin,
  inferVector,
  deriveOTP,
  encryptText,
  decryptText,
  prepareTwinFromPhrase
};

function statusSet(txt) {
  document.getElementById('statustxt').textContent = txt;
  console.log('status:', txt);
}