// app.js (module) - Firebase signaling + WebRTC DataChannel + AI-OTP encryption
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

/* ---------- FIREBASE CONFIG (your config already provided) ---------- */
const firebaseConfig = {
  apiKey: "AIzaSyDyOa1l-Xrw0rarEk2IRg3p0JoT40XHJLQ",
  authDomain: "entangle-chat-2090f.firebaseapp.com",
  projectId: "entangle-chat-2090f",
  storageBucket: "entangle-chat-2090f.firebasestorage.app",
  messagingSenderId: "448597724751",
  appId: "1:448597724751:web:2f74b317f8cd761094643a",
  measurementId: "G-8LQX0PMXGP"
};

/* ---------- DOM ---------- */
const makeQRBtn = document.getElementById('make-qr') || document.getElementById('makeQR') || document.getElementById('genQR');
const scanBtn = document.getElementById('scan-btn') || document.getElementById('scanQR');
const qrArea = document.getElementById('qrcode') || document.getElementById('qrArea');
const cam = document.getElementById('cam');
const qrcvs = document.getElementById('qrcvs');

const createRoomBtn = document.getElementById('create-room-fb');
const joinRoomBtn = document.getElementById('join-room-fb');
const roomIdInput = document.getElementById('room-id');
const copyRoomBtn = document.getElementById('copy-room');

const sendBtn = document.getElementById('send-btn') || document.getElementById('sendBtn');
const msgInput = document.getElementById('msg-input') || document.getElementById('msgInput');
const messagesEl = document.getElementById('messages');

const statusEl = document.getElementById('statustxt');
const twinHashEl = document.getElementById('twin-hash');
const otpPreviewEl = document.getElementById('otp-preview');

function appendMessage(txt, cls='sys') {
  const d = document.createElement('div');
  d.className = cls;
  d.innerText = txt;
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------- Utilities ---------- */
function djb2Hash(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h) + str.charCodeAt(i);
    h = h >>> 0;
  }
  return h;
}
function bytesFromString(str) { return new TextEncoder().encode(str); }
function stringFromBytes(bytes) { return new TextDecoder().decode(new Uint8Array(bytes)); }
function base64FromBytes(bytes) { return btoa(String.fromCharCode(...new Uint8Array(bytes))); }
function bytesFromBase64(b64) { return Uint8Array.from(atob(b64), c => c.charCodeAt(0)); }
function xorBuffers(buf, otp) {
  const out = new Uint8Array(buf.length);
  for (let i=0;i<buf.length;i++) out[i] = buf[i] ^ otp[i % otp.length];
  return out;
}
async function sha256Bytes(...parts) {
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

/* ---------- Deterministic Twin Model (inline here — small) ---------- */
let model = null;
async function buildDeterministicTwin(seedArray) {
  await tf.ready();
  if (model) return model;
  model = tf.sequential();
  model.add(tf.layers.dense({units:64, activation:'relu', inputShape:[33]}));
  model.add(tf.layers.dense({units:32, activation:'tanh'}));

  // deterministic weight gen from seed (simple seeded RNG)
  const seed = seedArray.reduce((a,b)=>a*1315423911 + b, 0) >>> 0;
  function rndFactory(s) {
    let x = s >>> 0;
    return () => {
      x ^= x << 13; x = x >>> 0;
      x ^= x >>> 17; x = x >>> 0;
      x ^= x << 5; x = x >>> 0;
      return (x >>> 0) / 0xFFFFFFFF;
    };
  }
  const rnd = rndFactory(seed);
  const k1 = [33,64], b1 = [64], k2 = [64,32], b2 = [32];
  const tot1 = k1[0]*k1[1], tot2 = k2[0]*k2[1];
  const makeArr = (n) => {
    const a = new Float32Array(n);
    for (let i=0;i<n;i++) a[i] = (rnd()*2 - 1) * 0.5;
    return a;
  };
  const w = [
    tf.tensor2d(makeArr(tot1), k1),
    tf.tensor1d(makeArr(b1[0])),
    tf.tensor2d(makeArr(tot2), k2),
    tf.tensor1d(makeArr(b2[0]))
  ];
  model.setWeights(w);
  w.forEach(t=>t.dispose && t.dispose());
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
  const phraseBytes = bytesFromString(seedPhrase);
  const f32 = new Float32Array(vec);
  const vecBytes = new Uint8Array(f32.buffer);
  const hash = await sha256Bytes(phraseBytes, vecBytes);
  return new Uint8Array(hash); // 32 bytes OTP
}

/* ---------- Pairing (QR generate & scan) ---------- */
let currentSeedPhrase = null; // we store as string (JSON array or passphrase)
let currentSeedArray = null;  // Uint8Array
let currentOTP = null;

function statusSet(s) { statusEl.textContent = s; console.log('status:', s); }

(makeQRBtn && makeQRBtn.addEventListener('click', () => {
  const pass = document.getElementById('seed-input')?.value?.trim();
  if (pass) {
    currentSeedPhrase = pass;
    // generate deterministic seed array from pass (hash)
    currentSeedArray = bytesFromString(pass).slice(0,33);
  } else {
    // random seed
    const arr = new Uint8Array(33);
    crypto.getRandomValues(arr);
    currentSeedArray = arr;
    currentSeedPhrase = JSON.stringify(Array.from(arr));
  }
  qrArea.innerHTML = '';
  new QRCode(qrArea, currentSeedPhrase);
  statusSet('seed generated');
  prepareTwinFromPhrase(currentSeedPhrase, currentSeedArray);
})) || null;

(scanBtn && scanBtn.addEventListener('click', async () => {
  statusSet('starting camera');
  try {
    cam.hidden = false;
    const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
    cam.srcObject = stream;
    await cam.play();
    const ctx = qrcvs.getContext('2d');
    qrcvs.width = 320; qrcvs.height = 240;
    const loop = () => {
      if (cam.paused || cam.ended) return;
      ctx.drawImage(cam,0,0,qrcvs.width,qrcvs.height);
      const img = ctx.getImageData(0,0,qrcvs.width,qrcvs.height);
      const code = jsQR(img.data, img.width, img.height);
      if (code) {
        cam.srcObject.getTracks().forEach(t => t.stop());
        cam.hidden = true;
        currentSeedPhrase = code.data;
        // if QR contained JSON array, parse to Uint8Array; else treat as passphrase
        try {
          const parsed = JSON.parse(code.data);
          if (Array.isArray(parsed)) {
            currentSeedArray = new Uint8Array(parsed.slice(0,33));
          } else {
            currentSeedArray = bytesFromString(code.data).slice(0,33);
          }
        } catch(e) {
          currentSeedArray = bytesFromString(code.data).slice(0,33);
        }
        statusSet('seed scanned');
        prepareTwinFromPhrase(currentSeedPhrase, currentSeedArray);
      } else {
        requestAnimationFrame(loop);
      }
    };
    loop();
  } catch (e) {
    console.error(e);
    statusSet('camera error: ' + e.message);
  }
})) || null;

/* ---------- Prepare twin and OTP preview ---------- */
async function prepareTwinFromPhrase(phrase, arr) {
  statusSet('preparing twin');
  await buildDeterministicTwin(Array.from(arr));
  const vec = inferVector(Array.from(arr));
  const otp = await deriveOTP(phrase, vec);
  currentOTP = otp;
  twinHashEl.textContent = (djb2Hash(phrase) >>> 0).toString(16).slice(0,8);
  otpPreviewEl.textContent = Array.from(otp.slice(0,32)).map(b => b.toString(16).padStart(2,'0')).join(' ');
  statusSet('twin ready');
}

/* ---------- Encryption helpers ---------- */
function encryptText(plain) {
  if (!currentOTP) throw new Error('OTP not ready');
  const bytes = bytesFromString(plain);
  const enc = xorBuffers(bytes, currentOTP);
  return base64FromBytes(enc);
}
function decryptText(b64) {
  if (!currentOTP) throw new Error('OTP not ready');
  const enc = bytesFromBase64(b64);
  const dec = xorBuffers(enc, currentOTP);
  return stringFromBytes(dec);
}

/* ---------- Firebase init ---------- */
let firebaseApp = null;
let database = null;
function initFirebase() {
  if (firebaseApp) return;
  firebaseApp = initializeApp(firebaseConfig);
  database = getDatabase(firebaseApp);
}

/* ---------- Realtime DB helper ---------- */
function roomPath(roomId, node='') { return `/rooms/${roomId}${node ? '/' + node : ''}`; }
async function pushCandidate(roomId, side, candidate) {
  try { await dbPush(dbRef(database, roomPath(roomId, `candidates/${side}`)), candidate); } catch(e){console.warn('pushCandidate failed', e);}
}

/* ---------- WebRTC + Firebase Signaling (DataChannel) ---------- */
let pc = null;
let dc = null;
let currentRoomId = null;

async function ensurePeer() {
  if (pc) return pc;
  pc = new RTCPeerConnection({ iceServers:[{urls:'stun:stun.l.google.com:19302'}] });

  pc.onicecandidate = e => {
    if (!e.candidate) return;
    const side = (pc.localDescription && pc.localDescription.type === 'offer') ? 'offer' : 'answer';
    if (currentRoomId) pushCandidate(currentRoomId, side, e.candidate);
    else appendMessage('Local ICE candidate gathered (manual flow).', 'sys');
  };

  pc.ondatachannel = evt => {
    setupDataChannel(evt.channel);
  };

  return pc;
}

function setupDataChannel(channel) {
  dc = channel;
  window._dc = dc;
  dc.onopen = () => appendMessage('DataChannel open', 'sys');
  dc.onmessage = e => {
    try {
      const obj = JSON.parse(e.data);
      if (obj.type === 'text') {
        // received ciphertext; show and allow decrypt
        appendMessage('RECV (cipher): ' + obj.payload, 'recv');
      } else {
        appendMessage('RECV: ' + e.data, 'recv');
      }
    } catch (err) {
      appendMessage('RECV RAW: ' + e.data, 'recv');
    }
  };
}

/* Create Room (Firebase) */
createRoomBtn && createRoomBtn.addEventListener('click', async () => {
  if (!currentSeedArray) { appendMessage('Pair first (generate/scan QR)', 'sys'); return; }
  initFirebase();
  await ensurePeer();
  // create data channel
  const channel = pc.createDataChannel('entangle-text');
  setupDataChannel(channel);

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // wait briefly for ICE to be gathered (simple)
  setTimeout(async () => {
    try {
      const roomId = Math.random().toString(36).slice(2,10);
      currentRoomId = roomId;
      await dbSet(dbRef(database, roomPath(roomId, 'offer')), JSON.parse(JSON.stringify(pc.localDescription)));
      // listen for answer
      onValue(dbRef(database, roomPath(roomId, 'answer')), async snap => {
        const data = snap.val();
        if (!data) return;
        await pc.setRemoteDescription(data);
        appendMessage('Remote answer applied (via Firebase)', 'sys');
      });
      // listen for remote ICE (answerer)
      onChildAdded(dbRef(database, roomPath(roomId, 'candidates/answer')), async snap => {
        const c = snap.val();
        if (!c) return;
        try { await pc.addIceCandidate(c); } catch(e){console.warn('addIceCandidate failed', e);}
      });
      roomIdInput.value = roomId;
      appendMessage('Room created: ' + roomId + ' (share with partner)', 'sys');
      statusSet('room created');
    } catch (e) {
      appendMessage('Create room error: ' + e.message, 'sys');
      console.error(e);
    }
  }, 1000);
});

/* Join Room (Firebase) */
joinRoomBtn && joinRoomBtn.addEventListener('click', async () => {
  const roomId = roomIdInput.value.trim();
  if (!roomId) { appendMessage('Paste room id to join', 'sys'); return; }
  if (!currentSeedArray) { appendMessage('Pair first (generate/scan QR)', 'sys'); return; }
  initFirebase();
  await ensurePeer();
  currentRoomId = roomId;
  try {
    // read offer
    const snap = await new Promise((res, rej) => { onValue(dbRef(database, roomPath(roomId, 'offer')), s => res(s), e => rej(e)); });
    const offer = snap.val();
    if (!offer) { appendMessage('Offer not found', 'sys'); return; }
    await pc.setRemoteDescription(offer);

    // create answer
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await dbSet(dbRef(database, roomPath(roomId, 'answer')), JSON.parse(JSON.stringify(pc.localDescription)));

    // listen for remote ICE (offerer)
    onChildAdded(dbRef(database, roomPath(roomId, 'candidates/offer')), async snap => {
      const c = snap.val();
      if (!c) return;
      try { await pc.addIceCandidate(c); } catch(e){console.warn('addIceCandidate failed', e);}
    });

    // setup handler for candidates from answerer (pushed by us? not necessary)
    onChildAdded(dbRef(database, roomPath(roomId, 'candidates/answer')), async snap => {
      const c = snap.val();
      if (!c) return;
      try { await pc.addIceCandidate(c); } catch(e){console.warn('addIceCandidate failed', e);}
    });

    appendMessage('Joined room and answered. Waiting for datachannel...', 'sys');
    statusSet('joined');
  } catch (e) {
    appendMessage('Join room failed: ' + e.message, 'sys');
    console.error(e);
  }
});

/* Copy Room ID */
copyRoomBtn && copyRoomBtn.addEventListener('click', () => {
  const v = roomIdInput.value.trim();
  if (v) navigator.clipboard?.writeText(v).then(()=> appendMessage('Room ID copied', 'sys'));
});

/* Send encrypted message over data channel */
sendBtn && sendBtn.addEventListener('click', async () => {
  const txt = msgInput.value.trim();
  if (!txt) return;
  if (!currentOTP) { appendMessage('OTP not ready - pair first', 'sys'); return; }
  try {
    const cipher = encryptText(txt);
    appendMessage('SENT (cipher): ' + cipher, 'sent');
    if (dc && dc.readyState === 'open') {
      dc.send(JSON.stringify({type:'text', payload:cipher}));
    } else {
      appendMessage('No datachannel open. Use Firebase room or check connection.', 'sys');
    }
    msgInput.value = '';
  } catch (e) {
    appendMessage('Encrypt/send failed: ' + e.message, 'sys');
  }
});

/* When remote peer sends ciphertext, user can decrypt using current OTP by clicking on 'RECV (cipher)' message — simple approach:
   (For simplicity we show decrypt button not implemented in UI; user can copy cipher and press a debug decrypt later.)
*/

/* ---------- Decrypt helper (for console/test) ---------- */
window.decrypt = (b64) => {
  try {
    return decryptText(b64);
  } catch (e) { return 'decrypt error: ' + e.message; }
};

/* ---------- Encryption wrappers ---------- */
function encryptText(plain) {
  const bytes = bytesFromString(plain);
  const enc = xorBuffers(bytes, currentOTP);
  return base64FromBytes(enc);
}
function decryptText(b64) {
  const enc = bytesFromBase64(b64);
  const dec = xorBuffers(enc, currentOTP);
  return stringFromBytes(dec);
}

/* ---------- Expose debug API ---------- */
window.entangle = {
  prepareTwinFromPhrase,
  encryptText,
  decryptText
};